from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import argparse
import os
import pprint
import logging
import json

import _init_paths
from core.config import config, update_config
from core.function import train_3d, validate_3d
from utils.utils import create_logger, save_checkpoint, load_checkpoint
from utils.utils import load_backbone_panoptic
from dataset.dance1 import dance1
import models
import random
import numpy as np

def custom_collate_fn(batch):
    inputs, targets_2d, target_weight, targets_3d, metas, root_cams = zip(*batch)
    return (
        torch.stack(inputs),
        torch.stack(targets_2d),
        torch.stack(target_weight),
        torch.stack(targets_3d),
        [item for sublist in metas for item in sublist],  # flatten
        torch.stack(root_cams)
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def get_optimizer(model):
    lr = config.TRAIN.LR
    with_root_net = not config.NETWORK.USE_GT
    freeze_root_net = config.NETWORK.FREEZE_ROOTNET
    train_backbone = config.NETWORK.TRAIN_BACKBONE

    if train_backbone:
        if model.module.backbone is not None:
            for params in model.module.backbone.parameters():
                params.requires_grad = True
    else:
        if model.module.backbone is not None:
            for params in model.module.backbone.parameters():
                params.requires_grad = False

    if not config.NETWORK.TRAIN_ONLY_2D:
        if not config.NETWORK.TRAIN_ONLY_ROOTNET:
            for params in model.module.pose_net.parameters():
                params.requires_grad = True
        if with_root_net:
            if freeze_root_net:
                for params in model.module.root_net.parameters():
                    params.requires_grad = False
            else:
                for params in model.module.root_net.parameters():
                    params.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)
    return model, optimizer


def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, "train")

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    gpus = [int(i) for i in config.GPUS.split(",")]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = dance1(
        config,
        image_set="train",
        is_train=True,
        transform=None  # dance1은 이미지가 아니라 pose vector를 사용하므로 transform 생략
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=0,
        pin_memory=False,
        collate_fn=custom_collate_fn
    )

    test_dataset = dance1(
        config,
        image_set="test",
        is_train=False,
        transform=None
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=custom_collate_fn
    )

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval("models." + config.MODEL + ".get_multi_person_pose_net")(config, is_train=True)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    model, optimizer = get_optimizer(model)
    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH
    last_epoch = -1
    best_precision = 0

    if config.NETWORK.PRETRAINED_BACKBONE:
        if config.NETWORK.PRETRAINED_BACKBONE_PSEUDOGT:
            if model.module.backbone is not None:
                model.module.backbone.init_weights(config.NETWORK.PRETRAINED_BACKBONE)
            else:
                print("Backbone이 None이므로 init_weights 생략")
        else:
            model = load_backbone_panoptic(model, config.NETWORK.PRETRAINED_BACKBONE)

    if config.NETWORK.INIT_ROOTNET:
        st_dict = {
            k.replace("root_net.", ""): v
            for k, v in torch.load(config.NETWORK.INIT_ROOTNET).items()
            if "root_net" in k
        }
        model.module.root_net.load_state_dict(st_dict, strict=True)

    if config.NETWORK.INIT_ALL:
        st_dict = torch.load(config.NETWORK.INIT_ALL)
        model.module.load_state_dict(st_dict, strict=True)

    if config.TRAIN.RESUME:
        start_epoch, model, optimizer, best_precision, last_epoch = load_checkpoint(
            model, optimizer, final_output_dir
        )

    writer_dict = {
        "writer": SummaryWriter(log_dir=tb_log_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR, last_epoch=last_epoch
    )

    for epoch in range(start_epoch, end_epoch):
        logger.info("Epoch: {}".format(epoch))
        logger.info("learning rate: {}".format(lr_scheduler.get_last_lr()))

        train_3d(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict)
        lr_scheduler.step()

        if config.NETWORK.TRAIN_ONLY_2D:
            precision = None
        else:
            precision = validate_3d(config, model, test_loader, epoch, final_output_dir, with_ssv=False)

        best_model = precision is not None and precision > best_precision
        if best_model:
            best_precision = precision

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.module.state_dict(),
                "precision": best_precision,
                "optimizer": optimizer.state_dict(),
            },
            best_model,
            final_output_dir,
        )

    final_model_state_file = os.path.join(final_output_dir, "dance1.pth.tar")
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict["writer"].close()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()