from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import pprint
import yaml

from yacs.config import CfgNode as CN
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

sys.path.insert(0, './lib')

import _init_paths
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)
    parser.add_argument('--bbox_json', type=str)
    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    # ✅ 1. YAML config 로딩 및 병합
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = CN(cfg_dict)

    cfg.defrost()
    if not cfg.TEST.USE_GT_BBOX:
        cfg.DATASET.TEST_JSON = args.bbox_json
    cfg.DATASET.TEST_IMGDIR = args.img_dir
    cfg.OUTPUT_DIR = args.out_dir
    cfg.freeze()

    # ✅ 2. 로거 생성
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'valid')
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # ✅ 3. CUDNN 설정
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # ✅ 4. 모델 로딩
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # ✅ 5. 손실 함수 정의
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    # ✅ 6. 데이터셋 정의
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([transforms.ToTensor(), normalize]),
        image_dir=cfg.TEST.IMG_DIR
    )

    # ✅ 7. DataLoader 구성
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # ✅ 8. 평가 실행
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)


if __name__ == '__main__':
    main()
