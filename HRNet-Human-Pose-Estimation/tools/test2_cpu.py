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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ✅ 1. args 정의
    args = parse_args()

    # ✅ 2. YAML 직접 UTF-8로 읽기
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = CN(cfg_dict)

    # ✅ 4. args로부터 동적 경로 설정
    cfg.defrost()
    if not cfg.TEST.USE_GT_BBOX:
        cfg.DATASET.TEST_JSON = args.bbox_json
    cfg.DATASET.TEST_IMGDIR = args.img_dir
    cfg.OUTPUT_DIR = args.out_dir
    cfg.freeze()

    # ✅ 5. 로그 및 출력 경로 설정
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # ✅ 6. CUDNN 설정
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # ✅ 7. 모델 로딩
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    ).to(device)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location=device), strict=False)
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file, map_location=device))

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model = model.to(device)

    # ✅ 8. 손실 함수
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).to(device)

    # ✅ 9. 데이터셋 로딩
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        image_dir=cfg.TEST.IMG_DIR
    )

    # ✅ 10. DataLoader 설정
    if torch.cuda.is_available() and len(cfg.GPUS) > 0:
        batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
        pin_memory = True
    else:
        batch_size = cfg.TEST.BATCH_SIZE_PER_GPU
        pin_memory = False

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=pin_memory
    )

    # ✅ 11. 평가 실행
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)


if __name__ == '__main__':
    main()
