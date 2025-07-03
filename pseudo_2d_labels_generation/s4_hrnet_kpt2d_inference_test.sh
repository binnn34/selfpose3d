#!/usr/bin/env bash

# build HR-Net from https://github.com/HRNet/HRNet-Human-Pose-Estimation

# ✅ D: 또는 E: 드라이브 자동 인식
if [ -d "D:/SelfPose3d" ]; then
  ROOT_PATH="D:/SelfPose3d"
elif [ -d "E:/SelfPose3d" ]; then
  ROOT_PATH="E:/SelfPose3d"
else
  echo "❌ D: 또는 E: 드라이브에 SelfPose3d 폴더가 없습니다."
  exit 1
fi

HRNET_PATH="${ROOT_PATH}/HRNet-Human-Pose-Estimation"
cd "${HRNET_PATH}" || exit

CONFIG_FILE=experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml
MODEL_WEIGHTS="${HRNET_PATH}/models/pose_hrnet_w48_384x288.pth"


TEST_JSON="${ROOT_PATH}/pseudo_labels/s3_pseudo_bboxes_test.json"
TEST_IMG_DIR="${ROOT_PATH}/frames"
OUTPUT_DIR="${ROOT_PATH}/pseudo_labels/kpt2d_hrnet/test"

mkdir -p "${OUTPUT_DIR}"
LOG="${OUTPUT_DIR}/eval.log"
python tools/test.py \
  --cfg "${CONFIG_FILE}" \
  TEST.MODEL_FILE "${MODEL_WEIGHTS}" \
  TEST.USE_GT_BBOX True \
  DATASET.TEST_JSON "${TEST_JSON}" \
  DATASET.TEST_IMGDIR "${TEST_IMG_DIR}" \
  OUTPUT_DIR "${OUTPUT_DIR}" 2>&1 | tee "${LOG}"