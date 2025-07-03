import json
import os
from PIL import Image

valid_image_ids = ["000001", "000614", "001229"]

# 입력 경로 설정
image_dir = "D:/SelfPose3d/frames_test"
txt_dir = "D:/SelfPose3d/yolov5/dance1_output/labels"  # YOLOv5 결과 텍스트 파일들 위치
output_json = "D:/SelfPose3d/pseudo_labels/pseudo_bboxes_yolov5_converted.json"

# 카테고리 (사람 1개만 있다고 가정)
categories = [{
    "supercategory": "person",
    "id": 1,
    "name": "person",
    "keypoints": [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ],
    "skeleton": [
        [16,14], [14,12], [17,15], [15,13], [12,13],
        [6,12], [7,13], [6,7], [6,8], [7,9],
        [8,10], [9,11], [2,3], [1,2], [1,3],
        [2,4], [3,5], [4,6], [5,7]
    ]
}]

images = []
annotations = []
ann_id = 0
img_idx = 0

for img_id in valid_image_ids:
    txt_file = f"{img_id}.txt"
    img_file = f"{img_id}.jpg"
    txt_path = os.path.join(txt_dir, txt_file)
    img_path = os.path.join(image_dir, img_file)

    if not os.path.exists(txt_path):
        print(f"⚠️ txt 파일 없음: {txt_path}")
        continue
    if not os.path.exists(img_path):
        print(f"⚠️ 이미지 없음: {img_path}")
        continue

    with Image.open(img_path) as im:
        width, height = im.size

    images.append({
        "file_name": img_file,
        "id": img_idx,
        "width": width,
        "height": height
    })

    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id, cx, cy, bw, bh = map(float, parts)
            x = (cx - bw / 2) * width
            y = (cy - bh / 2) * height
            w = bw * width
            h = bh * height

            annotations.append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "iscrowd": 0,
                "area": w * h
            })
            ann_id += 1

    img_idx += 1

# 최종 JSON 구성
output = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(output_json, 'w') as f:
    json.dump(output, f, indent=2)

print(f"✅ COCO JSON 저장 완료: {output_json}")
