import json
import os
from PIL import Image

image_ids = ["000001", "000614", "001229"]
image_dir = "D:/SelfPose3d/frames_test"
txt_dir = "D:/SelfPose3d/yolov5/dance1_output/labels"
output_path = "D:/SelfPose3d/pseudo_labels/pseudo_bboxes_yolov5_converted2.json"

results = []
img_id_map = {img_id: idx for idx, img_id in enumerate(image_ids)}

for img_id in image_ids:
    txt_path = os.path.join(txt_dir, f"{img_id}.txt")
    img_path = os.path.join(image_dir, f"{img_id}.jpg")

    if not os.path.exists(txt_path) or not os.path.exists(img_path):
        continue

    with Image.open(img_path) as im:
        width, height = im.size

    with open(txt_path, 'r') as f:
        for line in f.readlines():
            cls, cx, cy, bw, bh = map(float, line.strip().split())
            x = (cx - bw / 2) * width
            y = (cy - bh / 2) * height
            w = bw * width
            h = bh * height

            results.append({
                "image_id": img_id_map[img_id],
                "category_id": 1,
                "bbox": [x, y, w, h],
                "score": 1.0
            })

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ bbox 결과 (results array) 저장 완료: {output_path}")
