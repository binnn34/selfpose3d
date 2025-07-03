import cv2
import json
import os
from tqdm import tqdm
import sys

# D: 또는 E: 둘 중 존재하는 드라이브를 자동 인식
for drive_letter in ['D', 'E']:
    base_path = f"{drive_letter}:/SelfPose3D"
    if os.path.exists(base_path):
        ROOT_PATH = base_path
        break
else:
    print("❌ D:/ 또는 E:/ 드라이브에 SelfPose3D 폴더가 없습니다.")
    sys.exit(1)

# 경로 설정
IMAGE_DIR = os.path.join(ROOT_PATH, "frames")
OUT_FILE = os.path.join(ROOT_PATH, "pseudo_labels", "image_info_test.json")

def main():
    out_data = {"annotations": [], "images": [], "categories": []}

    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])
    for idx, img_name in enumerate(tqdm(image_files)):
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        out_data["images"].append({
            "file_name": f"frames/{img_name}",
            "id": idx,
            "height": height,
            "width": width,
            "key": img_name,
            "url": f"frames/{img_name}",
        })

        out_data["annotations"].append({
            "id": idx,
            "image_id": idx,
            "category_id": 1,
            "score": 1,
            "keypoints": [0] * 51,
            "iscrowd": 0,
            "area": 0,
            "bbox": [0] * 4,
        })

    out_data["categories"].append({
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
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
            [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
            [1, 3], [2, 4], [3, 5], [4, 6]
        ]
    })

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(out_data, f, indent=4)

    print(f"✅ JSON 저장 완료: {OUT_FILE}")

if __name__ == "__main__":
    main()