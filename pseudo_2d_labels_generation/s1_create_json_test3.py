import os
import json
import cv2
from tqdm import tqdm

# 이미지 프레임 폴더 경로
IMAGE_DIR = "D:/SelfPose3d/frames"

# 출력 JSON 파일 경로
OUT_FILE = "D:/SelfPose3d/pseudo_labels/image_info_test.json"

def main():
    out_data = {
        "annotations": [],
        "images": [],
        "categories": [
            {
                "supercategory": "person",
                "id": 1,
                "name": "person",
                "keypoints": [
                    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                    "right_knee", "left_ankle", "right_ankle"
                ],
                "skeleton": [
                    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
                    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
                    [1, 3], [2, 4], [3, 5], [4, 6]
                ]
            }
        ]
    }

    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png"))])

    for i, img_file in enumerate(tqdm(image_files)):
        img_path = os.path.join(IMAGE_DIR, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading {img_path}")
            continue
        height, width, _ = img.shape

        out_data["images"].append({
            "file_name": img_file,
            "id": i,
            "height": height,
            "width": width,
        })

        out_data["annotations"].append({
            "id": i,
            "image_id": i,
            "category_id": 1,
            "score": 1,
            "keypoints": [0] * 51,  # 17 keypoints x [x, y, v]
            "iscrowd": 0,
            "area": 0,
            "bbox": [0, 0, 0, 0]
        })

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(out_data, f, indent=4)

if __name__ == "__main__":
    main()