'''
Project: SelfPose3d
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import pickle
import cv2
import json
import os
from tqdm import tqdm

IMG_DIR = "D:/SelfPose3d/frames_test"
IMG_LIST = ["000001.jpg", "000614.jpg", "001229.jpg"]
OUT_FILE = "D:/SelfPose3d/pseudo_labels/pseudo_bboxes_3images.json"



def main():
    out_data = {"annotations": [], "images": [], "categories": []}
    for idx, fname in enumerate(IMG_LIST):
        path = os.path.join(IMG_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Cannot read image {path}")
            continue
        height, width, _ = img.shape
        out_data["images"].append({
            "file_name": fname,
            "id": idx,
            "height": height,
            "width": width
        })
        out_data["annotations"].append({
            "id": idx,
            "image_id": idx,
            "category_id": 1,
            "bbox": [0, 0, width, height],  # placeholder bbox
            "score": 1.0,
            "iscrowd": 0,
            "area": width * height,
            "keypoints": [0]*51  # placeholder
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
                [16, 14], [14, 12], [17, 15], [15, 13],
                [12, 13], [6, 12], [7, 13], [6, 7],
                [6, 8], [7, 9], [8, 10], [9, 11],
                [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
                [4, 6], [5, 7]
        ]
    })
    with open(OUT_FILE, "w") as f:
        json.dump(out_data, f, indent=4)

if __name__ == "__main__":
    main()