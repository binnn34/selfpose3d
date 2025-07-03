import json
import pickle
import os
import numpy as np

# ✅ D 또는 E 드라이브 자동 인식
for drive_letter in ['D', 'E']:
    base_path = f"{drive_letter}:/SelfPose3D"
    if os.path.exists(base_path):
        ROOT_PATH = base_path
        break
else:
    print("❌ D:/ 또는 E:/ 드라이브에 SelfPose3D 폴더가 없습니다.")
    exit()

JSON_FILE = os.path.join(ROOT_PATH, "pseudo_labels", "s5_pseudo_kpt2d_dance1_with_soft.json")
OUT_PKL = os.path.join(ROOT_PATH, "pseudo_labels", "kpt2d_hrnet", "group_dance1_train_sub_with_soft.pkl")

with open(JSON_FILE, "r") as f:
    data = json.load(f)

# 이미지 ID -> annotation 매핑
image_to_anns = {}
for ann in data["annotations"]:
    image_to_anns.setdefault(ann["image_id"], []).append(ann)

out_data = {
    "interval": 1,
    "cam_list": ["cam5"],  # 가상 카메라
    "sequence_list": ["dance1"],
    "db": [],
}

for img in data["images"]:
    image_id = img["id"]
    anns = image_to_anns.get(image_id, [])
    
    joints_2d = []
    joints_2d_vis = []
    joints_2d_soft = []
    joints_2d_vis_soft = []

    for ann in anns:
        kp = np.array(ann["keypoints"]).reshape(-1, 3)
        joints_2d.append(kp[:, :2])
        vis = (kp[:, 2] > 0).astype(bool)
        joints_2d_vis.append(np.stack([vis, vis], axis=1))

        if "keypoints_soft" in ann:
            kp_soft = np.array(ann["keypoints_soft"]).reshape(-1, 3)
            joints_2d_soft.append(kp_soft[:, :2])
            vis_soft = (kp_soft[:, 2] > 0).astype(bool)
            joints_2d_vis_soft.append(np.stack([vis_soft, vis_soft], axis=1))

    entry = {
        "key": img["key"],
        "image": img["file_name"],
        "width": img["width"],
        "height": img["height"],
        "joints_2d": joints_2d,
        "joints_2d_vis": joints_2d_vis,
        "joints_2d_soft": joints_2d_soft,
        "joints_2d_vis_soft": joints_2d_vis_soft,
        "camera": {
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "T": [0, 0, 0],
            "fx": 1000.0,
            "fy": 1000.0,
            "cx": img["width"] / 2,
            "cy": img["height"] / 2,
            "k": [[0.0], [0.0], [0.0]],
            "p": [[0.0], [0.0]]
        }
    }
    out_data["db"].append(entry)

with open(OUT_PKL, "wb") as f:
    pickle.dump(out_data, f)

print(f"✅ Saved template PKL to {OUT_PKL}")
