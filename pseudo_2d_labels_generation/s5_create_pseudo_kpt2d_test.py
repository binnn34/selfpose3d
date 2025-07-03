# import json
# import cv2
# import numpy as np
# import os
# from tqdm import tqdm
# from copy import deepcopy

# # ✅ D 또는 E 드라이브 자동 인식
# for drive_letter in ['D', 'E']:
#     base_path = f"{drive_letter}:/SelfPose3D"
#     if os.path.exists(base_path):
#         ROOT_PATH = base_path
#         break
# else:
#     print("❌ D:/ 또는 E:/ 드라이브에 SelfPose3D 폴더가 없습니다.")
#     exit()

# KPT_FILE = os.path.join(ROOT_PATH, "pseudo_labels", "kpt2d_hrnet", "dance1", "coco", "pose_hrnet", "w48_384x288_adam_lr1e-3_test2", "results", "keypoints_val_results_0.json")
# GT_FILE = os.path.join(ROOT_PATH, "pseudo_labels", "s3_pseudo_bboxes_test.json")
# OUT_FILE = os.path.join(ROOT_PATH, "pseudo_labels", "s5_pseudo_kpt2d_dance1.json")

# def process_kps(kpts):
#     pose = np.array(kpts).reshape(-1, 3)
#     f_kps = []
#     num_kps = 0
#     for px, py, sc in pose:
#         f_kps.extend([px, py, 2])  # 무조건 conf=2로 설정
#         num_kps += 1
#     return f_kps, num_kps

# def main():
#     _kpt = json.load(open(KPT_FILE))
#     gt = json.load(open(GT_FILE))

#     # image_id 기준으로 bbox 할당
#     image_to_anns = {}
#     for ann in gt["annotations"]:
#         image_to_anns.setdefault(ann["image_id"], []).append(ann)

#     for entry in tqdm(_kpt):
#         image_id = entry["image_id"]
#         if image_id not in image_to_anns:
#             continue
#         anns = image_to_anns[image_id]
#         # GT BBox 수와 HRNet 수가 같다고 가정하고 index 매칭
#         for ann, kp in zip(anns, image_to_anns[image_id]):
#             ann["keypoints_soft"] = kp["keypoints"]
#             ann["center"] = entry["center"]
#             ann["scale"] = entry["scale"]
#             f_kps, kps_count = process_kps(entry["keypoints"])
#             ann["keypoints"] = f_kps
#             ann["num_keypoints"] = kps_count
#             ann["delete"] = 0

#     gt["annotations"] = [ann for ann in gt["annotations"] if ann.get("delete", 0) == 0]

#     with open(OUT_FILE, "w") as f:
#         json.dump(gt, f, indent=4)
#     print("finish adding keypoint detections")

# if __name__ == "__main__":
#     main()

import json
import cv2
import numpy as np
import os
from tqdm import tqdm
from copy import deepcopy

# ✅ D 또는 E 드라이브 자동 인식
for drive_letter in ['D', 'E']:
    base_path = f"{drive_letter}:/SelfPose3D"
    if os.path.exists(base_path):
        ROOT_PATH = base_path
        break
else:
    print("❌ D:/ 또는 E:/ 드라이브에 SelfPose3D 폴더가 없습니다.")
    exit()

KPT_FILE = os.path.join(ROOT_PATH, "pseudo_labels", "kpt2d_hrnet", "dance1", "coco", "pose_hrnet", "w48_384x288_adam_lr1e-3_test2", "results", "keypoints_val_results_0.json")
GT_FILE = os.path.join(ROOT_PATH, "pseudo_labels", "s3_pseudo_bboxes_test.json")
OUT_FILE = os.path.join(ROOT_PATH, "pseudo_labels", "s5_pseudo_kpt2d_dance1_02.json")

def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def process_kps(kpts):
    pose = np.array(kpts).reshape(-1, 3)
    f_kps = []
    num_kps = 0
    for px, py, sc in pose:
        f_kps.extend([px, py, 2])  # conf=2
        num_kps += 1
    return f_kps, num_kps

def main():
    kpt_data = json.load(open(KPT_FILE))
    gt = json.load(open(GT_FILE))

    image_to_kpts = {}
    for entry in kpt_data:
        image_to_kpts.setdefault(entry["image_id"], []).append(entry)

    for ann in tqdm(gt["annotations"]):
        image_id = ann["image_id"]
        if image_id not in image_to_kpts:
            ann["delete"] = 1
            continue

        x, y, w, h = ann["bbox"]
        best_match = None
        best_score = 0
        for kp_entry in image_to_kpts[image_id]:
            cx, cy = kp_entry["center"]
            sw, sh = kp_entry["scale"]
            pred_bbox = [cx - sw*100, cy - sh*100, sw*200, sh*200]
            score = iou([x, y, w, h], pred_bbox)
            if score > best_score:
                best_score = score
                best_match = kp_entry

        if best_match:
            ann["center"] = best_match["center"]
            ann["scale"] = best_match["scale"]
            f_kps, kps_count = process_kps(best_match["keypoints"])
            ann["keypoints"] = f_kps
            ann["num_keypoints"] = kps_count
            ann["delete"] = 0
        else:
            ann["delete"] = 1

    gt["annotations"] = [ann for ann in gt["annotations"] if ann.get("delete", 0) == 0]

    with open(OUT_FILE, "w") as f:
        json.dump(gt, f, indent=4)
    print("finish adding keypoint detections")

if __name__ == "__main__":
    main()
