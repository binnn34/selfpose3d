import json
import os
import numpy as np
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
OUT_FILE = os.path.join(ROOT_PATH, "pseudo_labels", "s5_pseudo_kpt2d_dance1_with_soft.json")


def process_kps(kpts, x1, y1, x2, y2, thresh=0.05):
    pose = np.array(kpts).reshape(-1, 3)
    xd = pose[:, 0]
    yd = pose[:, 1]
    score = pose[:, 2]
    score = np.where(score < thresh, 0, 2)
    num_kps = int(np.sum(score == 2))
    f_kps = []
    kps_count = 0
    for p in range(len(score)):
        if score[p] == 2 and x1 <= xd[p] <= x2 and y1 <= yd[p] <= y2:
            f_kps.extend([xd[p], yd[p], 2])
            kps_count += 1
        else:
            f_kps.extend([0, 0, 0])
    return f_kps, kps_count


def main():
    kpt_data = json.load(open(KPT_FILE))
    gt = json.load(open(GT_FILE))

    image_to_kpts = {}
    for entry in kpt_data:
        image_to_kpts.setdefault(entry["image_id"], []).append(entry)

    id2img = {img["id"]: img for img in gt["images"]}

    for ann in tqdm(gt["annotations"]):
        image_id = ann["image_id"]
        if image_id not in image_to_kpts:
            ann["delete"] = 1
            continue

        x, y, w, h = ann["bbox"]
        x1, y1 = max(0, x), max(0, y)
        img_w = id2img[image_id]["width"]
        img_h = id2img[image_id]["height"]
        x2 = min(img_w - 1, x1 + max(0, w - 1))
        y2 = min(img_h - 1, y1 + max(0, h - 1))

        best_match = None
        best_score = 0
        for kp_entry in image_to_kpts[image_id]:
            cx, cy = kp_entry["center"]
            sw, sh = kp_entry["scale"]
            pred_bbox = [cx - sw * 100, cy - sh * 100, sw * 200, sh * 200]

            px, py, pw, ph = pred_bbox
            inter_x1 = max(x1, px)
            inter_y1 = max(y1, py)
            inter_x2 = min(x1 + w, px + pw)
            inter_y2 = min(y1 + h, py + ph)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            union_area = w * h + pw * ph - inter_area
            iou = inter_area / union_area if union_area > 0 else 0

            if iou > best_score:
                best_score = iou
                best_match = kp_entry

        if best_match:
            ann["keypoints_soft"] = best_match["keypoints"]
            ann["center"] = best_match["center"]
            ann["scale"] = best_match["scale"]
            f_kps, kps_count = process_kps(best_match["keypoints"], x1, y1, x2, y2, thresh=0.05)
            ann["keypoints"] = f_kps
            ann["num_keypoints"] = kps_count
            ann["delete"] = 0
        else:
            ann["delete"] = 1

    gt["annotations"] = [ann for ann in gt["annotations"] if ann.get("delete", 0) == 0]

    with open(OUT_FILE, "w") as f:
        json.dump(gt, f, indent=4)

    print(f"✅ 완료! 결과가 저장됨: {OUT_FILE}")


if __name__ == "__main__":
    main()
