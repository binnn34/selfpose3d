import json
import os
import numpy as np
import cv2
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

GT_TRAIN_JSON = os.path.join(ROOT_PATH, "pseudo_labels", "image_info_test.json")
DT_TRAIN_JSON = os.path.join(ROOT_PATH, "pseudo_labels", "s2_pseudo_bboxes_test.json")
OUT_PSEUDO_BBOX = os.path.join(ROOT_PATH, "pseudo_labels", "s3_pseudo_bboxes_test.json")



def update_anns(anno_dict):
    for index, ann in tqdm(enumerate(anno_dict), desc="🔧 Updating annotations"):
        ann["id"] = index + 1
        ann["num_keypoints"] = 0
        # ann["keypoints_krcnn"] = deepcopy(ann.get("keypoints", [0]*51))  # 안전하게 처리
        ann["keypoints"] = [0] * 51
        ann["area"] = ann["bbox"][2] * ann["bbox"][3]
        ann["iscrowd"] = 0
    return anno_dict


def create_pseudo_bboxes():
    print("loading files...")


    with open(GT_TRAIN_JSON, "r") as f:
        gt_train = json.load(f)

    with open(DT_TRAIN_JSON, "r") as f:
        dt_data = json.load(f)
        # ✅ "annotations" 키가 있는 경우 처리
        if "annotations" in dt_data:
            dt_train = dt_data["annotations"]
        else:
            dt_train = dt_data  # 리스트로 되어 있는 경우

    print("files loaded...")
    print(f"✅ 총 annotations 수: {len(dt_train)}")

    dt_train = [g for g in dt_train if g.get("category_id", 1) == 1 and g.get("score", 1.0) > 0.7]
    print(f"📏 필터링 후 annotations 수: {len(dt_train)}")

    print("procesing each annotations")
    dt_train = update_anns(dt_train)
    print("annotations procesing finished")

    print("finished processing")
    gt_train["annotations"] = dt_train

    os.makedirs(os.path.dirname(OUT_PSEUDO_BBOX), exist_ok=True)
    with open(OUT_PSEUDO_BBOX, "w") as f:
        json.dump(gt_train, f)

    print(f"✅ Pseudo BBox 저장 완료: {OUT_PSEUDO_BBOX}")


if __name__ == "__main__":
    create_pseudo_bboxes()
