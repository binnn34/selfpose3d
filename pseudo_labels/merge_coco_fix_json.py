# filter_bbox_to_gt.py
import json

gt_path = "D:/SelfPose3d/pseudo_labels/image_info_test_custom.json"
bbox_path = "D:/SelfPose3d/pseudo_labels/pseudo_bboxes_dance1_final.json"
output_path = "D:/SelfPose3d/pseudo_labels/pseudo_bboxes_dance1_final_filtered.json"

with open(gt_path, "r") as f:
    gt_data = json.load(f)
with open(bbox_path, "r") as f:
    bbox_data = json.load(f)

gt_ids = {img["id"] for img in gt_data["images"]}
filtered_annos = [a for a in bbox_data["annotations"] if a["image_id"] in gt_ids]
filtered_imgs = [img for img in bbox_data["images"] if img["id"] in gt_ids]

merged = {
    "images": filtered_imgs,
    "annotations": filtered_annos,
    "categories": [{"id": 1, "name": "person"}]
}

with open(output_path, "w") as f:
    json.dump(merged, f)

print("✅ 필터링된 bbox JSON 저장 완료:", output_path)
