import json

with open("pseudo_bboxes_dance1_full.json") as f:
    gt = json.load(f)
gt_ids = set(img["id"] for img in gt["images"])

with open("pseudo_bboxes_annotations_only.json") as f:
    det = json.load(f)
det_ids = set(d["image_id"] for d in det)

print("총 GT 이미지 수:", len(gt_ids))
print("총 Detection 이미지 수:", len(det_ids))
print("겹치는 ID 수:", len(gt_ids & det_ids))  # 이게 0이면 결과 없음
