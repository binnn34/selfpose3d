import json, os

json_path = "D:/SelfPose3d/pseudo_labels/pseudo_bboxes_3images.json"
image_dir = "D:/SelfPose3d/frames_test"

with open(json_path, 'r') as f:
    data = json.load(f)

errors = []

# 이미지 존재 여부 확인
for img in data["images"]:
    img_path = os.path.join(image_dir, img["file_name"])
    if not os.path.exists(img_path):
        errors.append(f"❌ 이미지 없음: {img['file_name']}")

# Annotation 대응 여부 확인
img_ids = set(img["id"] for img in data["images"])
annot_img_ids = set(ann["image_id"] for ann in data["annotations"])
missing = img_ids - annot_img_ids
if missing:
    errors.append(f"⚠️ annotation 없는 이미지 id: {missing}")

# bbox 이상치 확인
for ann in data["annotations"]:
    x, y, w, h = ann["bbox"]
    if w <= 0 or h <= 0:
        errors.append(f"⚠️ 잘못된 bbox (id={ann['id']}): {ann['bbox']}")

print("\n".join(errors) if errors else "✅ 모든 항목 정상!")
