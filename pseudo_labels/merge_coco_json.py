import json

# 파일 경로 설정
image_info_path = "D:/SelfPose3d/pseudo_labels/image_info_test_custom.json"
bbox_result_path = "D:/SelfPose3d/pseudo_labels/pseudo_bboxes_dance1.json"
output_path = "D:/SelfPose3d/pseudo_labels/pseudo_bboxes_dance1_full.json"

# 이미지 정보 불러오기
with open(image_info_path, "r") as f:
    image_data = json.load(f)

for img in image_data["images"]:
    img["id"] = int(img["id"])

# 바운딩 박스 결과 불러오기
with open(bbox_result_path, "r") as f:
    annotations_data = json.load(f)

for idx, anno in enumerate(annotations_data):
    anno["image_id"] = int(anno["image_id"])
    anno["id"] = idx

annotations_data.sort(key=lambda x: x["image_id"])

# COCO 포맷으로 병합
merged = {
    "images": image_data["images"],
    "annotations": annotations_data,
    "categories": [{"id": 1, "name": "person"}]
}

# 저장
with open(output_path, "w") as f:
    json.dump(merged, f)

print("✅ 병합 완료: pseudo_bboxes_dance1_full.json 생성됨")
