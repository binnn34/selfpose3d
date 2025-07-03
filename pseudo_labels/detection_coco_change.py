import json

# 원본 detection 결과 파일 로드
with open("pseudo_bboxes_3images_detection.json", "r") as f:
    detections = json.load(f)

# 이미지 정보는 GT에서 가져온다고 가정 (pseudo_bboxes_3images.json)
with open("pseudo_bboxes_3images.json", "r") as f:
    gt = json.load(f)

# 변환
converted = {
    "images": gt["images"],  # 동일한 이미지 정보 사용
    "annotations": [],
    "categories": gt["categories"]
}

for idx, det in enumerate(detections):
    ann = {
        "id": idx + 1,
        "image_id": det["image_id"],
        "category_id": det["category_id"],
        "bbox": det["bbox"],
        "area": det["bbox"][2] * det["bbox"][3],  # bbox width * height
        "iscrowd": 0,
        "score": det["score"]  # score는 COCO Eval에는 필요 없음 (선택 사항)
    }
    converted["annotations"].append(ann)

# 저장
with open("pseudo_bboxes_3images_detection_coco_format.json", "w") as f:
    json.dump(converted, f, indent=2)
