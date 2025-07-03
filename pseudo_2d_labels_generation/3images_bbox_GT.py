import json
import os

# 3장 이미지 이름과 ID 매핑
image_list = [
    ("000001.jpg", 1),
    ("000614.jpg", 2),
    ("001229.jpg", 3)
]

# bounding box 직접 수동으로 기입 (예시 bbox: [x, y, w, h])
# → 실제 bbox 정보는 yolov5나 detectron2 출력 결과를 참고해 적절히 수정하세요!
bbox_list = [
    # frame_000001
    [
        [1451.9984640000002, 1020.0006000000001, 237.000192, 586.9994399999999],  
        [2041.001856, 1056.99924, 226.00012800000002, 561.9996],
        [1782.9987840000001, 977.00148, 237.000192, 690.99912],
        [1073.0000639999998, 1000.00116, 301.999872, 696.9996],
        [2461.998912, 1003.00032, 294.999936, 720.9993599999999],
    ],
    
    # frame_000614
    [
        [2571.0003840000004, 1062.00072, 343.99987200000004, 648.0],
        [2211.00096, 988.0002000000001, 444.9984, 813.00024],
        [1423.0003199999999, 1035.00072, 460.00128, 751.99968],
        [1820.99904, 1021.99968, 472.00128, 859.99968],
        [1011.0009600000002, 1029.00024, 429.99935999999997, 681.00048],
    ],

    # frame_001229
    [
        [3018.0015359999998, 901.9998, 346.00012799999996, 908.99928],
        [549.000192, 955.9997999999998, 216.99993600000002, 835.9999200000001],
        [1847.9986560000002, 956.00088, 256.00012799999996, 857.9995200000001],
        [1178.9996159999998, 960.00012, 244.00012800000002, 846.00072],
        [2457.999936, 948.0002400000001, 296.99980800000003, 868.00032],
    ]
]

output_json = {
    "images": [],
    "annotations": [],
    "categories": [{
        "id": 1,
        "name": "person",
        "supercategory": "person"
    }]
}

ann_id = 1
for i, (filename, img_id) in enumerate(image_list):
    output_json["images"].append({
        "id": img_id,
        "file_name": filename,
        "width": 3840,
        "height": 2160
    })
    for bbox in bbox_list[i]:
        output_json["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        })
        ann_id += 1

# 저장 경로
save_path = "D:/SelfPose3d/pseudo_labels/pseudo_bboxes_3images.json"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 파일 저장
with open(save_path, 'w') as f:
    json.dump(output_json, f, indent=2)

print("✅ 3장용 COCO bbox JSON 생성 완료:", save_path)
