# import json

# path = "D:/SelfPose3d/pseudo_labels/pseudo_bboxes_3images_detection.json"
# with open(path, 'r') as f:
#     data = json.load(f)
#     print(f"✅ 총 객체 수: {len(data)}")
#     print(f"예시 항목: {data[0]}")

import json

with open('E:/SelfPose3d/pseudo_labels/pseudo_bboxes_yolov5_converted.json') as f:
    data = json.load(f)

print(f"총 detection 개수: {len(data)}")
# print(f"예시 항목:\n{data[0]}")