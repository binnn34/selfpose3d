import json

with open('D:/SelfPose3d/pseudo_labels/s3_pseudo_bboxes_test.json', 'r') as f:
    data = json.load(f)

if 'annotations' in data:
    data = data['annotations']

with open('D:/SelfPose3d/pseudo_labels/s3_pseudo_bboxes_test_fixed.json', 'w') as f:
    json.dump(data, f)
