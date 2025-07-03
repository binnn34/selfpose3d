import ast
import json

# 1. 원본 detection JSON 파일 경로 설정
input_json_path = "D:/SelfPose3d/pseudo_labels/pseudo_bboxes_3images_detection.json"

# 2. JSON 파일 로딩
with open(input_json_path, 'r', encoding='utf-8') as f:
    data_list = json.load(f)

# 3. 문자열 → dict 변환 + bbox 필터링
cleaned_list = []
for ann in data_list:
    if isinstance(ann, str):
        try:
            ann_dict = ast.literal_eval(ann)
        except Exception:
            continue
    else:
        ann_dict = ann

    # bbox 필터링 조건
    if 'bbox' in ann_dict and len(ann_dict['bbox']) >= 4:
        if ann_dict['bbox'][2] > 0 and ann_dict['bbox'][3] > 0:
            cleaned_list.append(ann_dict)

# 4. 새 JSON 파일로 저장
final_output_path = "D:/SelfPose3d/pseudo_labels/pseudo_bboxes_cleaned.json"
with open(final_output_path, 'w', encoding='utf-8') as f:
    json.dump(cleaned_list, f, indent=2)

print("✅ Saved cleaned bbox JSON to:", final_output_path)
