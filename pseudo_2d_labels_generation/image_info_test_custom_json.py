import os
import json

# 🔧 경로 설정
frames_dir = 'frames'  # 예: 'D:/SelfPose3d/frames'
output_json_path = 'pseudo_labels/image_info_test_custom.json'

# 📦 COCO 포맷 기본 구조
coco_format = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person"
        }
    ]
}

# 🔢 8자리 숫자 정렬용 함수
def extract_frame_number(filename):
    # "frame_00000001.jpg" → "00000001" → 1
    number_part = filename.replace("frame_", "").replace(".jpg", "")
    return int(number_part)

# 📂 프레임 이미지 파일 목록 (.jpg만)
image_files = sorted([
    f for f in os.listdir(frames_dir)
    if f.startswith("frame_") and f.endswith(".jpg")
], key=extract_frame_number)

# 📐 실제 이미지 해상도 입력 (예: 1920x1080)
image_width = 1920
image_height = 1080

# 🖼 이미지 정보 입력
for idx, filename in enumerate(image_files):
    coco_format["images"].append({
        "file_name": filename,
        "height": image_height,
        "width": image_width,
        "id": idx + 1
    })

# 💾 JSON 저장
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, 'w') as f:
    json.dump(coco_format, f, indent=4)

print(f"[✅ 완료] 총 {len(image_files)}장의 이미지 정보가 포함된 COCO JSON 저장 완료: {output_json_path}")
