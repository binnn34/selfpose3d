import os
import sys
import cv2
import json
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tqdm import tqdm

# 1️⃣ D: 또는 E: 드라이브 자동 인식
for drive_letter in ['D', 'E']:
    base_path = f"{drive_letter}:/SelfPose3D"
    detectron_path = f"{drive_letter}:/detectron2"
    if os.path.exists(base_path) and os.path.exists(detectron_path):
        ROOT_PATH = base_path
        DETECTRON_PATH = detectron_path
        break
else:
    print("❌ D:/ 또는 E:/ 드라이브에 필요한 폴더가 없습니다.")
    sys.exit(1)

# 2️⃣ 경로 설정
IMG_DIR = os.path.join(ROOT_PATH, "frames")
OUT_JSON = os.path.join(ROOT_PATH, "pseudo_labels", "s2_pseudo_bboxes_test.json")
MODEL_CFG = os.path.join(DETECTRON_PATH, "configs", "COCO-Keypoints", "keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
MODEL_WEIGHT = os.path.join(ROOT_PATH, "models", "detectron2", "model_final_5ad38f.pkl")

# 3️⃣ Detectron2 설정
cfg = get_cfg()
cfg.merge_from_file(MODEL_CFG)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = MODEL_WEIGHT
cfg.MODEL.DEVICE = "cuda"  # GPU 사용 시 "cuda"

predictor = DefaultPredictor(cfg)

# 4️⃣ 이미지 읽기 및 예측
img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
out_data = {"images": [], "annotations": [], "categories": []}
ann_id = 0

for img_id, img_name in enumerate(tqdm(img_files)):
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    outputs = predictor(img)

    out_data["images"].append({
        "file_name": f"frames/{img_name}",
        "id": img_id,
        "height": height,
        "width": width,
    })

    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else []
    scores = instances.scores if instances.has("scores") else []

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.numpy()
        w, h = x2 - x1, y2 - y1
        out_data["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "score": float(score),
        })
        ann_id += 1

out_data["categories"].append({
    "id": 1,
    "name": "person",
})

# 5️⃣ JSON 저장
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(out_data, f, indent=4)

print(f"✅ Detectron2 BBox JSON 저장 완료: {OUT_JSON}")
