'''
Project: SelfPose3d
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import random
import cv2
import json
import numpy as np
import os

# ✅ D 또는 E 드라이브 자동 인식
for drive_letter in ['D', 'E']:
    base_path = f"{drive_letter}:/SelfPose3D"
    if os.path.exists(base_path):
        ROOT_PATH = base_path
        break
else:
    print("❌ D:/ 또는 E:/ 드라이브에 SelfPose3D 폴더가 없습니다.")
    exit()

random.seed(0)
GT_FILE = os.path.join(ROOT_PATH, "pseudo_labels", "s5_pseudo_kpt2d_dance1.json")
IMGDIR = os.path.join(ROOT_PATH, "frames")
OUTPUT_DIR = os.path.join(ROOT_PATH, "pseudo_labels", "kpt2d_hrnet", "dance1", "vis")

COCO_COLOR_LIST = [
    "#e6194b",  # red,       nose
    "#3cb44b",  # green      left_eye
    "#ffe119",  # yellow     right_eye
    "#0082c8",  # blue       left_ear
    "#f58231",  # orange     right_ear
    "#911eb4",  # purple     left_shoulder
    "#46f0f0",  # cyan       right_shoulder
    "#f032e6",  # magenta    left_elbow
    "#d2f53c",  # lime       right_elbow
    "#fabebe",  # pink       left_wrist
    "#008080",  # teal       right_wrist
    "#e6beff",  # lavender   left_hip
    "#aa6e28",  # brown      right_hip
    "#fffac8",  # beige      left_knee
    "#800000",  # maroon     right_knee
    "#aaffc3",  # mint       left_ankle
    "#808000",  # olive      right_ankle
]
coco_colors_skeleton = [
    "m",
    "m",
    "g",
    "g",
    "r",
    "m",
    "g",
    "r",
    "m",
    "g",
    "m",
    "g",
    "r",
    "m",
    "g",
    "m",
    "g",
    "m",
    "g",
]
coco_pairs = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
]

def fixed_bright_colors():
    return [
        [207, 73, 179],
        [53, 84, 209],
        [31, 252, 54],
        [203, 173, 34],
        [229, 18, 115],
        [236, 31, 98],
        [50, 195, 222],
        [169, 52, 199],
        [44, 69, 172],
        [231, 4, 80],
        [191, 197, 33],
        [46, 194, 180],
        [35, 228, 69],
        [217, 211, 25],
        [253, 10, 48],
        [170, 213, 80],
        [206, 77, 13],
        [197, 178, 11],
        [204, 163, 32],
        [143, 222, 64],
        [45, 208, 109],
        [67, 185, 44],
        [91, 68, 230],
        [249, 246, 20],
        [75, 202, 201],
        [11, 202, 193],
        [221, 75, 180],
        [241, 16, 142],  
        [126, 9, 231],
        [40, 210, 122],
        [10, 136, 205],
        [38, 230, 105],
        [193, 97, 26],
        [203, 18, 101],
        [42, 173, 94],
        [222, 45, 135],
        [33, 184, 48],
        [121, 49, 195],
        [31, 39, 226],
        [204, 48, 143],
        [220, 47, 192],
        [223, 220, 73],
        [46, 177, 170],
        [17, 245, 161],
        [159, 51, 107],
        [10, 39, 205],
        [50, 237, 101],
        [116, 35, 171],
        [213, 76, 76],
        [88, 203, 47],
        [202, 205, 14],
        [100, 233, 4],
        [227, 34, 192],
        [21, 79, 239],
        [30, 198, 36],
        [140, 38, 240],
        [97, 26, 215],
        [48, 122, 225],
        [158, 51, 196],
        [11, 212, 45],
        [190, 173, 39],
        [34, 185, 34],
        [98, 58, 219],
        [147, 233, 66],
        [44, 239, 69],
        [192, 177, 38],
        [53, 233, 53],
        [41, 222, 44],
        [228, 70, 120],
        [221, 153, 58],
        [131, 19, 222],
        [203, 27, 140],
        [170, 72, 54],
        [182, 58, 173],
        [194, 218, 84],
        [233, 34, 30],
        [100, 173, 37],
        [72, 92, 227],
        [216, 90, 183],
        [66, 215, 125],
        [183, 63, 41],
        [228, 29, 54],
        [29, 221, 125],
        [172, 12, 207],
        [20, 228, 205],
        [16, 228, 121],
        [210, 21, 198],
        [80, 135, 206],
        [196, 165, 27],
    ]



def draw_2d_keypoints(image, pt2d, use_color):

    colors_skeleton = coco_colors_skeleton
    pairs = coco_pairs

    for idx in range(len(colors_skeleton)):
        color = use_color
        pair = pairs[idx]
        pt1, sc1 = tuple(pt2d[pair[0], :].astype(int)[0:2]), pt2d[pair[0]][2]
        pt2, sc2 = tuple(pt2d[pair[1], :].astype(int)[0:2]), pt2d[pair[1]][2]
        if sc1 > 0 and sc2 > 0:
            cv2.line(image, pt1, pt2, color, 4, cv2.LINE_AA)

    # draw keypoints
    for idx in range(len(COCO_COLOR_LIST)):
        pt, sc = tuple(pt2d[idx, :].astype(int)[0:2]), pt2d[idx][2]
        if sc > 0:
            c = COCO_COLOR_LIST[idx].lstrip("#")
            c = tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))
            c = (c[2], c[1], c[0]) # rgb->bgr
            cv2.circle(image, pt, 4, c, 3, cv2.LINE_AA)
            cv2.circle(image, pt, 5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_anns_v2(draw, anns, rand_colors):
    thickness = 2
    for ii, ann in enumerate(anns):
        x, y, w, h = [int(a) for a in ann["bbox"]]
        color = rand_colors[ii]
        cv2.rectangle(draw, (x, y), (x + w, y + h), color, thickness)
        if "keypoints_soft" in ann:
            kpts_2d = np.array(ann["keypoints_soft"]).reshape(17, 3)
        else:
            kpts_2d = np.array(ann["keypoints"]).reshape(17, 3)
        #print(ann["num_keypoints"], ann["num_keypoints_krcnn"])
        draw_2d_keypoints(draw, kpts_2d, color)
    return draw


def main():
    

    with open(GT_FILE) as f:
        gt = json.load(f)

    dets = {k["id"]: [] for k in gt["images"]}
    id2filename = {k["id"]: k["file_name"] for k in gt["images"]}
    rand_colors = fixed_bright_colors()
    for d in gt['annotations']:
        dets[d["image_id"]].append(d)

    for k in range(len(list(dets))):
        anns = dets[k]
        img_filename = os.path.basename(id2filename[k])  # 경로 제거
        file_path = os.path.join(IMGDIR, img_filename)
        print(f"[INFO] Processing image ID: {k}, file: {file_path}")

        if not os.path.exists(file_path):
            print(f"[WARNING] File not found: {file_path}")
            continue

        img = cv2.imread(file_path)
        if img is None:
            print(f"[WARNING] Failed to load image: {file_path}")
            continue

        anns = sorted(anns, key=lambda k: k["bbox"][0])
        img = draw_anns_v2(img, anns, rand_colors)
        out_path = os.path.join(OUTPUT_DIR, img_filename)
        cv2.imwrite(out_path, img)
        print(f"[INFO] Saved visualized image to: {out_path}")

        # if os.path.exists(file_name):
        #     img = cv2.imread(file_name)
        #     anns = sorted(anns, key=lambda k: k["bbox"][0])
        #     img = draw_anns_v2(img, anns, rand_colors)
        #     output_dir = os.path.join(ROOT_PATH, "pseudo_labels", "kpt2d_hrnet", "dance1", "vis")
        #     os.makedirs(output_dir, exist_ok=True)
        #     cv2.imwrite(os.path.join(output_dir, os.path.basename(file_name)), img)

if __name__ == "__main__":
    main()        