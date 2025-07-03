import json
import numpy as np
import argparse
from tqdm import tqdm

MAX_PEOPLE = 5

def convert(json_path, out_npy):
    data = json.load(open(json_path, 'r'))
    imgs = sorted(data['images'], key=lambda x: x['id'])
    imgid_to_anns = {}
    for ann in data['annotations']:
        imgid_to_anns.setdefault(ann['image_id'], []).append(ann)

    all_frames = []
    for img in tqdm(imgs, desc='Frames'):
        anns = imgid_to_anns.get(img['id'], [])
        people = []
        for ann in anns[:MAX_PEOPLE]:  # 여기서 최대 5명까지만 사용
            kp = np.array(ann['keypoints']).reshape(-1, 3)[:, :2]
            people.append(kp)
        if people:
            all_frames.append(np.stack(people, 0))  # (P_i, 17, 2)
        else:
            all_frames.append(np.zeros((0, 17, 2), dtype=float))

    # find max people across frames
    max_p = MAX_PEOPLE
    out = np.zeros((len(all_frames), max_p, 17, 2), dtype=float)

    for i, frame in enumerate(all_frames):
        out[i, :frame.shape[0]] = frame

    np.save(out_npy, out)
    print(f"Saved NPY shape {out.shape}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    convert(args.json, args.out)


