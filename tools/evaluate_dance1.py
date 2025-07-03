import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib')))

import torch
from dataset.dance1 import dance1
from core.config import config, update_config
import argparse
import models
import numpy as np
from tqdm import tqdm
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True, type=str)
parser.add_argument('--model', required=True, type=str)
parser.add_argument('--out', default='results_dance1_preds.pkl', type=str)
args = parser.parse_args()
update_config(args.cfg)

# 데이터셋 로딩
dataset = dance1(config, image_set='test', is_train=False, transform=None)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# 모델 로딩
model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(config, is_train=False)
state_dict = torch.load(args.model, map_location='cpu')
model.load_state_dict(state_dict['state_dict'] if 'state_dict' in state_dict else state_dict)
model.eval()

# 추론
results = []
for inputs, _, _, _, meta, _ in tqdm(loader):
    with torch.no_grad():
        outputs = model(inputs)
        results.append({
            'key': meta[0]['key'],
            'pred_3d': outputs.cpu().numpy()
        })

# 저장
with open(args.out, 'wb') as f:
    pickle.dump(results, f)

print(f"결과가 {args.out}로 저장되었습니다.")
