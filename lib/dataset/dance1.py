from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import logging
import os
import copy

from torch.utils.data import Dataset
from dataset.JointsDataset import JointsDataset
import torch
from torchvision import transforms as tv_transforms


class dance1(JointsDataset):
    def __init__(self, cfg, image_set, is_train=True, transform=None):
        super().__init__(cfg, image_set, is_train, transform)

        self.dataset_root = cfg.DATASET.ROOT
        self.db_file = cfg.DATASET.TRAIN_2D_FILE if is_train else cfg.DATASET.TEST_2D_FILE

        with open(self.db_file, 'rb') as f:
            info = pickle.load(f)

        self.sequence_list = info.get('sequence_list', [])
        self.cam_list = info.get('cam_list', [])
        self._interval = info.get('interval', 1)
        self.db = info['db']
        self.db_size = len(self.db)

    def __len__(self):
        return self.db_size

    def __getitem__(self, idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input_2d = self.db[idx]['joints_2d']
        input_2d = np.stack(input_2d, axis=0)
        num_person = len(input_2d)

        input_vis = self.db[idx]['joints_2d_vis']
        input_vis = np.stack(input_vis, axis=0)

        print("joints shape:", input_2d.shape)

        # print("Camera type:", type(self.db[idx]['camera']))
        # print("Camera content:", self.db[idx]['camera'])

        num_joints = input_2d.shape[0]

        centers = self.db[idx].get('centers', [])
        center = centers[0] if centers else [0.0, 0.0]
        scales = self.db[idx].get('scales', [])
        scale = scales[0] if scales else [1.0, 1.0]

        # centers = self.db[idx].get('centers', [])
        # center = centers[0] if isinstance(centers, list) and len(centers) > 0 else [0.0, 0.0]
        # if isinstance(center[0], list) or isinstance(center[0], np.ndarray):
        #     center = center[0]  # centers = [[[x, y]]] 같은 구조일 때 평탄화

        meta = {
                'key': self.db[idx]['key'],
            'image': self.db[idx]['image'],
            'camera': {
                'R': torch.tensor(self.db[idx]['camera']['R'], dtype=torch.float32),
                'T': torch.tensor(self.db[idx]['camera']['T'], dtype=torch.float32),
                'fx': float(self.db[idx]['camera']['fx']),
                'fy': float(self.db[idx]['camera']['fy']),
                'cx': float(self.db[idx]['camera']['cx']),
                'cy': float(self.db[idx]['camera']['cy']),
                'k': torch.tensor([c for sub in self.db[idx]['camera']['k'] for c in sub], dtype=torch.float32),
                'p': torch.tensor([c for sub in self.db[idx]['camera']['p'] for c in sub], dtype=torch.float32),
            },
            'bbox': self.db[idx].get('bboxes', []),
            'center': [np.array(center, dtype=np.float32)],
            'scale': [np.array(scale, dtype=np.float32)],
            'rotation': [0.0],

            'joints_2d': input_2d,
            'joints_2d_vis': input_vis,
            'num_person': num_person,
        }

        target_weight = np.array(input_vis).astype(float)

        transform_fn = self.transform if self.transform else tv_transforms.ToTensor()
        input_2d = [transform_fn(img) if isinstance(img, np.ndarray) else img for img in input_2d]

        num_joints = 15
        heatmap_size = (80, 80)
        cube_size = (80, 80, 20)
        num_views = 1
        inputs = torch.zeros(num_views, num_joints, *heatmap_size)

        targets_3d = torch.zeros(1, cube_size[0], cube_size[1], cube_size[2], device=device)
        targets_2d = torch.zeros(num_views, num_joints, cube_size[0], cube_size[1])
        target_weight = torch.ones(num_views, num_joints, *cube_size, 1)
        root_cam = torch.zeros(3)

        return inputs, targets_2d, target_weight, targets_3d, [meta], root_cam

    def evaluate(self, preds, scores, output_dir, *args, **kwargs):
        print(f"Evaluation not implemented. preds shape: {len(preds)}, scores: {len(scores)}")
        return {}