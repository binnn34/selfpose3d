# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from models import pose_resnet
from models.cuboid_proposal_net import CuboidProposalNet
from models.pose_regression_net import PoseRegressionNet
from core.loss import PerJointMSELoss
from core.loss import PerJointL1Loss


class MultiPersonPoseNet(nn.Module):
    def __init__(self, backbone, cfg):
        super(MultiPersonPoseNet, self).__init__()
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.num_joints = cfg.NETWORK.NUM_JOINTS

        self.train_only_2d = cfg.NETWORK.TRAIN_ONLY_2D
        self.backbone = backbone
        if not self.train_only_2d:
            self.root_net = CuboidProposalNet(cfg)
            self.pose_net = PoseRegressionNet(cfg)

        self.USE_GT = cfg.NETWORK.USE_GT
        self.root_id = cfg.DATASET.ROOTIDX
        self.dataset_name = cfg.DATASET.TEST_DATASET

    def forward(self, views=None, meta=None, targets_2d=None, weights_2d=None, targets_3d=None, input_heatmaps=None):
        print(f"[디버그] meta 타입: {type(meta)}, meta[0] 타입: {type(meta[0])}")
        print(f"[디버그] meta[0] keys: {meta[0].keys()}")

        if views is not None:
            all_heatmaps = []
            for view in views:
                if self.backbone is not None:
                    heatmaps = self.backbone(view)
                else:
                    heatmaps = view
                all_heatmaps.append(heatmaps)
        else:
            all_heatmaps = input_heatmaps

        root_cubes, grid_centers = self.root_net(all_heatmaps, meta)

        device = all_heatmaps[0].device
        batch_size = all_heatmaps[0].shape[0]

        # Loss 초기화
        criterion = nn.MSELoss().cuda()
        loss_2d = torch.tensor(0.0, device=device, requires_grad=True)

        # 2D loss 계산
        if targets_2d is not None:
            for t, o in zip(targets_2d, all_heatmaps):
                loss_2d = loss_2d + criterion(o, t)
            loss_2d = loss_2d / len(all_heatmaps)

        if self.train_only_2d:
            return loss_2d, all_heatmaps
        else:
            # dummy로 초기화할 때도 requires_grad 유지
            dummy = root_cubes.sum() * torch.tensor(0.0, device=device, requires_grad=True)
            loss_3d = dummy
            loss_cord = dummy.clone()

            if self.USE_GT:
                num_person = meta[0]['num_person']
                grid_centers = torch.zeros(batch_size, self.num_cand, 5, device=device)
                grid_centers[:, :, 0:3] = meta[0]['roots_3d'].float()
                grid_centers[:, :, 3] = -1.0
                for i in range(batch_size):
                    grid_centers[i, :num_person[i], 3] = torch.tensor(range(num_person[i]), device=device)
                    grid_centers[i, :num_person[i], 4] = 1.0
            else:
                root_cubes, grid_centers = self.root_net(all_heatmaps, meta)

                if targets_3d is not None and root_cubes is not None:
                    if root_cubes.shape == targets_3d.shape:
                        loss_3d = criterion(root_cubes, targets_3d)
                    else:
                        print(f"[❌ shape mismatch] root_cubes={root_cubes.shape}, targets_3d={targets_3d.shape}")
                        loss_3d = dummy

            pred = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
            pred[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)

            criterion_cord = PerJointL1Loss().cuda()
            count = 0

            for n in range(self.num_cand):
                index = (pred[:, n, 0, 3] >= 0)
                if torch.sum(index) > 0:
                    single_pose = self.pose_net(all_heatmaps, meta, grid_centers[:, n])
                    pred[:, n, :, 0:3] = single_pose

                    # 3D pose loss 계산
                    if self.training and 'joints_3d' in meta[0] and 'joints_3d_vis' in meta[0]:
                        gt_3d = meta[0]['joints_3d'].float()
                        for i in range(batch_size):
                            if pred[i, n, 0, 3] >= 0:
                                targets = gt_3d[i:i + 1, pred[i, n, 0, 3].long()]
                                weights_3d = meta[0]['joints_3d_vis'][i:i + 1, pred[i, n, 0, 3].long(), :, 0:1].float()
                                count += 1
                                loss_cord = (loss_cord * (count - 1) +
                                            criterion_cord(single_pose[i:i + 1], targets, True, weights_3d)) / count
                    del single_pose

            return root_cubes, all_heatmaps, grid_centers, loss_2d, loss_3d, loss_cord


def get_multi_person_pose_net(cfg, is_train=True):
    if cfg.BACKBONE_MODEL:
        backbone = eval(cfg.BACKBONE_MODEL + '.get_pose_net')(cfg, is_train=is_train)
    else:
        backbone = None
    model = MultiPersonPoseNet(backbone, cfg)
    return model
