# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils.cameras as cameras
from utils.transforms import get_affine_transform as get_transform
from utils.transforms import affine_transform_pts_cuda as do_transform


class ProjectLayer(nn.Module):
    def __init__(self, cfg):
        super(ProjectLayer, self).__init__()

        self.img_size = cfg.NETWORK.IMAGE_SIZE
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE

    def compute_grid(self, boxSize, boxCenter, nBins, device=None):
        if isinstance(boxSize, int) or isinstance(boxSize, float):
            boxSize = [boxSize, boxSize, boxSize]
        if isinstance(nBins, int):
            nBins = [nBins, nBins, nBins]

        grid1Dx = torch.linspace(-boxSize[0] / 2, boxSize[0] / 2, nBins[0], device=device)
        grid1Dy = torch.linspace(-boxSize[1] / 2, boxSize[1] / 2, nBins[1], device=device)
        grid1Dz = torch.linspace(-boxSize[2] / 2, boxSize[2] / 2, nBins[2], device=device)
        gridx, gridy, gridz = torch.meshgrid(
            grid1Dx + boxCenter[0],
            grid1Dy + boxCenter[1],
            grid1Dz + boxCenter[2],
        )
        gridx = gridx.contiguous().view(-1, 1)
        gridy = gridy.contiguous().view(-1, 1)
        gridz = gridz.contiguous().view(-1, 1)
        grid = torch.cat([gridx, gridy, gridz], dim=1)
        return grid

    def get_voxel(self, heatmaps, meta, grid_size, grid_center, cube_size, flip_xcoords=None):
        device = heatmaps[0].device
        batch_size = heatmaps[0].shape[0]
        num_joints = heatmaps[0].shape[1]
        nbins = cube_size[0] * cube_size[1] * cube_size[2]
        n = len(heatmaps)
        cubes = torch.zeros(batch_size, num_joints, 1, nbins, n, device=device)
        # h, w = heatmaps[0].shape[2], heatmaps[0].shape[3]
        w, h = self.heatmap_size
        grids = torch.zeros(batch_size, nbins, 3, device=device)
        bounding = torch.zeros(batch_size, 1, 1, nbins, n, device=device)
        for i in range(batch_size):
            if len(grid_center[0]) == 3 or grid_center[i][3] >= 0:
                # This part of the code can be optimized because the projection operation is time-consuming.
                # If the camera locations always keep the same, the grids and sample_grids are repeated across frames
                # and can be computed only one time.
                if len(grid_center) == 1:
                    grid = self.compute_grid(grid_size, grid_center[0], cube_size, device=device)
                else:
                    grid = self.compute_grid(grid_size, grid_center[i], cube_size, device=device)
                grids[i:i + 1] = grid
                for c in range(n):
                    center = meta[i]['center'][c] if c < len(meta[i]['center']) else meta[i]['center'][0]
                    if not isinstance(center, (list, tuple, np.ndarray)):
                        center = [center, center]  # float이면 리스트로 변환
                    width, height = center[0] * 2, center[1] * 2

                    scale = meta[c]['scale'][i]
                    rotation = meta[c]['rotation'][i]


                    trans = torch.as_tensor(
                        get_transform(center, scale, rotation, self.img_size),
                        dtype=torch.float,
                        device=device)
                    cam = meta[c]['camera']

                    xy = cameras.project_pose(grid, cam)

                    bounding[i, 0, 0, :, c] = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < width) & (
                                xy[:, 1] < height)
                    xy = torch.clamp(xy, -1.0, max(width, height))
                    xy = do_transform(xy, trans)
                    if flip_xcoords is not None and flip_xcoords[i]:
                        xy[..., 0] = self.img_size[0] - xy[..., 0]
                    xy = xy * torch.tensor(
                        [w, h], dtype=torch.float, device=device) / torch.tensor(
                        self.img_size, dtype=torch.float, device=device)
                    sample_grid = xy / torch.tensor(
                        [w - 1, h - 1], dtype=torch.float,
                        device=device) * 2.0 - 1.0
                    sample_grid = torch.clamp(sample_grid.view(1, 1, nbins, 2), -1.1, 1.1)

                    # if pytorch version < 1.3.0, align_corners=True should be omitted.
                    cubes[i:i + 1, :, :, :, c] += F.grid_sample(heatmaps[c][i:i + 1, :, :, :], sample_grid, align_corners=True)

        # cubes = cubes.mean(dim=-1)
        cubes = torch.sum(torch.mul(cubes, bounding), dim=-1) / (torch.sum(bounding, dim=-1) + 1e-6)
        cubes = cubes.clone()
        cubes[cubes != cubes] = 0.0
        cubes = cubes.clamp(0.0, 1.0)

        cubes = cubes.view(batch_size, num_joints, cube_size[0], cube_size[1], cube_size[2])  ##
        return cubes, grids

    def forward(self, heatmaps, meta, grid_size, grid_center, cube_size, flip_xcoords=None):
        cubes, grids = self.get_voxel(heatmaps, meta, grid_size, grid_center, cube_size, flip_xcoords=flip_xcoords)
        return cubes, grids