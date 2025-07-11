# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):
    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = cfg.POSE_RESNET.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            cfg.POSE_RESNET.NUM_DECONV_LAYERS,
            cfg.POSE_RESNET.NUM_DECONV_FILTERS,
            cfg.POSE_RESNET.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=cfg.POSE_RESNET.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.NETWORK.NUM_JOINTS,
            kernel_size=cfg.POSE_RESNET.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if cfg.POSE_RESNET.FINAL_CONV_KERNEL == 3 else 0,
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(
            num_filters
        ), "ERROR: num_deconv_layers is different len(num_deconv_filters)"
        assert num_layers == len(
            num_kernels
        ), "ERROR: num_deconv_layers is different len(num_deconv_filters)"

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias,
                )
            )
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, attn=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        out = self.final_layer(x)

        if attn:
            return out, x
        return out

    def init_weights(self, pretrained="", mapping=None):
        this_dir = os.path.dirname(__file__)
        pretrained = os.path.join(this_dir, "../..", pretrained)
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info("=> loading pretrained models {}".format(pretrained))

            # Remove final_layer keys if they mismatch
            model_state_dict = self.state_dict()
            for k in list(pretrained_state_dict.keys()):
                if 'final_layer' in k:
                    logger.info(f"Removing key {k} from pretrained weights due to mismatch")
                    del pretrained_state_dict[k]

            m, v = self.load_state_dict(pretrained_state_dict, strict=False)
            logger.info("=> missing keys in the backbone {}".format(m))
            logger.info("=> invalid keys in the backbone {}".format(v))
        else:
            logger.info("=> init weights from normal distribution")

        # Initialize layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3]),
}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.POSE_RESNET.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if is_train:
        model.init_weights(cfg.NETWORK.PRETRAINED, mapping=cfg.COCO_TO_PANOPTIC_MAPPING)

    return model


class PoseResAttnNet(nn.Module):
    def __init__(self, block, layers, cfg, **kwargs):
        super(PoseResAttnNet, self).__init__()
        self.backbone = PoseResNet(block, layers, cfg, **kwargs)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.backbone(x)
        out = self.sigmoid(out)
        return out
    
    def init_weights(self, pretrained="", mapping=None):
        self.backbone.init_weights(pretrained, mapping)

class PoseResAttnSharedNet(nn.Module):
    def __init__(self, block, layers, cfg, **kwargs):
        super(PoseResAttnSharedNet, self).__init__()

        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        self.final_layer = nn.Conv2d(
            in_channels=cfg.POSE_RESNET.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.NETWORK.NUM_JOINTS,
            kernel_size=cfg.POSE_RESNET.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if cfg.POSE_RESNET.FINAL_CONV_KERNEL == 3 else 0,
        )
    
    def forward(self, x):
        out = self.final_layer(x)
        # out = self.sigmoid(out)
        out = self.relu(out)
        return out
    

def get_pose_attn_net(cfg, is_train, **kwargs):
    num_layers = cfg.ATTN_NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = PoseResAttnNet(block_class, layers, cfg, **kwargs)

    if is_train:
        model.init_weights(cfg.NETWORK.PRETRAINED, mapping=cfg.COCO_TO_PANOPTIC_MAPPING)

    return model
