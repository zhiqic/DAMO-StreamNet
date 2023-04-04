#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Copyright (c) DAMO Academy, Alibaba Group and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F

from exps.model.darknet import CSPDarknet
from exps.model.damo_yolo import GiraffeNeckV2
from yolox.models.network_blocks import BaseConv, CSPLayer, DWConv


class BACKBONENECKV2(nn.Module):
    """
    use GiraffeNeckV2 as neck
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        neck_cfg=None,
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, 
                                   width, 
                                   depthwise=depthwise, 
                                   act=act)
        self.in_features = in_features
        self.in_channels = in_channels

        self.neck = GiraffeNeckV2(**neck_cfg)

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """


        #  backbone
        rurrent_out_features = self.backbone(input)
        rurrent_features = [rurrent_out_features[f] for f in self.in_features]
        neck_outputs = self.neck(rurrent_features)

        return neck_outputs


