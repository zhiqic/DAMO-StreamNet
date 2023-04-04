#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2022 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from yolox.models.network_blocks import BaseConv


class ContextPool(nn.Module):
    """
    Context Pooling
    """

    def __init__(
        self, 
        in_channels=(256, 512, 1024),
        feat_channels=(128, 256, 512),
        context_channels=(128, 256, 512),
        fusion_mode="concat", # "concat" or "add"
        with_project=False,
        act="silu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.context_channels = context_channels
        self.fusion_mode = fusion_mode
        self.with_project = with_project
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        if fusion_mode == "concat":
            self.jian2_feat = BaseConv(
                                in_channels=in_channels[0],
                                out_channels=feat_channels[0],
                                ksize=1,
                                stride=1,
                                act=act,
                            )

            self.jian1_feat = BaseConv(
                                in_channels=in_channels[1],
                                out_channels=feat_channels[1],
                                ksize=1,
                                stride=1,
                                act=act,
                            )

            self.jian0_feat = BaseConv(
                                in_channels=in_channels[2],
                                out_channels=feat_channels[2],
                                ksize=1,
                                stride=1,
                                act=act,
                            )
            self.jian2_context = BaseConv(
                                in_channels=in_channels[0],
                                out_channels=context_channels[0],
                                ksize=1,
                                stride=1,
                                act=act,
                            )
            self.jian1_context = BaseConv(
                                in_channels=in_channels[1],
                                out_channels=context_channels[1],
                                ksize=1,
                                stride=1,
                                act=act,
                            )

            self.jian0_context = BaseConv(
                                in_channels=in_channels[2],
                                out_channels=context_channels[2],
                                ksize=1,
                                stride=1,
                                act=act,
                            )
        if with_project:
            self.project2 = BaseConv(
                                in_channels=in_channels[0],
                                out_channels=in_channels[0],
                                ksize=1,
                                stride=1,
                                act=act,
                            )

            self.project1 = BaseConv(
                                in_channels=in_channels[1],
                                out_channels=in_channels[1],
                                ksize=1,
                                stride=1,
                                act=act,
                            )

            self.project0 = BaseConv(
                                in_channels=in_channels[2],
                                out_channels=in_channels[2],
                                ksize=1,
                                stride=1,
                                act=act,
                            )


    def forward(self, x):
        context_feats = []
        feat_num = len(x)
        for i in range(feat_num):
            feat_shape = x[i].shape
            context_feat = self.adaptive_pool(x[i])
            context_feat = torch.tile(context_feat, (1, 1, feat_shape[-2], feat_shape[-1]))
            context_feats.append(context_feat)

        # fuse
        if self.fusion_mode == "add":
            for i in range(feat_num):
                x[i] = x[i] + context_feats[i]
        elif self.fusion_mode == "concat":
            x[0] = self.jian2_feat(x[0])
            x[1] = self.jian1_feat(x[1])
            x[2] = self.jian0_feat(x[2])
            context_feats[0] = self.jian2_context(context_feats[0])
            context_feats[1] = self.jian1_context(context_feats[1])
            context_feats[2] = self.jian0_context(context_feats[2])
            for i in range(feat_num):
                x[i] = torch.cat([x[i], context_feats[i]], dim=1)
        else:
            raise Exception("Not implemented.")

        # project
        if self.with_project:
            x[0] = self.project2(x[0])
            x[1] = self.project1(x[1])
            x[2] = self.project0(x[2])

        return x


