#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
# Copyright (c) DAMO Academy, Alibaba Group and its affiliates.

import torch
import torch.nn as nn

from exps.model.tal_head import TALHead
from exps.model.dfp_pafpn_long import DFPPAFPNLONG
from exps.model.dfp_pafpn_short import DFPPAFPNSHORT

from yolox.models.network_blocks import BaseConv


class YOLOXLONGSHORTCP(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(
        self, 
        long_backbone=None, 
        short_backbone=None, 
        backbone_neck=None,
        head=None, 
        merge_form="add", 
        in_channels=[256, 512, 1024], 
        width=1.0, 
        act="silu",
        with_short_cut=False,
        short_cfg=None,
        long_cfg=None,
    ):
        """Summary
        
        Args:
            long_backbone (None, optional): Description
            short_backbone (None, optional): Description
            head (None, optional): Description
            merge_form (str, optional): "add" or "concat" or "pure_concat" or "long_fusion"
            in_channels (list, optional): Description
        """
        super().__init__()
        if short_backbone is None:
            short_backbone = DFPPAFPNSHORTV2()
        if head is None:
            head = TALHead(20)

        self.long_backbone = long_backbone
        self.short_backbone = short_backbone
        self.backbone = backbone_neck
        self.head = head
        self.merge_form = merge_form
        self.in_channels = in_channels
        self.with_short_cut = with_short_cut
        self.short_frame_num = short_cfg["frame_num"]
        self.long_frame_num = long_cfg["frame_num"] if self.long_backbone is not None else 0
        if merge_form == "concat":
            self.jian2 = BaseConv(
                        in_channels=int(in_channels[0] * width),
                        out_channels=int(in_channels[0] * width) // 2,
                        ksize=1,
                        stride=1,
                        act=act,
                    )

            self.jian1 = BaseConv(
                        in_channels=int(in_channels[1] * width),
                        out_channels=int(in_channels[1] * width) // 2,
                        ksize=1,
                        stride=1,
                        act=act,
                    )

            self.jian0 = BaseConv(
                        in_channels=int(in_channels[2] * width),
                        out_channels=int(in_channels[2] * width) // 2,
                        ksize=1,
                        stride=1,
                        act=act,
                    )
        elif merge_form == "long_fusion":
            assert long_cfg is not None and "out_channels" in long_cfg
            self.jian2 = BaseConv(
                        in_channels=sum([x[0][0]*x[1] for x in long_cfg["out_channels"]]),
                        out_channels=int(in_channels[0] * width) // 2,
                        ksize=1,
                        stride=1,
                        act=act,
                    )

            self.jian1 = BaseConv(
                        in_channels=sum([x[0][1]*x[1] for x in long_cfg["out_channels"]]),
                        out_channels=int(in_channels[1] * width) // 2,
                        ksize=1,
                        stride=1,
                        act=act,
                    )

            self.jian0 = BaseConv(
                        in_channels=sum([x[0][2]*x[1] for x in long_cfg["out_channels"]]),
                        out_channels=int(in_channels[2] * width) // 2,
                        ksize=1,
                        stride=1,
                        act=act,
                    )

    def forward(self, x, targets=None, buffer=None, mode='off_pipe'):
        # fpn output content features of [dark3, dark4, dark5]
        assert mode in ['off_pipe', 'on_pipe']

        if mode == 'off_pipe':
            cur_bs = x[0].shape[0]
            x_cat = torch.cat([x[0][:, i*3:(i+1)*3, ...] for i in range(self.short_frame_num)] + [x[1][:, i*3:(i+1)*3, ...] for i in range(self.long_frame_num)], dim=0)

            pan_outs = self.backbone(x_cat)
            short_inputs = [[pan_outs[j][i*cur_bs:(i+1)*cur_bs, ...] for j in range(len(pan_outs))] for i in range(self.short_frame_num)]
            long_inputs = [[pan_outs[j][(i+self.short_frame_num)*cur_bs:(i+self.short_frame_num+1)*cur_bs, ...] for j in range(len(pan_outs))] for i in range(self.long_frame_num)]

            short_fpn_outs, rurrent_pan_outs = self.short_backbone(short_inputs, buffer=buffer, mode='off_pipe')
            long_fpn_outs = self.long_backbone(long_inputs, buffer=buffer, mode='off_pipe') if self.long_backbone is not None else None
            if not self.with_short_cut:
                if self.long_backbone is None:
                    fpn_outs = short_fpn_outs
                else:
                    if self.merge_form == "add":
                        fpn_outs = [x + y for x, y in zip(short_fpn_outs, long_fpn_outs)]
                    elif self.merge_form == "concat":
                        fpn_outs_2 = torch.cat([self.jian2(short_fpn_outs[0]), self.jian2(long_fpn_outs[0])], dim=1)
                        fpn_outs_1 = torch.cat([self.jian1(short_fpn_outs[1]), self.jian1(long_fpn_outs[1])], dim=1)
                        fpn_outs_0 = torch.cat([self.jian0(short_fpn_outs[2]), self.jian0(long_fpn_outs[2])], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    elif self.merge_form == "pure_concat":
                        fpn_outs_2 = torch.cat([short_fpn_outs[0], long_fpn_outs[0]], dim=1)
                        fpn_outs_1 = torch.cat([short_fpn_outs[1], long_fpn_outs[1]], dim=1)
                        fpn_outs_0 = torch.cat([short_fpn_outs[2], long_fpn_outs[2]], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    elif self.merge_form == "long_fusion":
                        fpn_outs_2 = torch.cat([short_fpn_outs[0], self.jian2(long_fpn_outs[0])], dim=1)
                        fpn_outs_1 = torch.cat([short_fpn_outs[1], self.jian1(long_fpn_outs[1])], dim=1)
                        fpn_outs_0 = torch.cat([short_fpn_outs[2], self.jian0(long_fpn_outs[2])], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    else:
                        raise Exception(f'merge_form must be in ["add", "concat"]')
            else:
                if self.long_backbone is None:
                    fpn_outs = [x + y for x, y in zip(short_fpn_outs, rurrent_pan_outs)]
                else:
                    if self.merge_form == "add":
                        fpn_outs = [x + y + z for x, y, z in zip(short_fpn_outs, long_fpn_outs, rurrent_pan_outs)]
                    elif self.merge_form == "concat":
                        fpn_outs_2 = torch.cat([self.jian2(short_fpn_outs[0]), self.jian2(long_fpn_outs[0])], dim=1)
                        fpn_outs_1 = torch.cat([self.jian1(short_fpn_outs[1]), self.jian1(long_fpn_outs[1])], dim=1)
                        fpn_outs_0 = torch.cat([self.jian0(short_fpn_outs[2]), self.jian0(long_fpn_outs[2])], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                        fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                    elif self.merge_form == "pure_concat":
                        fpn_outs_2 = torch.cat([short_fpn_outs[0], long_fpn_outs[0]], dim=1)
                        fpn_outs_1 = torch.cat([short_fpn_outs[1], long_fpn_outs[1]], dim=1)
                        fpn_outs_0 = torch.cat([short_fpn_outs[2], long_fpn_outs[2]], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                        fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                    elif self.merge_form == "long_fusion":
                        fpn_outs_2 = torch.cat([short_fpn_outs[0], self.jian2(long_fpn_outs[0])], dim=1)
                        fpn_outs_1 = torch.cat([short_fpn_outs[1], self.jian1(long_fpn_outs[1])], dim=1)
                        fpn_outs_0 = torch.cat([short_fpn_outs[2], self.jian0(long_fpn_outs[2])], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                        fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                    else:
                        raise Exception(f'merge_form must be in ["add", "concat"]')

            if self.training:
                assert targets is not None
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                    fpn_outs, targets, x
                )
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
            else:
                outputs = self.head(fpn_outs)

            return outputs
        elif mode == 'on_pipe':
            fpn_outs, buffer_ = self.backbone(x,  buffer=buffer, mode='on_pipe')
            outputs = self.head(fpn_outs)
            
            return outputs, buffer_




