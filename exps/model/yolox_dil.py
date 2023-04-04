#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
# Copyright (c) DAMO Academy, Alibaba Group and its affiliates.
import torch
import torch.nn as nn

from exps.model.tal_head import TALHead
from exps.model.dfp_pafpn import DFPPAFPN


class YOLOXDIL(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone_s=None, head_s=None, backbone_t=None, head_t=None, coef_cfg=None, dil_loc="head", still_teacher=True):
        super().__init__()

        self.backbone = backbone_s # student model
        self.head = head_s
        self.backbone_t = backbone_t # teacher model
        self.head_t = head_t
        self._freeze_teacher_model()
        self._set_eval_teacher_model()
        self.dil_loss = nn.MSELoss(reduction='sum')
        coef_cfg = coef_cfg if coef_cfg is not None else dict()
        self.dil_loss_coef = coef_cfg.get("dil_loss_coef", 1.)
        self.det_loss_coef = coef_cfg.get("det_loss_coef", 1.)
        self.reg_coef = coef_cfg.get("reg_coef", 1.)
        self.cls_coef = coef_cfg.get("cls_coef", 1.)
        self.obj_coef = coef_cfg.get("obj_coef", 1.)
        self.dil_loc = dil_loc
        self.still_teacher = still_teacher
        assert dil_loc in ("head", "neck")

    def forward(self, x, targets=None, buffer=None, mode='off_pipe'):
        # fpn output content features of [dark3, dark4, dark5]
        assert mode in ['off_pipe', 'on_pipe']

        if mode == 'off_pipe':
            if self.training:
                if self.dil_loc == "head":
                    if self.still_teacher:
                        fpn_outs = self.backbone(x[:,:6,:,:], buffer=buffer, mode='off_pipe')
                        fpn_outs_t = self.backbone_t(x[:,-3:,:,:], buffer=buffer, mode='off_pipe')
                    else:
                        fpn_outs = self.backbone(x[:,:6,:,:], buffer=buffer, mode='off_pipe')
                        fpn_outs_t = self.backbone_t(x[:,:6,:,:], buffer=buffer, mode='off_pipe')
                    assert targets is not None
                    (loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg), reg_outputs, obj_outputs, cls_outputs = self.head(
                        fpn_outs, targets, x
                    )
                    reg_outputs_t, obj_outputs_t, cls_outputs_t = self.head_t(fpn_outs_t)
                    reg_dil_losses = []
                    cls_dil_losses = []
                    obj_dil_losses = []
                    for i in range(len(reg_outputs)):
                        cur_loss = self.dil_loss(reg_outputs[i], reg_outputs_t[i])
                        reg_dil_losses.append(cur_loss)
                    reg_dil_loss = self.reg_coef * (torch.sum(torch.stack(reg_dil_losses)) / self._get_tensors_numel(reg_outputs))
                    for i in range(len(cls_outputs)):
                        cur_loss = self.dil_loss(cls_outputs[i], cls_outputs_t[i])
                        cls_dil_losses.append(cur_loss)
                    cls_dil_loss = self.cls_coef * (torch.sum(torch.stack(cls_dil_losses)) / self._get_tensors_numel(cls_outputs))
                    for i in range(len(obj_outputs)):
                        cur_loss = self.dil_loss(obj_outputs[i], obj_outputs_t[i])
                        obj_dil_losses.append(cur_loss)
                    obj_dil_loss = self.obj_coef * (torch.sum(torch.stack(obj_dil_losses)) / self._get_tensors_numel(obj_outputs))

                    dil_loss = self.dil_loss_coef * (reg_dil_loss + cls_dil_loss + obj_dil_loss)
                    loss = self.det_loss_coef * loss
                    total_loss = dil_loss + loss
                    
                    outputs = {
                        "total_loss": total_loss,
                        "det_loss": loss,
                        "iou_loss": iou_loss,
                        "l1_loss": l1_loss,
                        "conf_loss": conf_loss,
                        "cls_loss": cls_loss,
                        "dil_loss": dil_loss,
                        "reg_dil_loss": reg_dil_loss,
                        "cls_dil_loss": cls_dil_loss,
                        "obj_dil_loss": obj_dil_loss,
                        "num_fg": num_fg,
                    }
                elif self.dil_loc == "neck":
                    fpn_outs = self.backbone(x[:,:6,:,:], buffer=buffer, mode='off_pipe')
                    fpn_outs_t = self.backbone_t(x[:,-3:,:,:], buffer=buffer, mode='off_pipe')
                    assert targets is not None
                    (loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg), reg_outputs, obj_outputs, cls_outputs = self.head(
                        fpn_outs, targets, x
                    )
                    neck_dil_losses = []
                    for i in range(len(fpn_outs)):
                        cur_loss = self.dil_loss(fpn_outs[i], fpn_outs_t[i])
                        neck_dil_losses.append(cur_loss)
                    neck_dil_loss = torch.sum(torch.stack(neck_dil_losses)) / self._get_tensors_numel(fpn_outs)

                    dil_loss = self.dil_loss_coef * neck_dil_loss
                    loss = self.det_loss_coef * loss
                    total_loss = dil_loss + loss
                    
                    outputs = {
                        "total_loss": total_loss,
                        "det_loss": loss,
                        "iou_loss": iou_loss,
                        "l1_loss": l1_loss,
                        "conf_loss": conf_loss,
                        "cls_loss": cls_loss,
                        "dil_loss": dil_loss,
                        "num_fg": num_fg,
                    }
                else:
                    raise Exception('dil_loc must be in ("head", "neck")')
            else:
                fpn_outs = self.backbone(x, buffer=buffer, mode='off_pipe')
                outputs = self.head(fpn_outs)
                # TODOlcy
                # fpn_outs = self.backbone_t(x[:,-3:,:,:], buffer=buffer, mode='off_pipe')
                # outputs = self.head_t(fpn_outs)

            return outputs
        elif mode == 'on_pipe':
            fpn_outs, buffer_ = self.backbone(x,  buffer=buffer, mode='on_pipe')
            outputs = self.head(fpn_outs)
            
            return outputs, buffer_

    def _freeze_teacher_model(self):
        for name, param in self.backbone_t.named_parameters():
            param.requires_grad = False
        for name, param in self.head_t.named_parameters():
            param.requires_grad = False

    def _set_eval_teacher_model(self):
        self.backbone_t.eval()
        self.head_t.eval()

    def _get_tensors_numel(self, tensors):
        num = 0
        for t in tensors:
            num += torch.numel(t)
        return num