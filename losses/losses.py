import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as PLT
import numpy as np
import cv2
from torch.autograd import Variable
            
def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class compute_transform_losses(nn.Module):
    def __init__(self, device='GPU'):
        super(compute_transform_losses, self).__init__()
        self.device = device
        self.l1_loss = L1Loss()

    def forward(self, outputs, retransform_output):
        loss = F.l1_loss(outputs, retransform_output, size_average=False)
        return loss


class compute_losses(nn.Module):
    def __init__(self, device='GPU'):
        super(compute_losses, self).__init__()
        self.device = device
        self.L1Loss = nn.L1Loss()

    def forward(self, opt, weight, inputs, outputs, features, retransform_features, label_features, label_retransform_features):
        losses = {}
        self.opt= opt
        type = opt.type
        losses["topview_loss"] = 0
        losses["transform_topview_loss"] = 0
        losses["transform_loss"] = 0

        focal_weight = 10
        
        losses["topview_loss"] = focal_weight*self.compute_topview_focal_loss(
            outputs["topview"],
            inputs[type])
        losses["transform_topview_loss"] = focal_weight*self.compute_topview_focal_loss(
            outputs["transform_topview"],
            inputs[type])
            
        losses["transform_loss"] = self.compute_transform_losses(
            features,
            retransform_features)
        losses["label_retransform_loss"] = self.compute_transform_losses(
            label_features,
            label_retransform_features)
        losses["loss"] = losses["topview_loss"] + 0.001 * losses["transform_loss"] \
                         + losses["transform_topview_loss"] + 0.001 *losses["label_retransform_loss"] 

        return losses

    def compute_transform_losses(self, outputs, retransform_output):
        loss = self.L1Loss(outputs, retransform_output)
        return loss

    def compute_topview_focal_loss(self, outputs, true_top_view):
        generated_top_view = outputs
        true_top_view = torch.squeeze(true_top_view.long())
        loss = FocalLoss()
        output = loss(generated_top_view, true_top_view)
        return output.mean()
