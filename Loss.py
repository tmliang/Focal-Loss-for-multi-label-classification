import torch
import torch.nn as nn
import torch.nn.functional as F


def FocalLoss(y_pred, y_true, pos_weight, gamma):
    # y_pred is the logits before Sigmoid
    assert y_pred.shape == y_true.shape
    pt = torch.exp(-F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')).detach()
    sample_weight = (1 - pt) ** gamma
    return F.binary_cross_entropy_with_logits(y_pred, y_true, weight=sample_weight, pos_weight=pos_weight)
