import torch
from torch import nn
import torchvision as tv


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        intersection = torch.sum(y_true * y_pred)
        denom = torch.sum(y_true) + torch.sum(y_pred)
        dice_loss = 1 - (2 * intersection + 1e-7) / (denom + 1e-7)

        return dice_loss


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return tv.ops.sigmoid_focal_loss(
            y_pred,
            y_true,
            alpha=-1,  # no reweighting based on class frequency
            reduction="mean"
        )
