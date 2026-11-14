"""Segmentation losses.

Soft Dice (per region channel, batch-aggregated):
    Dice_c = (2 * sum(p*g) + s) / (sum(p) + sum(g) + s)
    L_dice = 1 - mean_c Dice_c
Dice directly optimises the overlap metric we report and is insensitive to the
severe foreground/background imbalance (tumour << brain). Binary cross-entropy
gives smooth per-voxel gradients early in training when Dice is near zero.
Total loss = L_dice + L_bce (Isensee et al., nnU-Net, 2021 use the same pairing).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)  # aggregate over batch and space -> one Dice per region
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    dice = (2 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dw, self.bw = dice_weight, bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return self.dw * soft_dice_loss(logits, target) + self.bw * bce
