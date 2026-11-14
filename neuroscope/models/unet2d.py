"""2D U-Net with spatial dropout for Monte Carlo uncertainty estimation.

Architecture: 4-level encoder/decoder (Ronneberger et al., 2015) with
BatchNorm + ReLU double-conv blocks, 2x2 max-pooling, transposed-conv upsampling
and skip concatenations. Dropout2d is placed in the bottleneck and the two
deepest decoder blocks; at inference `enable_mc_dropout()` keeps those layers
stochastic so repeated forward passes approximate a Bayesian posterior over
weights (Gal & Ghahramani, 2016).

Input : [B, 4, H, W]   (t1, t1ce, t2, flair; missing modality = all-zero channel)
Output: [B, 3, H, W]   logits for regions (WT, TC, ET), sigmoid activation
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4, dropout)
        self.bottleneck = DoubleConv(c4, c5, dropout)
        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.dec4 = DoubleConv(c5, c4, dropout)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = DoubleConv(c4, c3, dropout)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c2, c1)
        self.head = nn.Conv2d(c1, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)

    def enable_mc_dropout(self) -> None:
        """Eval mode for BatchNorm, train mode for Dropout layers only."""
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: dict) -> UNet2D:
    return UNet2D(
        in_channels=4,
        out_channels=3,
        base=int(cfg.get("base_channels", 32)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
