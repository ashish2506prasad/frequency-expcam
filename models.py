"""Mask generators ported from AM0/AM6 (CSAE and UNetCSAE)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSAE(nn.Module):
    """Stride-2 conv autoencoder that emits mask logits at 224×224."""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec4 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        z = F.relu(self.enc4(x))
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        return self.dec4(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetCSAE(nn.Module):
    """Lightweight U-Net mask generator used in the paper (AM0 cell 5)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        bn = self.bottleneck(self.pool4(d4))
        u4 = self.dec4(torch.cat([self.up4(bn), d4], dim=1))
        u3 = self.dec3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.dec2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.dec1(torch.cat([self.up1(u2), d1], dim=1))
        return self.out_conv(u1)


def _fourier_conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.ReLU(inplace=True),
    )


class _FEnc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = _fourier_conv_block(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        features = self.conv(x)
        return self.pool(features), features


class _FDec(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = _fourier_conv_block(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class FourierMaskNet(nn.Module):
    """WaveX shared polar Fourier mask: U-Net → GAP → Linear → (R×A) sigmoid."""

    def __init__(
        self,
        in_channels: int = 3,
        radial_bins: int = 16,
        angular_bins: int = 16,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.radial_bins = int(radial_bins)
        self.angular_bins = int(angular_bins)
        self.temperature = float(temperature)
        self.enc1 = _FEnc(in_channels, 32)
        self.enc2 = _FEnc(32, 64)
        self.enc3 = _FEnc(64, 128)
        self.enc4 = _FEnc(128, 256)
        self.bottleneck = _fourier_conv_block(256, 256)
        self.dec4 = _FDec(256, 256, 128)
        self.dec3 = _FDec(128, 128, 64)
        self.dec2 = _FDec(64, 64, 32)
        self.dec1 = _FDec(32, 32, 32)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.polar_proj = nn.Linear(32, self.radial_bins * self.angular_bins)
        nn.init.xavier_uniform_(self.polar_proj.weight, gain=0.1)
        nn.init.zeros_(self.polar_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p1, s1 = self.enc1(x)
        p2, s2 = self.enc2(p1)
        p3, s3 = self.enc3(p2)
        p4, s4 = self.enc4(p3)
        b = self.bottleneck(p4)
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        bsz = x.shape[0]
        pooled = self.global_pool(d1).view(bsz, -1)
        logits = self.polar_proj(pooled).view(bsz, 1, self.radial_bins, self.angular_bins)
        temp = max(self.temperature, 1e-6)
        return torch.sigmoid(logits / temp)


class TinyFourierMaskNet(nn.Module):
    """One-skip polar mask net: 16/32 channels, then GAP → (R×A) sigmoid."""

    def __init__(
        self,
        in_channels: int = 3,
        radial_bins: int = 16,
        angular_bins: int = 16,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.radial_bins = int(radial_bins)
        self.angular_bins = int(angular_bins)
        self.temperature = float(temperature)
        self.enc1 = _FEnc(in_channels, 16)
        self.bottleneck = _fourier_conv_block(16, 32)
        self.dec1 = _FDec(32, 16, 16)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.polar_proj = nn.Linear(16, self.radial_bins * self.angular_bins)
        nn.init.xavier_uniform_(self.polar_proj.weight, gain=0.1)
        nn.init.zeros_(self.polar_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p1, s1 = self.enc1(x)
        b = self.bottleneck(p1)
        d1 = self.dec1(b, s1)
        bsz = x.shape[0]
        pooled = self.global_pool(d1).view(bsz, -1)
        logits = self.polar_proj(pooled).view(bsz, 1, self.radial_bins, self.angular_bins)
        temp = max(self.temperature, 1e-6)
        return torch.sigmoid(logits / temp)


class SpectrumMask(nn.Module):
    """Learnable 1-channel mask on the rFFT half-spectrum, shared across RGB."""

    def __init__(self, height: int, width: int, init: float = 0.0):
        super().__init__()
        coeff_w = int(width) // 2 + 1
        self.logits = nn.Parameter(torch.full((1, 1, int(height), coeff_w), float(init)))

    def forward(self, x=None):
        return self.logits


def last_shared_layer(mask_net: nn.Module) -> nn.Module:
    """Last layer used by all decoder branches (for GradNorm-style balancing)."""
    if isinstance(mask_net, SpectrumMask):
        return mask_net
    if isinstance(mask_net, (FourierMaskNet, TinyFourierMaskNet)):
        return mask_net.dec1
    if isinstance(mask_net, UNetCSAE):
        return mask_net.dec1
    return mask_net.dec3
