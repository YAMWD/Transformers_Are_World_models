from __future__ import annotations

import torch
import torch.nn as nn


class ConvDecoder64(nn.Module):
    """Small conv-transpose decoder: z -> 64x64 RGB in [0,1]."""

    def __init__(self, z_dim: int, base_channels: int = 256) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.base_channels = int(base_channels)
        self.fc = nn.Linear(self.z_dim, self.base_channels * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.base_channels, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, z_dim)
        batch = z.shape[0]
        x = self.fc(z).view(batch, self.base_channels, 4, 4)
        return self.deconv(x)

