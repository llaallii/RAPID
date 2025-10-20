"""Autoencoder components used for reconstruction regularisation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .init import apply_initialisation


class DepthDecoder(nn.Module):
    """Lightweight decoder that mirrors :class:`DepthEncoder`."""

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )
        apply_initialisation(self)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.proj(latent)
        x = x.view(latent.shape[0], 256, 4, 4)
        return self.deconv(x)


__all__ = ["DepthDecoder"]
