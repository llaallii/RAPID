"""Neural encoders for observation modalities."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .init import apply_initialisation


class DepthEncoder(nn.Module):
    """Convolutional encoder for depth images."""

    def __init__(self, latent_dim: int = 128, freeze_for_actor: bool = False) -> None:
        super().__init__()
        self.freeze_for_actor = freeze_for_actor
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(256 * 4 * 4, latent_dim)
        apply_initialisation(self)

    def forward(self, depth: torch.Tensor, *, actor_mode: bool = False) -> torch.Tensor:
        depth = depth.float()
        if depth.ndim == 4:
            x = depth
        elif depth.ndim == 3:
            x = depth.unsqueeze(0)
        else:
            raise ValueError(f"Expected depth tensor of shape (B, 1, 64, 64), got {depth.shape}")

        if actor_mode and self.freeze_for_actor:
            with torch.no_grad():
                features = self.backbone(x)
                flat = features.flatten(start_dim=1)
                return self.head(flat)

        features = self.backbone(x)
        flat = features.flatten(start_dim=1)
        latent = self.head(flat)
        return latent


__all__ = ["DepthEncoder"]
