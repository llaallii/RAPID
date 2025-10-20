"""Weight initialisation utilities."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    """Apply orthogonal init to linear layers."""

    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def delta_orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    """Apply delta orthogonal init to convolution kernels."""

    if isinstance(module, nn.Conv2d):
        if module.weight.data.ndim < 3:
            return
        nn.init.delta_orthogonal_(module.weight, gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def apply_initialisation(module: nn.Module) -> nn.Module:
    """Convenience to apply default initialisation rules."""

    module.apply(orthogonal_init)
    module.apply(delta_orthogonal_init)
    return module


__all__ = ["orthogonal_init", "delta_orthogonal_init", "apply_initialisation"]
