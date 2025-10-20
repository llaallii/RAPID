"""Utility functions for Soft Actor-Critic."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def compute_target_value(reward: torch.Tensor, discount: float, next_value: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
    """Bellman backup used for critic targets."""

    return reward + discount * (1.0 - done) * next_value


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging between source and target parameters."""

    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.lerp_(source_param.data, tau)


def entropy_loss(log_alpha: torch.Tensor, log_pi: torch.Tensor, target_entropy: float) -> torch.Tensor:
    """Temperature adjustment loss for SAC."""

    return -(log_alpha.exp() * (log_pi + target_entropy).detach()).mean()


__all__ = ["compute_target_value", "soft_update", "entropy_loss"]
