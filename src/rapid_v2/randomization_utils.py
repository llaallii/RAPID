"""Randomisation helpers shared across RAPID builders.

Phase 0 exposes deterministic seeding hooks and controller noise stubs
for the ToyWorld bridge. Isaac Sim integration layers in Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass
class RandomizationConfig:
    """Configuration for lightweight ToyWorld randomisation."""

    seed: int | None = None
    velocity_jitter: float = 0.0
    yaw_jitter: float = 0.0


def set_global_seed(seed: int | None) -> np.random.Generator:
    """Return an RNG seeded for reproducible runs."""

    return np.random.default_rng(seed)


def apply_velocity_jitter(
    sequence: Iterable[np.ndarray],
    stddev: float,
    *,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Inject Gaussian velocity noise."""

    if stddev <= 0.0:
        return [np.asarray(item, dtype=np.float32) for item in sequence]

    jittered: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for item in sequence:
        arr = np.asarray(item, dtype=np.float32)
        arr = arr + rng.normal(scale=stddev, size=arr.shape).astype(np.float32)
        jittered.append(arr)
    return jittered


def wrap_policy(policy: Callable, yaw_std: float, *, seed: int | None = None) -> Callable:
    """Return a policy wrapper that perturbs yaw commands."""

    if yaw_std <= 0.0:
        return policy

    rng = np.random.default_rng(seed)

    def wrapped(obs):
        action = np.asarray(policy(obs), dtype=np.float32)
        action[:, 1] = action[:, 1] + rng.normal(scale=yaw_std, size=action.shape[0]).astype(np.float32)
        return action

    return wrapped


__all__ = [
    "RandomizationConfig",
    "set_global_seed",
    "apply_velocity_jitter",
    "wrap_policy",
]
