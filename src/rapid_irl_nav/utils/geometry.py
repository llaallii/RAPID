"""Basic geometry helpers for navigation tasks."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def normalize_quaternion(quat: Iterable[float]) -> Tuple[float, float, float, float]:
    """Ensure a quaternion has unit length."""

    q = np.asarray(list(quat), dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm == 0:
        # TODO: decide on better fall back strategy.
        return (0.0, 0.0, 0.0, 1.0)
    q /= norm
    return tuple(float(v) for v in q)


def cylindrical_to_cartesian(radius: float, angle: float, height: float = 0.0) -> Tuple[float, float, float]:
    """Convert cylindrical coordinates to Cartesian."""

    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    return (float(x), float(y), float(height))


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi]``."""

    return float((angle + math.pi) % (2 * math.pi) - math.pi)


__all__ = ["normalize_quaternion", "cylindrical_to_cartesian", "wrap_angle"]
