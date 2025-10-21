"""Core simulator and expert interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np


@dataclass
class Obs:
    depth: np.ndarray         # (1,64,64) float32
    vel: np.ndarray           # (3,) float32 -> (vx, vy, wz)
    quat: np.ndarray          # (4,) float32 [w,x,y,z]
    pos_xy: np.ndarray        # (2,) float32 (world)
    goal: np.ndarray          # (3,) float32 (gx, gy, gyaw)


class Env(Protocol):
    def reset(self, seed: Optional[int] = None) -> Obs:
        ...

    def step(self, action: np.ndarray) -> Tuple[Obs, float, bool, dict]:
        ...


class Expert(Protocol):
    def act(self, obs: Obs) -> np.ndarray:
        ...


__all__ = ["Obs", "Env", "Expert"]
