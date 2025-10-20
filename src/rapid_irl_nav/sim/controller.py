"""Controller scaffolding for drone stabilization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class GeometricController:
    """Placeholder geometric controller.

    TODO: Integrate with a real SE(3) controller once the dynamics model is in
    place. For now we simulate a noisy tracking error that the policy must
    counteract.
    """

    kp: float = 1.0
    kd: float = 0.1
    noise_std: float = 0.05

    def compute_control(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        error = target - current
        control = self.kp * error
        if self.noise_std > 0:
            control += np.random.normal(scale=self.noise_std, size=control.shape)
        return control


__all__ = ["GeometricController"]
