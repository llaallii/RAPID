"""Transform utilities for preprocessing observations."""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np


class DepthNormalize:
    """Normalize depth values into ``[0, 1]`` or standardized units."""

    def __init__(self, *, min_val: float = 0.0, max_val: float = 10.0) -> None:
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)
        normalized = (depth - self.min_val) / max(self.max_val - self.min_val, 1e-5)
        normalized = np.clip(normalized, 0.0, 1.0)
        return normalized


class FrameStack:
    """Temporal stacking for depth frames.

    TODO: extend to handle heterogeneous observation dictionaries (state + depth).
    """

    def __init__(self, num_frames: int = 4) -> None:
        self.num_frames = num_frames
        self._buffer: Deque[np.ndarray] = deque(maxlen=num_frames)

    def reset(self) -> None:
        self._buffer.clear()

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 2:
            depth = depth[None, ...]
        if not self._buffer:
            for _ in range(self.num_frames):
                self._buffer.append(depth)
        else:
            self._buffer.append(depth)
        stacked = np.concatenate(list(self._buffer), axis=0)
        return stacked


__all__ = ["DepthNormalize", "FrameStack"]
