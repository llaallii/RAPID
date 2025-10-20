"""Data schemas used throughout the project."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, validator

DepthArray = NDArray[np.float32]
Vector3 = NDArray[np.float32]
Quat = NDArray[np.float32]
ActionArray = NDArray[np.float32]


class RapidNavTimestep(BaseModel):
    """Single timestep sample recorded from the simulator or robot."""

    depth: DepthArray = Field(..., description="Depth image with shape (1, 64, 64)")
    vel: Vector3 = Field(..., description="Linear velocity in body frame")
    quat: Quat = Field(..., description="Orientation quaternion (wxyz)")
    goal: Vector3 = Field(..., description="Goal position in world coordinates")
    action_raw: ActionArray = Field(..., description="Planned cylindrical waypoint deltas")

    @validator("depth")
    def _check_depth(cls, value: DepthArray) -> DepthArray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (1, 64, 64):
            raise ValueError(f"depth must have shape (1, 64, 64), got {array.shape}")
        return array

    @validator("vel", "goal")
    def _check_vec3(cls, value: Vector3) -> Vector3:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (3,):
            raise ValueError(f"Vector must have shape (3,), got {array.shape}")
        return array

    @validator("quat")
    def _check_quat(cls, value: Quat) -> Quat:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (4,):
            raise ValueError(f"Quaternion must have shape (4,), got {array.shape}")
        return array

    @validator("action_raw")
    def _check_action(cls, value: ActionArray) -> ActionArray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (10, 2):
            raise ValueError(f"Action must have shape (10, 2), got {array.shape}")
        return array


class EpisodeMetadata(BaseModel):
    """Summary metadata for an episode."""

    episode_id: str
    num_steps: int
    reward_sum: float = 0.0
    terminated: bool = False
    truncated: bool = False
    notes: Optional[str] = None


__all__ = ["RapidNavTimestep", "EpisodeMetadata"]
