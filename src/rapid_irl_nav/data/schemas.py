"""Schemas and validation helpers for the expert dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, validator


class TimestepDtypes:
    """Central place for dtype constants so they stay consistent."""

    timestamp = np.float64
    depth = np.float32
    vel = np.float32
    quat = np.float32
    pos_xy = np.float32
    goal = np.float32
    action_raw = np.float32
    done = np.uint8


class DatasetMeta(BaseModel):
    """Global metadata saved into dataset.json."""

    schema_version: str = Field(..., description="Semantic version of the on-disk schema.")
    sensor_spec: Mapping[str, Any] = Field(
        default_factory=dict,
        description="Key/value pairs describing sensor configuration.",
    )
    source: str = Field(
        default="unknown",
        description="Name of the simulator or logging backend that produced the dataset.",
    )
    notes: str | None = Field(
        default=None,
        description="Optional free-form notes about how the dataset was recorded.",
    )


class EpisodeMeta(BaseModel):
    """Per-episode metadata saved alongside array data."""

    episode_id: str
    env: str
    success: bool
    collision_count: int = 0
    goal_xy: Sequence[float]
    seed: int
    notes: str | None = None

    @validator("goal_xy")
    def _check_goal_xy(cls, value: Sequence[float]) -> Sequence[float]:
        if len(value) != 2:
            raise ValueError(f"goal_xy must have length 2, got {len(value)}")
        return value


class ValidateReport(BaseModel):
    """Summary of dataset validation results."""

    dataset_root: Path
    total_episodes: int
    valid_episodes: int
    invalid_episodes: int
    success_rate: float | None = None
    mean_length: float | None = None
    issues: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.invalid_episodes == 0 and not self.issues

    def format_summary(self) -> str:
        status = "OK" if self.ok else "WARN"
        metrics = []
        if self.mean_length is not None:
            metrics.append(f"mean T={self.mean_length:.1f}")
        if self.success_rate is not None:
            metrics.append(f"success={self.success_rate:.2%}")
        metric_str = ", ".join(metrics) if metrics else "no metrics"
        return (
            f"[{status}] episodes={self.total_episodes} valid={self.valid_episodes} "
            f"invalid={self.invalid_episodes} ({metric_str})"
        )


EpisodeArrays = Mapping[str, NDArray[Any]]


def _require_keys(arrays: EpisodeArrays, required: Iterable[str]) -> None:
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"Missing episode arrays: {', '.join(missing)}")


def _check_shape(name: str, array: NDArray[Any], expected: tuple[int | None, ...]) -> None:
    if len(array.shape) != len(expected):
        raise ValueError(f"{name} must have {len(expected)} dimensions, got shape {array.shape}")

    for idx, (dim, exp) in enumerate(zip(array.shape, expected, strict=True)):
        if exp is None:
            continue
        if dim != exp:
            raise ValueError(f"{name} axis {idx} must be {exp}, got {dim} (shape {array.shape})")


def check_episode_shapes(arrays: EpisodeArrays) -> int:
    """Validate array shapes/dtypes for a single episode.

    Returns
    -------
    int
        The number of timesteps `T`.

    Raises
    ------
    ValueError
        If any required array is missing or has an incorrect shape/dtype.
    """

    required_arrays = {
        "timestamps": (TimestepDtypes.timestamp, (None,)),
        "depth": (TimestepDtypes.depth, (None, 1, 64, 64)),
        "vel": (TimestepDtypes.vel, (None, 3)),
        "quat": (TimestepDtypes.quat, (None, 4)),
        "pos_xy": (TimestepDtypes.pos_xy, (None, 2)),
        "goal": (TimestepDtypes.goal, (None, 3)),
        "action_raw": (TimestepDtypes.action_raw, (None, 10, 2)),
        "done": (TimestepDtypes.done, (None,)),
    }

    _require_keys(arrays, required_arrays.keys())

    lengths: list[int] = []
    for name, (dtype, shape) in required_arrays.items():
        array = np.asarray(arrays[name])
        if array.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}, got {array.dtype}")
        _check_shape(name, array, shape)
        lengths.append(array.shape[0])

    T = lengths[0]
    if any(length != T for length in lengths):
        raise ValueError(f"All arrays must have the same length; got lengths={lengths}")

    return T


__all__ = [
    "DatasetMeta",
    "EpisodeMeta",
    "ValidateReport",
    "TimestepDtypes",
    "check_episode_shapes",
]
