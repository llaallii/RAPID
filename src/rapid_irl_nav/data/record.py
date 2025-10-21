"""Episode recording helpers for the expert dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .schemas import DatasetMeta, EpisodeMeta, TimestepDtypes, ValidateReport, check_episode_shapes

SCHEMA_VERSION = "1.0"
EP_ARRAY_KEYS = (
    "timestamps",
    "depth",
    "vel",
    "quat",
    "pos_xy",
    "goal",
    "action_raw",
    "done",
)


def _model_dump_json(model) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(indent=2)
    if hasattr(model, "json"):
        return model.json(indent=2)
    if hasattr(model, "dict"):
        return json.dumps(model.dict(), indent=2)
    raise TypeError(f"Unsupported model type: {type(model)}")


def _parse_dataset_model(model_cls, payload: dict):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def load_episode_meta(meta_path: Path) -> EpisodeMeta:
    payload = json.loads(meta_path.read_text())
    return _parse_dataset_model(EpisodeMeta, payload)


def _format_shard_index(index: int) -> str:
    return f"shard-{index:06d}"


def _format_episode_index(index: int) -> str:
    return f"ep-{index:06d}"


class EpisodeWriter:
    """Buffers timestep data and materialises an episode on disk."""

    def __init__(self, shard_name: str, episode_dir: Path, meta: EpisodeMeta) -> None:
        self.shard_name = shard_name
        self.episode_dir = episode_dir
        self.meta = meta
        self._timestamps: List[float] = []
        self._depth: List[np.ndarray] = []
        self._vel: List[np.ndarray] = []
        self._quat: List[np.ndarray] = []
        self._pos_xy: List[np.ndarray] = []
        self._goal: List[np.ndarray] = []
        self._action_raw: List[np.ndarray] = []
        self._done: List[int] = []
        self._closed = False

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if exc is None:
            self.close(success=self.meta.success)

    @property
    def episode_name(self) -> str:
        return self.meta.episode_id

    def step(
        self,
        depth: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        pos_xy: np.ndarray,
        goal: np.ndarray,
        action_raw: np.ndarray,
        done: bool,
        timestamp: float,
    ) -> None:
        if self._closed:
            raise RuntimeError("EpisodeWriter is already closed")

        self._timestamps.append(float(timestamp))
        self._depth.append(np.asarray(depth, dtype=TimestepDtypes.depth))
        self._vel.append(np.asarray(vel, dtype=TimestepDtypes.vel))
        self._quat.append(np.asarray(quat, dtype=TimestepDtypes.quat))
        self._pos_xy.append(np.asarray(pos_xy, dtype=TimestepDtypes.pos_xy))
        self._goal.append(np.asarray(goal, dtype=TimestepDtypes.goal))
        self._action_raw.append(np.asarray(action_raw, dtype=TimestepDtypes.action_raw))
        self._done.append(1 if done else 0)

    def update_meta(self, **kwargs: object) -> None:
        """Update episode metadata prior to closing."""

        self.meta = self.meta.copy(update=kwargs)

    def close(self, success: bool) -> int:
        if self._closed:
            return len(self._timestamps)

        self._closed = True
        if not self._timestamps:
            raise RuntimeError("No timesteps recorded; call step() before close().")
        self.meta = self.meta.copy(update={"success": success})
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        arrays: Dict[str, np.ndarray] = {
            "timestamps": np.asarray(self._timestamps, dtype=TimestepDtypes.timestamp),
            "depth": np.ascontiguousarray(np.stack(self._depth, axis=0)),
            "vel": np.ascontiguousarray(np.stack(self._vel, axis=0)),
            "quat": np.ascontiguousarray(np.stack(self._quat, axis=0)),
            "pos_xy": np.ascontiguousarray(np.stack(self._pos_xy, axis=0)),
            "goal": np.ascontiguousarray(np.stack(self._goal, axis=0)),
            "action_raw": np.ascontiguousarray(np.stack(self._action_raw, axis=0)),
            "done": np.asarray(self._done, dtype=TimestepDtypes.done),
        }

        if arrays["action_raw"].size > 0:
            arrays["action_raw"][0] = np.nan

        T = check_episode_shapes(arrays)

        for name, array in arrays.items():
            np.save(self.episode_dir / f"{name}.npy", array)

        meta_path = self.episode_dir / "meta.json"
        meta_path.write_text(_model_dump_json(self.meta))
        return T


@dataclass
class _ShardState:
    index: int
    path: Path
    next_episode_idx: int
    episode_count: int


class ShardManager:
    """Utility to group episodes into shards of a fixed size."""

    def __init__(
        self,
        dataset_root: Path | str,
        *,
        dataset_meta: DatasetMeta | None = None,
        max_eps_per_shard: int = 200,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.shards_dir = self.dataset_root / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_meta_path = self.dataset_root / "dataset.json"
        self.max_eps_per_shard = max_eps_per_shard
        self.dataset_meta = self._ensure_dataset_meta(dataset_meta)
        self._state = self._discover_state()

    def _ensure_dataset_meta(self, dataset_meta: DatasetMeta | None) -> DatasetMeta:
        if self.dataset_meta_path.exists():
            payload = json.loads(self.dataset_meta_path.read_text())
            if hasattr(DatasetMeta, "model_validate"):
                return DatasetMeta.model_validate(payload)
            return DatasetMeta.parse_obj(payload)

        if dataset_meta is None:
            dataset_meta = DatasetMeta(
                schema_version=SCHEMA_VERSION,
                sensor_spec={"depth": {"shape": [1, 64, 64], "dtype": "float32"}},
                source="unknown",
            )
        self.dataset_meta_path.write_text(_model_dump_json(dataset_meta))
        return dataset_meta

    def _discover_state(self) -> _ShardState:
        shard_paths = sorted(
            (p for p in self.shards_dir.iterdir() if p.is_dir() and p.name.startswith("shard-")),
            key=lambda path: path.name,
        )
        if not shard_paths:
            shard_path = self._shard_path_for_index(0)
            shard_path.mkdir(parents=True, exist_ok=True)
            return _ShardState(index=0, path=shard_path, next_episode_idx=0, episode_count=0)

        last_shard = shard_paths[-1]
        shard_index = int(last_shard.name.split("-", maxsplit=1)[1])
        episode_indices = [
            int(ep.name.split("-", maxsplit=1)[1])
            for ep in last_shard.iterdir()
            if ep.is_dir() and ep.name.startswith("ep-")
        ]
        if episode_indices:
            next_idx = max(episode_indices) + 1
            episode_count = len(episode_indices)
        else:
            next_idx = 0
            episode_count = 0

        if episode_count >= self.max_eps_per_shard:
            shard_index += 1
            shard_path = self._shard_path_for_index(shard_index)
            shard_path.mkdir(parents=True, exist_ok=True)
            return _ShardState(index=shard_index, path=shard_path, next_episode_idx=0, episode_count=0)

        return _ShardState(
            index=shard_index,
            path=last_shard,
            next_episode_idx=next_idx,
            episode_count=episode_count,
        )

    def _shard_path_for_index(self, index: int) -> Path:
        return self.shards_dir / _format_shard_index(index)

    def _rotate_if_needed(self) -> None:
        if self._state.episode_count < self.max_eps_per_shard:
            return
        new_index = self._state.index + 1
        shard_path = self._shard_path_for_index(new_index)
        shard_path.mkdir(parents=True, exist_ok=True)
        self._state = _ShardState(index=new_index, path=shard_path, next_episode_idx=0, episode_count=0)

    def create_episode_writer(
        self,
        env: str,
        goal_xy: Sequence[float],
        seed: int,
        *,
        notes: str | None = None,
    ) -> EpisodeWriter:
        self._rotate_if_needed()
        episode_idx = self._state.next_episode_idx
        episode_name = _format_episode_index(episode_idx)
        episode_dir = self._state.path / episode_name
        episode_dir.mkdir(parents=True, exist_ok=True)

        meta = EpisodeMeta(
            episode_id=episode_name,
            env=env,
            success=False,
            collision_count=0,
            goal_xy=list(goal_xy),
            seed=seed,
            notes=notes,
        )
        writer = EpisodeWriter(self._state.path.name, episode_dir, meta)
        self._state = _ShardState(
            index=self._state.index,
            path=self._state.path,
            next_episode_idx=episode_idx + 1,
            episode_count=self._state.episode_count + 1,
        )
        return writer


def load_episode_arrays(episode_dir: Path, keys: Iterable[str] = EP_ARRAY_KEYS) -> Dict[str, np.ndarray]:
    """Load the standard NumPy arrays from an episode directory."""

    arrays: Dict[str, np.ndarray] = {}
    for key in keys:
        path = episode_dir / f"{key}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        arrays[key] = np.load(path)
    return arrays


def write_index(dataset_root: Path | str) -> Path:
    """Rebuild the Parquet index for the dataset."""

    dataset_root = Path(dataset_root)
    shards_dir = dataset_root / "shards"
    rows: List[Dict[str, object]] = []

    for shard_path in sorted(shards_dir.iterdir()):
        if not shard_path.is_dir() or not shard_path.name.startswith("shard-"):
            continue
        for episode_path in sorted(shard_path.iterdir()):
            if not episode_path.is_dir() or not episode_path.name.startswith("ep-"):
                continue

            meta_path = episode_path / "meta.json"
            if not meta_path.exists():
                continue
            meta = load_episode_meta(meta_path)

            arrays = load_episode_arrays(episode_path)
            try:
                T = check_episode_shapes(arrays)
            except ValueError:
                T = arrays["timestamps"].shape[0]

            timestamps = arrays["timestamps"]
            depth = arrays["depth"]
            mean_depth_ok = bool(np.isfinite(depth).mean() > 0.90)

            rows.append(
                {
                    "shard": shard_path.name,
                    "episode": episode_path.name,
                    "T": int(T),
                    "t0": float(timestamps[0]) if len(timestamps) else np.nan,
                    "t1": float(timestamps[-1]) if len(timestamps) else np.nan,
                    "success": bool(meta.success),
                    "mean_depth_ok": mean_depth_ok,
                }
            )

    index_path = dataset_root / "index.parquet"
    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(index_path, index=False)
    else:
        # Create an empty file so downstream tools know the dataset exists.
        pd.DataFrame(columns=["shard", "episode", "T", "t0", "t1", "success", "mean_depth_ok"]).to_parquet(
            index_path, index=False
        )

    return index_path


def validate_dataset(dataset_root: Path | str) -> ValidateReport:
    """Validate the dataset contents and return a summary report."""

    dataset_root = Path(dataset_root)
    shards_dir = dataset_root / "shards"
    index_path = dataset_root / "index.parquet"

    if not index_path.exists():
        raise FileNotFoundError(f"index.parquet missing at {index_path}")

    df = pd.read_parquet(index_path)
    total = int(len(df))
    issues: List[str] = []
    valid = 0

    for row in df.itertuples(index=False):
        episode_dir = shards_dir / getattr(row, "shard") / getattr(row, "episode")
        try:
            arrays = load_episode_arrays(episode_dir)
            check_episode_shapes(arrays)
            valid += 1
        except (FileNotFoundError, ValueError) as exc:
            issues.append(f"{getattr(row, 'shard')}/{getattr(row, 'episode')}: {exc}")

    invalid = total - valid
    success_rate = float(df["success"].mean()) if total else None
    mean_length = float(df["T"].mean()) if total else None

    return ValidateReport(
        dataset_root=dataset_root,
        total_episodes=total,
        valid_episodes=valid,
        invalid_episodes=invalid,
        success_rate=success_rate,
        mean_length=mean_length,
        issues=issues,
    )


__all__ = [
    "EpisodeWriter",
    "ShardManager",
    "load_episode_meta",
    "load_episode_arrays",
    "write_index",
    "validate_dataset",
    "SCHEMA_VERSION",
]



