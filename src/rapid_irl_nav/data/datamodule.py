"""Data module utilities bridging recorded datasets and training loops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .record import EP_ARRAY_KEYS, load_episode_arrays, load_episode_meta


class RapidNavDataset(Dataset):
    """Synthetic dataset used as a fallback when no recordings are available."""

    def __init__(self, length: int = 1024, seed: int = 0) -> None:
        super().__init__()
        self.length = length
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        depth = torch.rand((1, 64, 64), generator=self.generator)
        vel = torch.randn((3,), generator=self.generator) * 0.1
        quat = torch.randn((4,), generator=self.generator)
        quat = quat / quat.norm(p=2)
        goal = torch.randn((3,), generator=self.generator)
        pos_xy = torch.randn((2,), generator=self.generator)
        action = torch.randn((10, 2), generator=self.generator)
        done = torch.zeros((1,), dtype=torch.uint8)

        return {
            "depth": depth,
            "vel": vel,
            "quat": quat,
            "pos_xy": pos_xy,
            "goal": goal,
            "action_raw": action,
            "done": done,
        }


class NpzEpisodeDataset(Dataset):
    """Dataset that streams recorded episodes backed by the shard layout."""

    def __init__(self, dataset_root: Path | str, split: str = "train") -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.shards_dir = self.dataset_root / "shards"
        self.index_path = self.dataset_root / "index.parquet"

        if not self.index_path.exists():
            raise FileNotFoundError(self.index_path)

        self._frame = pd.read_parquet(self.index_path)
        if "split" in self._frame.columns:
            self._frame = self._frame[self._frame["split"] == split]
        self._frame = self._frame.sort_values(["shard", "episode"]).reset_index(drop=True)

    def __len__(self) -> int:
        return int(len(self._frame))

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self._frame.iloc[index]
        shard_name = str(row["shard"])
        episode_name = str(row["episode"])
        episode_dir = self.shards_dir / shard_name / episode_name
        arrays = load_episode_arrays(episode_dir)
        meta_path = episode_dir / "meta.json"
        meta = load_episode_meta(meta_path)
        payload: Dict[str, object] = {"meta": meta}
        for key in EP_ARRAY_KEYS:
            payload[key] = arrays[key]
        return payload


@dataclass
class RapidNavDataModule:
    batch_size: int = 32
    num_workers: int = 0
    dataset_root: Optional[Path] = None
    synthetic_length: int = 1024
    seed: int = 0

    _hint_printed: bool = False

    def _dataset(self, split: str) -> Tuple[Dataset, bool]:
        if self.dataset_root is not None:
            root = Path(self.dataset_root)
            try:
                dataset = NpzEpisodeDataset(root, split=split)
                return dataset, True
            except FileNotFoundError:
                if not self._hint_printed:
                    print(
                        f"No dataset at {root}. Run: python -m rapid_irl_nav.cli.collect_expert --dataset-root {root}"
                    )
                    self._hint_printed = True
            except Exception as exc:
                if not self._hint_printed:
                    print(f"Failed to load dataset at {root}: {exc}")
                    self._hint_printed = True
        return RapidNavDataset(length=self.synthetic_length, seed=self.seed), False

    def _loader(self, split: str, *, shuffle: bool, drop_last: bool) -> DataLoader:
        dataset, uses_recordings = self._dataset(split)
        if uses_recordings:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=shuffle,
                drop_last=False,
                collate_fn=lambda batch: batch,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(split="train", shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(split="val", shuffle=False, drop_last=False)


__all__ = ["RapidNavDataset", "NpzEpisodeDataset", "RapidNavDataModule"]


