"""Minimal data module scaffolding for development."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .schemas import RapidNavTimestep


class RapidNavDataset(Dataset):
    """Dataset that currently generates random samples on the fly.

    TODO: swap this with a dataset that streams episodes from recorded shards or
    simulator rollouts. The synthetic version keeps the training loop runnable
    while we iterate on the surrounding infrastructure.
    """

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
        action = torch.randn((10, 2), generator=self.generator)

        sample = RapidNavTimestep(
            depth=depth.numpy(),
            vel=vel.numpy(),
            quat=quat.numpy(),
            goal=goal.numpy(),
            action_raw=action.numpy(),
        )

        return {
            "depth": torch.from_numpy(sample.depth),
            "vel": torch.from_numpy(sample.vel),
            "quat": torch.from_numpy(sample.quat),
            "goal": torch.from_numpy(sample.goal),
            "action": torch.from_numpy(sample.action_raw),
        }


@dataclass
class RapidNavDataModule:
    batch_size: int = 32
    num_workers: int = 0
    length: int = 1024
    seed: int = 0

    def _dataset(self) -> RapidNavDataset:
        return RapidNavDataset(length=self.length, seed=self.seed)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )


__all__ = ["RapidNavDataset", "RapidNavDataModule"]
