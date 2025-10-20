"""Interfaces describing how we interact with simulators or log files."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np

from ..data.schemas import RapidNavTimestep


class ObservationSource(abc.ABC):
    """Abstract source that yields timesteps."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the iterator to the start of a new episode."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator[RapidNavTimestep]:
        ...


class RecordedDatasetSource(ObservationSource):
    """Source that replays timesteps from a pre-recorded dataset."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._timesteps = self._load(path)
        self._cursor = 0

    def _load(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(path)
        # TODO: replace with proper on-disk format (e.g. NPZ / Parquet)
        return np.load(path, allow_pickle=True)

    def reset(self) -> None:
        self._cursor = 0

    def __iter__(self) -> Iterator[RapidNavTimestep]:
        self.reset()
        while self._cursor < len(self._timesteps):
            payload: Dict[str, np.ndarray] = self._timesteps[self._cursor].item()
            self._cursor += 1
            yield RapidNavTimestep(**payload)


# TODO: Provide an ObservationSource backed by Isaac Sim for hardware-in-the-loop experiments.


__all__ = ["ObservationSource", "RecordedDatasetSource"]
