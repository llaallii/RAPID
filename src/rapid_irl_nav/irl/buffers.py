"""Replay buffer utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    absorbing: bool


class ReplayBuffer:
    """Simple list-backed replay buffer for rapid prototyping."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self.storage: List[Transition] = []
        self._ptr = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: Transition) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self._ptr] = transition
        self._ptr = (self._ptr + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        indices = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[i] for i in indices]
        obs = np.stack([t.obs for t in batch])
        actions = np.stack([t.action for t in batch])
        rewards = np.asarray([t.reward for t in batch], dtype=np.float32)
        next_obs = np.stack([t.next_obs for t in batch])
        dones = np.asarray([t.done for t in batch], dtype=np.float32)
        absorbing = np.asarray([t.absorbing for t in batch], dtype=np.float32)
        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
            "absorbing": absorbing,
        }


class ExpertReplayBuffer(ReplayBuffer):
    """Dedicated storage for expert demonstrations."""

    pass


__all__ = ["ReplayBuffer", "ExpertReplayBuffer", "Transition"]
