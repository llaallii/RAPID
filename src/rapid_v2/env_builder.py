"""Entry point for constructing RAPID simulation environments.

Phase 0 implements a compatibility wrapper around the legacy ToyWorld so
CLI smoke tests continue to function while the Isaac Sim stack comes
online.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rapid_irl_nav.sim.toy_world import ScriptedExpert, ToyWorld, ToyWorldConfig


class Backend(Protocol):
    """Protocol for env backends targeted by the builder."""

    def reset(self, seed: int | None = None):
        ...

    def step(self, action):
        ...


@dataclass
class BuilderConfig:
    """Builder-level configuration shared across backends."""

    seed: int | None = None


class ToyWorldBackend:
    """ToyWorld adapter satisfying the backend protocol."""

    def __init__(
        self,
        config: ToyWorldConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.config = config or ToyWorldConfig()
        self.seed = seed
        self._env = ToyWorld(self.config, seed=seed)
        self._expert = ScriptedExpert()

    @property
    def env(self) -> ToyWorld:
        return self._env

    @property
    def expert(self) -> ScriptedExpert:
        return self._expert

    def reset(self):
        """Reset the toy environment."""

        return self._env.reset(seed=self.seed)

    def step(self, action):
        """Perform a step against the toy backend."""

        return self._env.step(action)


def build_toy_world(
    config: BuilderConfig | None = None,
    *,
    toy_config: ToyWorldConfig | None = None,
) -> ToyWorldBackend:
    """Factory for the ToyWorld backend."""

    cfg = config or BuilderConfig()
    backend = ToyWorldBackend(config=toy_config, seed=cfg.seed)
    return backend


# TODO(Phase 1): Extend to Isaac Sim builders using the Backend protocol.

__all__ = [
    "Backend",
    "BuilderConfig",
    "ToyWorldBackend",
    "build_toy_world",
]
