"""Phase 0 regression test for the rapid_v2 ToyWorld bridge."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rapid_v2.env_builder import BuilderConfig, build_toy_world
from rapid_v2.tools.smoke_env import run_smoke
from rapid_irl_nav.sim.toy_world import ToyWorldConfig


def test_smoke_env_deterministic_with_seed():
    result_a = run_smoke(seed=123)
    result_b = run_smoke(seed=123)

    assert result_a["steps"] == result_b["steps"]
    assert np.isclose(result_a["total_reward"], result_b["total_reward"])
    assert result_a["info"] == result_b["info"]


def test_toy_world_builder_respects_custom_config():
    builder_cfg = BuilderConfig(seed=202)
    toy_cfg = ToyWorldConfig(success_radius=0.05, max_steps=42)
    backend = build_toy_world(builder_cfg, toy_config=toy_cfg)

    obs = backend.reset()
    assert obs.goal.shape == (3,)

    action = backend.expert.act(obs)
    next_obs, reward, done, info = backend.step(action)

    assert next_obs.depth.shape == obs.depth.shape
    assert isinstance(reward, (float, np.floating))
    assert isinstance(done, bool)
    assert "collisions" in info
    assert backend.env.config.success_radius == toy_cfg.success_radius
