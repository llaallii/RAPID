"""Temporary smoke test bridging ToyWorld into the rapid_v2 builder."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from rapid_v2.env_builder import BuilderConfig, build_toy_world
from rapid_v2.randomization_utils import RandomizationConfig, wrap_policy


def run_smoke(seed: int | None) -> dict:
    """Execute a short ToyWorld rollout and capture summary stats."""

    builder_cfg = BuilderConfig(seed=seed)
    random_cfg = RandomizationConfig(seed=seed)

    backend = build_toy_world(builder_cfg)
    policy = wrap_policy(backend.expert.act, yaw_std=random_cfg.yaw_jitter, seed=random_cfg.seed)

    obs = backend.reset()
    total_reward = 0.0
    steps = 0
    while steps < 30:
        action = policy(obs)
        obs, reward, done, info = backend.step(action)
        steps += 1
        total_reward += float(reward)
        if done:
            break

    return {
        "seed": seed,
        "steps": steps,
        "total_reward": total_reward,
        "done": done,
        "info": info,
        "builder_config": asdict(builder_cfg),
        "randomization_config": asdict(random_cfg),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic ToyWorld runs.")
    args = parser.parse_args()

    result = run_smoke(args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
