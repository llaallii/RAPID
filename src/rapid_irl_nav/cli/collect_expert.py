"""Collect scripted expert demonstrations into the shard dataset layout."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress

try:
    import tyro  # type: ignore
except ImportError:  # pragma: no cover
    tyro = None  # type: ignore
    import argparse

from ..data.record import SCHEMA_VERSION, DatasetMeta, ShardManager, write_index
from ..sim.toy_world import ScriptedExpert, ToyWorld, ToyWorldConfig


def _default_dataset_meta() -> DatasetMeta:
    return DatasetMeta(
        schema_version=SCHEMA_VERSION,
        sensor_spec={
            "timestamps": {"shape": ["T"], "dtype": "float64"},
            "depth": {"shape": [1, 64, 64], "dtype": "float32"},
            "vel": {"shape": [3], "dtype": "float32"},
            "quat": {"shape": [4], "dtype": "float32"},
            "pos_xy": {"shape": [2], "dtype": "float32"},
            "goal": {"shape": [3], "dtype": "float32"},
            "action_raw": {"shape": [10, 2], "dtype": "float32"},
            "done": {"shape": [], "dtype": "uint8"},
        },
        source="toy_world",
        notes="Generated with ScriptedExpert in ToyWorld.",
    )


@dataclass
class CollectArgs:
    dataset_root: Path = Path("data/rapid_demo")
    episodes: int = 50
    seed: int = 123
    max_steps: int = 300
    success_radius: float = 0.2
    max_eps_per_shard: int = 200
    dt: float = 0.1


def _parse_args() -> CollectArgs:
    if tyro is not None:
        return tyro.cli(CollectArgs)

    parser = argparse.ArgumentParser(description="Collect scripted expert demonstrations.")
    parser.add_argument("--dataset-root", type=Path, default=CollectArgs.dataset_root)
    parser.add_argument("--episodes", type=int, default=CollectArgs.episodes)
    parser.add_argument("--seed", type=int, default=CollectArgs.seed)
    parser.add_argument("--max-steps", type=int, default=CollectArgs.max_steps)
    parser.add_argument("--success-radius", type=float, default=CollectArgs.success_radius)
    parser.add_argument("--max-eps-per-shard", type=int, default=CollectArgs.max_eps_per_shard)
    parser.add_argument("--dt", type=float, default=CollectArgs.dt)
    ns = parser.parse_args()
    return CollectArgs(
        dataset_root=ns.dataset_root,
        episodes=ns.episodes,
        seed=ns.seed,
        max_steps=ns.max_steps,
        success_radius=ns.success_radius,
        max_eps_per_shard=ns.max_eps_per_shard,
        dt=ns.dt,
    )


def _timestamp_generator(start_time: float, dt: float):
    current = start_time
    while True:
        yield current
        current += dt


def collect_episode(
    env: ToyWorld,
    expert: ScriptedExpert,
    writer,
    initial_obs,
    *,
    dt: float,
    max_steps: int,
) -> tuple[int, bool, dict]:
    timestamp_iter = _timestamp_generator(start_time=time.time(), dt=dt)
    obs = initial_obs
    writer.update_meta(goal_xy=list(obs.goal[:2]))

    total_steps = 0
    success = False
    last_info: dict = {}

    for step in range(max_steps):
        action = expert.act(obs)
        next_obs, _, done, info = env.step(action)
        ts = next(timestamp_iter)

        writer.step(
            depth=obs.depth,
            vel=obs.vel,
            quat=obs.quat,
            pos_xy=obs.pos_xy,
            goal=obs.goal,
            action_raw=action,
            done=done,
            timestamp=ts,
        )

        total_steps += 1
        last_info = info
        obs = next_obs
        if done:
            break

    if last_info:
        writer.update_meta(collision_count=int(last_info.get("collisions", 0)))
        success = bool(
            last_info.get("distance_to_goal", float("inf")) <= env.config.success_radius
            and int(last_info.get("collisions", 0)) == 0
        )

    return total_steps, success, last_info


def main(args: Optional[CollectArgs] = None) -> None:
    console = Console()
    if args is None:
        args = _parse_args()

    config = ToyWorldConfig(dt=args.dt, max_steps=args.max_steps, success_radius=args.success_radius)
    env = ToyWorld(config=config, seed=args.seed)
    expert = ScriptedExpert()
    manager = ShardManager(
        args.dataset_root,
        dataset_meta=_default_dataset_meta(),
        max_eps_per_shard=args.max_eps_per_shard,
    )

    successes = 0
    total_T = 0

    with Progress(console=console) as progress:
        task = progress.add_task("Collecting", total=args.episodes)
        for episode_idx in range(args.episodes):
            episode_seed = args.seed + episode_idx
            obs = env.reset(seed=episode_seed)
            writer = manager.create_episode_writer(
                env="toy_world",
                goal_xy=obs.goal[:2],
                seed=episode_seed,
            )

            steps, success, _ = collect_episode(
                env,
                expert,
                writer,
                initial_obs=obs,
                dt=args.dt,
                max_steps=args.max_steps,
            )
            writer.update_meta(goal_xy=list(env.goal[:2]))
            writer.close(success)

            successes += int(success)
            total_T += steps
            progress.update(task, advance=1)

    index_path = write_index(args.dataset_root)
    mean_length = total_T / args.episodes if args.episodes else 0.0
    success_rate = successes / args.episodes if args.episodes else 0.0

    console.print(
        f"[green]Collected {args.episodes} episodes[/] "
        f"(mean length={mean_length:.1f}, success={success_rate:.0%}) -> {index_path}"
    )


if __name__ == "__main__":
    main()


