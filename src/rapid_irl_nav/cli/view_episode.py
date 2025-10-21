"""CLI to inspect a recorded episode and visualize the trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console

try:
    import tyro  # type: ignore
except ImportError:  # pragma: no cover
    tyro = None  # type: ignore
    import argparse

from ..data.record import load_episode_arrays, load_episode_meta
from ..utils.viz import plot_waypoints


@dataclass
class ViewArgs:
    dataset_root: Path = Path("data/rapid_demo")
    ep: Optional[str] = None
    idx: Optional[int] = None


def _parse_args() -> ViewArgs:
    if tyro is not None:
        return tyro.cli(ViewArgs)

    parser = argparse.ArgumentParser(description="Visualize a recorded navigation episode.")
    parser.add_argument("--dataset-root", type=Path, default=ViewArgs.dataset_root)
    parser.add_argument("--ep", type=str, help="Episode path like shard-000000/ep-000123.")
    parser.add_argument("--idx", type=int, help="Row index in index.parquet.")
    ns = parser.parse_args()
    return ViewArgs(dataset_root=ns.dataset_root, ep=ns.ep, idx=ns.idx)


def _resolve_episode(args: ViewArgs, frame: pd.DataFrame) -> tuple[str, str]:
    if args.ep:
        ep_str = args.ep.replace("\\", "/").strip("/")
        parts = ep_str.split("/")
        if len(parts) == 2:
            return parts[0], parts[1]
        if len(parts) == 1:
            matches = frame[frame["episode"] == parts[0]]
            if matches.empty:
                raise FileNotFoundError(f"Episode {parts[0]} not found in index.")
            row = matches.iloc[0]
            return str(row["shard"]), str(row["episode"])
        raise ValueError(f"Unable to parse episode identifier: {args.ep}")

    if args.idx is not None:
        if args.idx < 0 or args.idx >= len(frame):
            raise IndexError(f"Index {args.idx} out of range (0 <= idx < {len(frame)}).")
        row = frame.iloc[args.idx]
        return str(row["shard"]), str(row["episode"])

    raise ValueError("Specify either --ep or --idx to select an episode.")


def main(args: Optional[ViewArgs] = None) -> None:
    console = Console()
    if args is None:
        args = _parse_args()

    dataset_root = args.dataset_root
    index_path = dataset_root / "index.parquet"
    if not index_path.exists():
        console.print(f"[red]index.parquet not found at[/] {index_path}")
        return

    frame = pd.read_parquet(index_path)
    if frame.empty:
        console.print(f"[yellow]Empty index at[/] {index_path}")
        return

    try:
        shard_name, episode_name = _resolve_episode(args, frame)
    except (FileNotFoundError, ValueError, IndexError) as exc:
        console.print(f"[red]Error:[/] {exc}")
        return

    episode_dir = dataset_root / "shards" / shard_name / episode_name
    meta_path = episode_dir / "meta.json"
    if not meta_path.exists():
        console.print(f"[red]Missing meta.json at[/] {meta_path}")
        return

    arrays = load_episode_arrays(episode_dir)
    meta = load_episode_meta(meta_path)

    console.print(f"[green]Episode:[/] {shard_name}/{episode_name}")
    console.print(f"success={meta.success} collisions={meta.collision_count} seed={meta.seed}")
    if meta.notes:
        console.print(f"notes: {meta.notes}")

    waypoints = arrays["pos_xy"]
    goal_traj = arrays["goal"]
    goal_xy = goal_traj[-1, :2] if len(goal_traj) else None
    plot_waypoints(waypoints, goal=goal_xy, title=f"{shard_name}/{episode_name}")


if __name__ == "__main__":
    main()



