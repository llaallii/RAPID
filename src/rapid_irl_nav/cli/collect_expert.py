"""Expert data collection CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

try:
    import tyro  # type: ignore
except ImportError:  # pragma: no cover
    tyro = None  # type: ignore
    import argparse


@dataclass
class CollectArgs:
    output: Path = Path("expert_rollouts.npz")
    num_episodes: int = 10


def _parse_args() -> CollectArgs:
    if tyro is not None:
        return tyro.cli(CollectArgs)

    parser = argparse.ArgumentParser(description="Collect expert demonstrations")
    parser.add_argument("--output", type=Path, default=CollectArgs.output, help="Output path for rollouts.")
    parser.add_argument("--num-episodes", type=int, default=CollectArgs.num_episodes, help="Number of expert episodes to record.")
    ns = parser.parse_args()
    return CollectArgs(output=ns.output, num_episodes=ns.num_episodes)


def main(args: Optional[CollectArgs] = None) -> None:
    console = Console()
    if args is None:
        args = _parse_args()

    console.print(
        "[yellow]Expert collection is not yet implemented. TODO: integrate planner or oracle policy.[/]"
    )
    console.print(f"Would store rollouts at: {args.output} (episodes={args.num_episodes})")


if __name__ == "__main__":
    main()
