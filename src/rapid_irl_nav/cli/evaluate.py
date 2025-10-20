"""Evaluation CLI."""

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
class EvalArgs:
    checkpoint: Path
    episodes: int = 5


def _parse_args() -> EvalArgs:
    if tyro is not None:
        return tyro.cli(EvalArgs)

    parser = argparse.ArgumentParser(description="Evaluate a saved policy")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the saved weights.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    ns = parser.parse_args()
    return EvalArgs(checkpoint=ns.checkpoint, episodes=ns.episodes)


def main(args: Optional[EvalArgs] = None) -> None:
    console = Console()
    if args is None:
        args = _parse_args()

    if not args.checkpoint.exists():
        console.print(f"[red]Checkpoint {args.checkpoint} not found. Nothing to evaluate yet.[/]")
        return

    console.print(
        "[yellow]Evaluation pipeline is a placeholder. TODO: load policy and run rollouts.[/]"
    )


if __name__ == "__main__":
    main()
