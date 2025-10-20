"""Training CLI entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

try:
    import tyro  # type: ignore
except ImportError:  # pragma: no cover - fallback
    tyro = None  # type: ignore
    import argparse


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainArgs:
    """Command line arguments for training."""

    config: Path = PACKAGE_ROOT / "configs" / "training.yaml"
    logdir: Path = Path("runs/rapid_irl_nav")
    seed: int = 42


def _parse_args() -> TrainArgs:
    if tyro is not None:
        return tyro.cli(TrainArgs)

    parser = argparse.ArgumentParser(description="Rapid IRL Navigation trainer")
    parser.add_argument("--config", type=Path, default=TrainArgs.config, help="Path to YAML config file.")
    parser.add_argument("--logdir", type=Path, default=TrainArgs.logdir, help="Directory for TensorBoard logs.")
    parser.add_argument("--seed", type=int, default=TrainArgs.seed, help="Random seed.")
    ns = parser.parse_args()
    return TrainArgs(config=ns.config, logdir=ns.logdir, seed=ns.seed)


def main(args: Optional[TrainArgs] = None) -> None:
    console = Console()
    if args is None:
        args = _parse_args()

    console.print(f"[bold green]Starting training[/] with config {args.config}")
    from ..irl.trainer import Trainer  # Local import to keep CLI light-weight

    trainer = Trainer(args.config, args.logdir, seed=args.seed)
    trainer.run()
    console.print("[bold green]Training finished.[/]")


if __name__ == "__main__":
    main()
