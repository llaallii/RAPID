"""CLI entry-point for dataset validation."""

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

from ..data.record import validate_dataset, write_index


@dataclass
class ValidateArgs:
    dataset_root: Path = Path("data/rapid_demo")
    rebuild_index: bool = False


def _parse_args() -> ValidateArgs:
    if tyro is not None:
        return tyro.cli(ValidateArgs)

    parser = argparse.ArgumentParser(description="Validate a Rapid IRL navigation dataset.")
    parser.add_argument("--dataset-root", type=Path, default=ValidateArgs.dataset_root)
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild index.parquet before validation.")
    ns = parser.parse_args()
    return ValidateArgs(dataset_root=ns.dataset_root, rebuild_index=ns.rebuild_index)


def main(args: Optional[ValidateArgs] = None) -> None:
    console = Console()
    if args is None:
        args = _parse_args()

    dataset_root = args.dataset_root
    if args.rebuild_index:
        write_index(dataset_root)

    try:
        report = validate_dataset(dataset_root)
    except FileNotFoundError as exc:
        console.print(f"[red]Validation failed:[/] {exc}")
        return

    console.print(report.format_summary())
    if report.issues:
        for issue in report.issues:
            console.print(f"- {issue}")


if __name__ == "__main__":
    main()
