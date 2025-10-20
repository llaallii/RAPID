"""Thin logging wrapper with optional TensorBoard support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - TensorBoard is optional in tests
    SummaryWriter = None  # type: ignore


class ExperimentLogger:
    """Convenience wrapper that writes both to stdout and TensorBoard."""

    def __init__(self, name: str, log_dir: Optional[Path] = None, level: int = logging.INFO) -> None:
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(level)
        self.logger.propagate = False

        self._writer: Optional[SummaryWriter] = None
        if log_dir is not None and SummaryWriter is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(str(log_dir))

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(msg, *args, **kwargs)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Record scalar metrics to stdout and TensorBoard."""

        if metrics:
            pretty = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            self.logger.info("step %d | %s", step, pretty)

        if self._writer is None:
            return

        for key, value in metrics.items():
            self._writer.add_scalar(key, value, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        self.close()


__all__ = ["ExperimentLogger"]
