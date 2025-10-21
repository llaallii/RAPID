"""Quick visualization helpers for debugging."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_depth(depth: np.ndarray, title: str = "depth", *, cmap: str = "viridis") -> None:
    """Display a depth map using matplotlib.

    This function is intentionally lightweight so it can be called from unit
    tests or notebooks without additional dependencies. A more sophisticated
    dashboard belongs in a dedicated visualization module.
    """

    plt.figure(figsize=(4, 4))
    plt.imshow(depth.squeeze(), cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    # TODO: allow saving figures directly to TensorBoard or Rich console outputs.


def plot_waypoints(
    waypoints: np.ndarray,
    *,
    goal: Optional[Sequence[float]] = None,
    title: str = "trajectory",
) -> None:
    """Plot a 2D trajectory with an optional goal marker."""

    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError(f"Waypoints must have shape (T, 2), got {waypoints.shape}")

    plt.figure(figsize=(5, 5))
    plt.plot(waypoints[:, 0], waypoints[:, 1], "-o", color="tab:blue", markersize=2, linewidth=1.5, label="trajectory")

    if goal is not None:
        gx, gy = goal
        plt.scatter([gx], [gy], marker="*", color="gold", s=150, edgecolor="black", linewidths=0.5, label="goal")

    plt.title(title)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()


def close_all() -> None:
    """Close all open matplotlib figures."""

    plt.close("all")


__all__ = ["plot_depth", "plot_waypoints", "close_all"]
