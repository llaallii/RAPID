"""Quick visualization helpers for debugging."""

from __future__ import annotations

from typing import Optional

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


def close_all() -> None:
    """Close all open matplotlib figures."""

    plt.close("all")


__all__ = ["plot_depth", "close_all"]
