"""Utilities for deterministic experiment setup."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic_cudnn: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch RNGs.

    Parameters
    ----------
    seed:
        The random seed to apply. If ``None`` we keep the current global seed.
    deterministic_cudnn:
        Whether to ask cuDNN for deterministic kernels. This can reduce
        throughput, so it is opt-in.

    Returns
    -------
    int
        The seed that was actually applied. Returned for convenience so callers
        can log it easily.
    """

    if seed is None:
        return DEFAULT_SEED

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    return seed


__all__ = ["seed_everything", "DEFAULT_SEED"]
