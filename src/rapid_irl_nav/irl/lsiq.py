"""LS-IQ related utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class LSIQObjective:
    """Placeholder LS-IQ objective wrapper.

    TODO: implement the inverse Bellman error minimisation described in the
    RAPID paper once the necessary dynamics terms are available.
    """

    discount: float

    def implicit_reward(self, q_values: torch.Tensor, value_targets: torch.Tensor) -> torch.Tensor:
        """Compute an implicit reward signal.

        Parameters
        ----------
        q_values:
            Current Q estimates.
        value_targets:
            Bootstrapped targets derived from expert demonstrations.
        """

        # TODO: implement LS-IQ closed-form solution. For now we return the
        # temporal-difference residual as a proxy so the trainer can proceed.
        return value_targets - q_values


__all__ = ["LSIQObjective"]
