"""Actor / critic networks for SAC training."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .init import apply_initialisation


def _d2rl_block(in_features: int, hidden_features: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_features, hidden_features), nn.ReLU(inplace=True))


class WaypointActor(nn.Module):
    """Gaussian policy that outputs cylindrical waypoint deltas."""

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_waypoints: int = 10,
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
        max_radius: float = 2.0,
        max_yaw: float = 3.1416,
    ) -> None:
        super().__init__()
        self.num_waypoints = num_waypoints
        self.log_std_bounds = log_std_bounds
        self.max_radius = max_radius
        self.max_yaw = max_yaw

        self.fc1 = _d2rl_block(latent_dim, hidden_dim)
        self.fc2 = _d2rl_block(hidden_dim + latent_dim, hidden_dim)
        self.fc3 = _d2rl_block(hidden_dim + latent_dim, hidden_dim)

        out_dim = num_waypoints * 2
        self.mu_layer = nn.Linear(hidden_dim + latent_dim, out_dim)
        self.log_std_layer = nn.Linear(hidden_dim + latent_dim, out_dim)

        apply_initialisation(self)

    def forward(self, latent: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.fc1(latent)
        x2 = self.fc2(torch.cat([x1, latent], dim=-1))
        features = self.fc3(torch.cat([x2, latent], dim=-1))
        trunk = torch.cat([features, latent], dim=-1)

        mu = self.mu_layer(trunk)
        log_std = self.log_std_layer(trunk)
        min_log_std, max_log_std = self.log_std_bounds
        log_std = torch.clamp(log_std, min_log_std, max_log_std)
        std = log_std.exp()

        dist = torch.distributions.Normal(mu, std)
        if deterministic:
            pre_tanh = mu
        else:
            pre_tanh = dist.rsample()

        tanh_action = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1 - tanh_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        action = tanh_action.view(latent.shape[0], self.num_waypoints, 2)
        radius = (action[..., 0] + 1.0) * 0.5 * self.max_radius
        yaw = action[..., 1] * self.max_yaw
        scaled = torch.stack([radius, yaw], dim=-1)
        return scaled, mu, log_prob


class CriticNetwork(nn.Module):
    """Q-function with D2RL-style skip connections."""

    def __init__(self, latent_dim: int = 128, action_dim: int = 20, hidden_dim: int = 256) -> None:
        super().__init__()
        self.fc1 = _d2rl_block(latent_dim + action_dim, hidden_dim)
        self.fc2 = _d2rl_block(hidden_dim + latent_dim + action_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim + latent_dim + action_dim, 1)
        apply_initialisation(self)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        flat_action = action.view(action.shape[0], -1)
        concat = torch.cat([latent, flat_action], dim=-1)
        x1 = self.fc1(concat)
        x2 = self.fc2(torch.cat([x1, concat], dim=-1))
        q = self.fc3(torch.cat([x2, concat], dim=-1))
        return q


class TwinQNetwork(nn.Module):
    """Container for two independent critics."""

    def __init__(self, latent_dim: int = 128, action_dim: int = 20, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = CriticNetwork(latent_dim, action_dim, hidden_dim)
        self.q2 = CriticNetwork(latent_dim, action_dim, hidden_dim)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(latent, action), self.q2(latent, action)


__all__ = ["WaypointActor", "TwinQNetwork", "CriticNetwork"]
