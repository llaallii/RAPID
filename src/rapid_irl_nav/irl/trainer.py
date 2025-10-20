"""Simplified training loop that wires together project components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ..data.datamodule import RapidNavDataModule
from ..model.actor_critic import TwinQNetwork, WaypointActor
from ..model.autoencoder import DepthDecoder
from ..model.encoders import DepthEncoder
from ..utils.logging import ExperimentLogger
from ..utils.seeding import seed_everything
from .sac import compute_target_value, entropy_loss, soft_update


@dataclass
class TrainerConfig:
    batch_size: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    horizon_steps: int = 10


class Trainer:
    """High level trainer that keeps the loop runnable."""

    def __init__(self, config_path: Path, log_dir: Path, seed: int = 42) -> None:
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        with config_path.open("r", encoding="utf-8") as handle:
            raw_cfg = yaml.safe_load(handle) or {}
        self.cfg = TrainerConfig(**raw_cfg)
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = ExperimentLogger("rapid_irl_nav", self.log_dir)
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    def run(self) -> None:
        seed_everything(self.seed)
        data = RapidNavDataModule(batch_size=self.cfg.batch_size, seed=self.seed)
        train_loader = data.train_dataloader()
        batch_iter = iter(train_loader)

        encoder = DepthEncoder().to(self.device)
        decoder = DepthDecoder().to(self.device)
        actor = WaypointActor().to(self.device)
        critics = TwinQNetwork().to(self.device)
        target_critics = TwinQNetwork().to(self.device)
        target_critics.load_state_dict(critics.state_dict())

        encoder_opt = torch.optim.Adam(encoder.parameters(), lr=self.cfg.lr)
        decoder_opt = torch.optim.Adam(decoder.parameters(), lr=self.cfg.lr)
        actor_opt = torch.optim.Adam(actor.parameters(), lr=self.cfg.lr)
        critic_opt = torch.optim.Adam(critics.parameters(), lr=self.cfg.lr)
        log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        alpha_opt = torch.optim.Adam([log_alpha], lr=self.cfg.lr)
        target_entropy = -float(actor.num_waypoints * 2)

        for step in range(self.cfg.horizon_steps):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader)
                batch = next(batch_iter)

            depth = batch["depth"].to(self.device)
            latent = encoder(depth)

            with torch.no_grad():
                next_latent = latent  # Placeholder; no transitions yet.
                next_action, _, next_log_prob = actor(next_latent)
                q1_target, q2_target = target_critics(next_latent, next_action)
                target_q = torch.min(q1_target, q2_target) - log_alpha.exp() * next_log_prob
                reward = torch.zeros_like(target_q)
                done = torch.zeros_like(target_q)
                q_backup = compute_target_value(reward, self.cfg.gamma, target_q, done)

            action, _, log_prob = actor(latent.detach())
            q1, q2 = critics(latent.detach(), action.detach())
            critic_loss = F.mse_loss(q1, q_backup) + F.mse_loss(q2, q_backup)

            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            action_pi, _, log_prob_pi = actor(latent)
            q1_pi, q2_pi = critics(latent, action_pi)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (log_alpha.exp() * log_prob_pi - q_pi).mean()

            recon = decoder(latent)
            recon_loss = F.mse_loss(recon, depth)

            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            actor_opt.zero_grad()
            total_actor = actor_loss + 0.1 * recon_loss
            total_actor.backward()
            actor_opt.step()
            encoder_opt.step()
            decoder_opt.step()

            alpha_loss = entropy_loss(log_alpha, log_prob_pi.detach(), target_entropy)
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()

            soft_update(target_critics, critics, self.cfg.tau)

            metrics = {
                "loss/critic": float(critic_loss.item()),
                "loss/actor": float(actor_loss.item()),
                "loss/recon": float(recon_loss.item()),
                "alpha": float(log_alpha.exp().item()),
            }
            self.logger.log_metrics(metrics, step)

        self.logger.info("Training loop completed (stub implementation).")
        self.logger.close()


__all__ = ["Trainer", "TrainerConfig"]
