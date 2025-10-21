"""Toy 2D environment and scripted expert for quick dataset generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .interfaces import Env, Expert, Obs


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the [-pi, pi] range."""

    return (angle + math.pi) % (2 * math.pi) - math.pi


def yaw_to_quat(yaw: float) -> np.ndarray:
    """Convert yaw to a quaternion [w, x, y, z]."""

    half = yaw * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def quat_to_yaw(quat: np.ndarray) -> float:
    """Extract yaw from a [w, x, y, z] quaternion."""

    return math.atan2(2.0 * quat[0] * quat[3], 1.0 - 2.0 * (quat[3] ** 2))


@dataclass
class ToyWorldConfig:
    dt: float = 0.1
    max_steps: int = 300
    success_radius: float = 0.2
    world_radius: float = 5.0


class ToyWorld(Env):
    """Simple point-mass world with synthetic observations."""

    def __init__(self, config: ToyWorldConfig | None = None, *, seed: int | None = None) -> None:
        self.config = config or ToyWorldConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(3, dtype=np.float32)  # x, y, yaw
        self.goal = np.zeros(3, dtype=np.float32)
        self.steps = 0
        self.collision_count = 0
        self.last_velocity = np.zeros(3, dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> Obs:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.seed is not None:
            self.rng = np.random.default_rng(self.seed)

        self.state = self._sample_start()
        self.goal = self._sample_goal()
        self.steps = 0
        self.collision_count = 0
        self.last_velocity = np.zeros(3, dtype=np.float32)
        return self._get_obs()

    def _sample_start(self) -> np.ndarray:
        radius = self.rng.uniform(0.0, self.config.world_radius * 0.25)
        theta = self.rng.uniform(-math.pi, math.pi)
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        yaw = self.rng.uniform(-math.pi, math.pi)
        return np.array([x, y, yaw], dtype=np.float32)

    def _sample_goal(self) -> np.ndarray:
        radius = self.rng.uniform(self.config.world_radius * 0.3, self.config.world_radius * 0.8)
        theta = self.rng.uniform(-math.pi, math.pi)
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        desired_yaw = math.atan2(y - self.state[1], x - self.state[0])
        return np.array([x, y, desired_yaw], dtype=np.float32)

    def _synthetic_depth(self) -> np.ndarray:
        distance = float(np.linalg.norm(self.state[:2] - self.goal[:2]))
        noise = self.rng.normal(scale=0.05)
        reading = np.clip(distance + noise, 0.0, self.config.world_radius)
        return np.full((1, 64, 64), reading, dtype=np.float32)

    def _get_obs(self) -> Obs:
        depth = self._synthetic_depth()
        quat = yaw_to_quat(float(self.state[2]))
        pos_xy = self.state[:2].astype(np.float32)
        goal = self.goal.astype(np.float32)
        vel = self.last_velocity.astype(np.float32)
        return Obs(depth=depth, vel=vel, quat=quat, pos_xy=pos_xy, goal=goal)

    def step(self, action: np.ndarray):
        if action.shape != (10, 2):
            raise ValueError(f"Action must have shape (10, 2), got {action.shape}")

        delta_r, delta_yaw = action[0].astype(np.float32)
        delta_r = float(np.clip(delta_r, -0.5, 0.5))
        delta_yaw = float(np.clip(delta_yaw, -math.pi / 4.0, math.pi / 4.0))

        new_yaw = wrap_angle(float(self.state[2]) + delta_yaw)
        dx_world = delta_r * math.cos(new_yaw)
        dy_world = delta_r * math.sin(new_yaw)

        new_x = float(self.state[0]) + dx_world
        new_y = float(self.state[1]) + dy_world
        distance_to_origin = math.hypot(new_x, new_y)
        if distance_to_origin > self.config.world_radius:
            scale = self.config.world_radius / distance_to_origin
            new_x *= scale
            new_y *= scale
            self.collision_count += 1

        self.state = np.array([new_x, new_y, new_yaw], dtype=np.float32)

        vx_body = delta_r / self.config.dt
        vy_body = 0.0
        wz = delta_yaw / self.config.dt
        self.last_velocity = np.array([vx_body, vy_body, wz], dtype=np.float32)

        obs = self._get_obs()
        distance_to_goal = float(np.linalg.norm(obs.pos_xy - obs.goal[:2]))
        reward = -distance_to_goal

        self.steps += 1
        success = distance_to_goal <= self.config.success_radius
        timeout = self.steps >= self.config.max_steps
        done = success or timeout
        info = {
            "collisions": self.collision_count,
            "distance_to_goal": distance_to_goal,
            "timeout": timeout,
        }
        return obs, reward, done, info


class ScriptedExpert(Expert):
    """Greedy expert that aligns yaw with the goal and marches forward."""

    def __init__(
        self,
        *,
        forward_gain: float = 0.5,
        yaw_gain: float = 1.5,
        max_delta_r: float = 0.5,
        max_delta_yaw: float = math.pi / 4.0,
    ) -> None:
        self.forward_gain = forward_gain
        self.yaw_gain = yaw_gain
        self.max_delta_r = max_delta_r
        self.max_delta_yaw = max_delta_yaw

    def act(self, obs: Obs) -> np.ndarray:
        current_yaw = quat_to_yaw(obs.quat)
        goal_pos = obs.goal[:2]
        to_goal = goal_pos - obs.pos_xy
        distance = float(np.linalg.norm(to_goal))
        desired_yaw = math.atan2(to_goal[1], to_goal[0])
        heading_error = wrap_angle(desired_yaw - current_yaw)

        delta_r = np.clip(self.forward_gain * distance, -self.max_delta_r, self.max_delta_r)
        delta_yaw = np.clip(self.yaw_gain * heading_error, -self.max_delta_yaw, self.max_delta_yaw)

        action = np.zeros((10, 2), dtype=np.float32)
        action[:, 0] = delta_r
        action[:, 1] = delta_yaw
        return action


__all__ = ["ToyWorld", "ToyWorldConfig", "ScriptedExpert"]
