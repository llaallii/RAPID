"""Trajectory helper utilities."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from ..utils.geometry import cylindrical_to_cartesian


def cylindrical_deltas_to_cartesian(origin: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Convert cylindrical waypoint deltas to Cartesian coordinates."""

    waypoints: List[np.ndarray] = []
    current = origin.astype(np.float32)
    for radius, dpsi in deltas:
        cart = np.asarray(cylindrical_to_cartesian(radius, dpsi, current[2]), dtype=np.float32)
        current = current + cart
        waypoints.append(current.copy())
    return np.stack(waypoints, axis=0)


def bspline_interpolate(points: np.ndarray, num_samples: int = 50) -> np.ndarray:
    """Simple B-spline interpolation using numpy.

    TODO: swap with a dedicated spline library for better fidelity.
    """

    if len(points) < 4:
        return np.linspace(points[0], points[-1], num_samples)

    t = np.linspace(0.0, 1.0, len(points))
    samples = np.linspace(0.0, 1.0, num_samples)
    result = np.empty((num_samples, points.shape[1]), dtype=np.float32)
    for dim in range(points.shape[1]):
        result[:, dim] = np.interp(samples, t, points[:, dim])
    return result


__all__ = ["cylindrical_deltas_to_cartesian", "bspline_interpolate"]
