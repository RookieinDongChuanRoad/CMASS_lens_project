"""Integration primitives used by production Numba kernels."""

from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True)
def trapezoid_1d(y: np.ndarray, x: np.ndarray) -> float:
    """Return an explicit trapezoid integral over one one-dimensional grid."""

    total = 0.0
    for index in range(x.shape[0] - 1):
        total += 0.5 * (y[index + 1] + y[index]) * (x[index + 1] - x[index])
    return total


__all__ = ["trapezoid_1d"]
