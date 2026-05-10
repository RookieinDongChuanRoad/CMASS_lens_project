"""
Interpolation helpers compatible with the project's numerical requirements.

The scientific code needs a single, predictable interpolation policy:
- one-dimensional linear interpolation
- clip to the nearest boundary value outside the tabulated range
- no extrapolation
"""

from __future__ import annotations

import numpy as np


def clipped_linear_interp(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """
    Interpolate linearly and clip outside the tabulated range.

    Why we wrap `numpy.interp` instead of calling it inline:
    - it makes the project's interpolation rule explicit in one location
    - tests can lock the behavior once and reuse it everywhere
    - future backend-specific implementations can be swapped in behind the same
      function boundary if profiling shows a real need
    """

    return np.interp(x_new, x, y, left=float(y[0]), right=float(y[-1]))
