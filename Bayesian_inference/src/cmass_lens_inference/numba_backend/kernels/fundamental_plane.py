"""
Fundamental Plane summary kernels shared by production models.

The statistic is a compact numerical reducer: accumulate the
sufficient statistics for a one-predictor ordinary least-squares relation
between active stellar mass and model velocity dispersion.

Keeping the indices here gives downstream production code one auditable schema
for FP summaries instead of scattering positional constants across model files.
"""

from __future__ import annotations

import numba as nb
import numpy as np


FP_OLS_SUMMARY_SIZE = 6
FP_OLS_COUNT_INDEX = 0
FP_OLS_SUM_X1_INDEX = 1
FP_OLS_SUM_X1X1_INDEX = 2
FP_OLS_SUM_Y_INDEX = 3
FP_OLS_SUM_X1Y_INDEX = 4
FP_OLS_SUM_YY_INDEX = 5


@nb.njit(cache=True, inline="always")
def accumulate_fp_ols_summary(
    fp_summary: np.ndarray,
    mstar: float,
    log_sigma_model: float,
    pivot_mstar: float,
) -> None:
    """
    Accumulate one row of one-predictor FP sufficient statistics in-place.

    `x1` is the active stellar-mass coordinate after subtracting the requested
    pivot.  `log_sigma_model` is the dependent variable.  The six stored values
    are exactly the moments needed by the Python-side OLS solve, so the hot
    Monte Carlo loop avoids carrying every selected population draw back to the
    host.
    """

    x1 = mstar - pivot_mstar
    fp_summary[FP_OLS_COUNT_INDEX] += 1.0
    fp_summary[FP_OLS_SUM_X1_INDEX] += x1
    fp_summary[FP_OLS_SUM_X1X1_INDEX] += x1 * x1
    fp_summary[FP_OLS_SUM_Y_INDEX] += log_sigma_model
    fp_summary[FP_OLS_SUM_X1Y_INDEX] += x1 * log_sigma_model
    fp_summary[FP_OLS_SUM_YY_INDEX] += log_sigma_model * log_sigma_model


__all__ = [
    "FP_OLS_COUNT_INDEX",
    "FP_OLS_SUM_X1X1_INDEX",
    "FP_OLS_SUM_X1Y_INDEX",
    "FP_OLS_SUM_X1_INDEX",
    "FP_OLS_SUM_Y_INDEX",
    "FP_OLS_SUM_YY_INDEX",
    "FP_OLS_SUMMARY_SIZE",
    "accumulate_fp_ols_summary",
]
