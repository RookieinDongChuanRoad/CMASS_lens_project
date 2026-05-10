"""
Fundamental Plane summary kernels shared by concrete posterior models.

The statistics are compact numerical reducers: they accumulate the sufficient
moments for ordinary least-squares Fundamental Plane fits without returning
every Monte Carlo draw to Python.  The reducers support both a one-predictor
mass--velocity-dispersion relation and a two-predictor relation using active
stellar mass plus a size residual.

Keeping the indices here gives downstream production code auditable schemas for
FP summaries instead of scattering positional constants across model files.
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

FP_OLS2_SUMMARY_SIZE = 10
FP_OLS2_COUNT_INDEX = 0
FP_OLS2_SUM_X1_INDEX = 1
FP_OLS2_SUM_X2_INDEX = 2
FP_OLS2_SUM_X1X1_INDEX = 3
FP_OLS2_SUM_X1X2_INDEX = 4
FP_OLS2_SUM_X2X2_INDEX = 5
FP_OLS2_SUM_Y_INDEX = 6
FP_OLS2_SUM_X1Y_INDEX = 7
FP_OLS2_SUM_X2Y_INDEX = 8
FP_OLS2_SUM_YY_INDEX = 9


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


@nb.njit(cache=True, inline="always")
def accumulate_fp_ols2_summary(
    fp_summary: np.ndarray,
    mstar: float,
    delta_r: float,
    log_sigma_model: float,
    pivot_mstar: float,
    sample_weight: float,
) -> None:
    """
    Accumulate one row of two-predictor FP sufficient statistics in-place.

    `x1` is active stellar mass relative to the FP pivot.  `x2` is the size
    residual relative to the model-owned size relation.  These are the two
    predictors needed before applying priors to the fitted scatter, intercept,
    and mass slope.  The count slot stores the sum of row weights, so callers can
    combine direct parent draws and weighted integration schemes through one
    compact summary contract.
    """

    x1 = mstar - pivot_mstar
    x2 = delta_r
    fp_summary[FP_OLS2_COUNT_INDEX] += sample_weight
    fp_summary[FP_OLS2_SUM_X1_INDEX] += sample_weight * x1
    fp_summary[FP_OLS2_SUM_X2_INDEX] += sample_weight * x2
    fp_summary[FP_OLS2_SUM_X1X1_INDEX] += sample_weight * x1 * x1
    fp_summary[FP_OLS2_SUM_X1X2_INDEX] += sample_weight * x1 * x2
    fp_summary[FP_OLS2_SUM_X2X2_INDEX] += sample_weight * x2 * x2
    fp_summary[FP_OLS2_SUM_Y_INDEX] += sample_weight * log_sigma_model
    fp_summary[FP_OLS2_SUM_X1Y_INDEX] += sample_weight * x1 * log_sigma_model
    fp_summary[FP_OLS2_SUM_X2Y_INDEX] += sample_weight * x2 * log_sigma_model
    fp_summary[FP_OLS2_SUM_YY_INDEX] += sample_weight * log_sigma_model * log_sigma_model


__all__ = [
    "FP_OLS2_COUNT_INDEX",
    "FP_OLS2_SUMMARY_SIZE",
    "FP_OLS2_SUM_X1X1_INDEX",
    "FP_OLS2_SUM_X1X2_INDEX",
    "FP_OLS2_SUM_X1Y_INDEX",
    "FP_OLS2_SUM_X1_INDEX",
    "FP_OLS2_SUM_X2X2_INDEX",
    "FP_OLS2_SUM_X2Y_INDEX",
    "FP_OLS2_SUM_X2_INDEX",
    "FP_OLS2_SUM_Y_INDEX",
    "FP_OLS2_SUM_YY_INDEX",
    "FP_OLS_COUNT_INDEX",
    "FP_OLS_SUM_X1X1_INDEX",
    "FP_OLS_SUM_X1Y_INDEX",
    "FP_OLS_SUM_X1_INDEX",
    "FP_OLS_SUM_Y_INDEX",
    "FP_OLS_SUM_YY_INDEX",
    "FP_OLS_SUMMARY_SIZE",
    "accumulate_fp_ols2_summary",
    "accumulate_fp_ols_summary",
]
