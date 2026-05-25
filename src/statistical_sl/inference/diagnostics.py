"""Shared diagnostics for the production Numba/emcee inference backend.

Model-owned posterior adapters build model-specific posterior values, but the
shape of emcee blobs is a backend/output contract.  Keeping the dtype and blob
constructors here prevents each model from growing its own HDF5-facing schema.
"""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from statistical_sl.inference.types import CompiledModel


NUMBA_DIAGNOSTIC_BLOB_DTYPE = np.dtype(
    [
        ("total_log_prob_seconds", np.float64),
        ("likelihood_seconds", np.float64),
        ("normalization_seconds", np.float64),
        ("fp_prior_seconds", np.float64),
        ("normalization_value", np.float64),
        ("fp_prior_log_term", np.float64),
        ("fpfit_mu", np.float64),
        ("fpfit_beta", np.float64),
        ("fpfit_xi", np.float64),
        ("fpfit_scatter", np.float64),
        ("backend", "S16"),
        ("kernel", "S32"),
        ("parallel_strategy", "S16"),
    ]
)


def build_timing_blob(
    *,
    total_log_prob_seconds: float,
    likelihood_seconds: float,
    normalization_seconds: float,
    fp_prior_seconds: float,
    normalization_value: float,
    fp_prior_log_term: float,
    fpfit_mu: float,
    fpfit_beta: float,
    fpfit_xi: float,
    fpfit_scatter: float,
    kernel: str,
    parallel_strategy: str,
) -> np.void:
    """
    Build one HDF5-safe structured diagnostic record.

    The same dtype is used on accepted and rejected proposals.  That stability
    is required because emcee's HDF backend appends blobs row-by-row and cannot
    tolerate schema drift during a run.
    """

    return np.array(
        (
            float(total_log_prob_seconds),
            float(likelihood_seconds),
            float(normalization_seconds),
            float(fp_prior_seconds),
            float(normalization_value),
            float(fp_prior_log_term),
            float(fpfit_mu),
            float(fpfit_beta),
            float(fpfit_xi),
            float(fpfit_scatter),
            b"numba",
            str(kernel).encode("utf-8"),
            str(parallel_strategy).encode("utf-8"),
        ),
        dtype=NUMBA_DIAGNOSTIC_BLOB_DTYPE,
    )[()]


def build_reject_result(
    total_start: float,
    compiled_model: CompiledModel,
    kernel: str,
) -> tuple[float, np.void]:
    """Return a consistent rejected log-probability result."""

    return (
        -np.inf,
        build_timing_blob(
            total_log_prob_seconds=perf_counter() - total_start,
            likelihood_seconds=0.0,
            normalization_seconds=0.0,
            fp_prior_seconds=0.0,
            normalization_value=0.0,
            fp_prior_log_term=0.0,
            fpfit_mu=math.nan,
            fpfit_beta=math.nan,
            fpfit_xi=math.nan,
            fpfit_scatter=math.nan,
            kernel=kernel,
            parallel_strategy=compiled_model.parallelism.strategy,
        ),
    )


__all__ = [
    "NUMBA_DIAGNOSTIC_BLOB_DTYPE",
    "build_reject_result",
    "build_timing_blob",
]
