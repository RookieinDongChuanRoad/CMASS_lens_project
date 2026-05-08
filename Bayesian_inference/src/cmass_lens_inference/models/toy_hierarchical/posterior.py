"""Posterior assembly for the toy hierarchical model.

This adapter is model-owned, just like CMASS and Sonnenfeld production modules.
It uses the shared backend diagnostics schema and returns an emcee-compatible
structured blob, but no runner or sampler code needs to know the model exists.
"""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from ...numba_backend.diagnostics import build_timing_blob
from ...types import CompiledModel


def _gaussian_population_log_likelihood(
    *,
    population_mean: float,
    population_scatter: float,
    observed_values: np.ndarray,
    observed_errors: np.ndarray,
) -> float:
    """
    Evaluate a simple Gaussian hierarchical likelihood.

    The intrinsic scatter and measurement error add in quadrature.  This is the
    minimal population model needed to verify the production extension path:
    sampled hyper-parameters govern a latent population distribution, and fixed
    observation errors enter the per-datum likelihood.
    """

    if population_scatter <= 0.0 or not math.isfinite(population_scatter):
        return -np.inf

    total = 0.0
    log_2pi = math.log(2.0 * math.pi)
    for value, error in zip(observed_values, observed_errors, strict=True):
        variance = population_scatter * population_scatter + float(error) * float(error)
        if variance <= 0.0:
            return -np.inf
        residual = float(value) - population_mean
        total += -0.5 * (residual * residual / variance + math.log(variance) + log_2pi)
    return float(total)


def log_prob(theta: np.ndarray, compiled_model: CompiledModel, total_start: float) -> tuple[float, np.void]:
    """Evaluate the toy hierarchical posterior and build a standard blob."""

    likelihood_start = perf_counter()
    population_mean = float(theta[0])
    population_scatter = math.exp(float(theta[1]))
    context = compiled_model.context
    likelihood_value = _gaussian_population_log_likelihood(
        population_mean=population_mean,
        population_scatter=population_scatter,
        observed_values=context.observed_values,
        observed_errors=context.observed_errors,
    )
    likelihood_seconds = perf_counter() - likelihood_start

    return (
        likelihood_value,
        build_timing_blob(
            total_log_prob_seconds=perf_counter() - total_start,
            likelihood_seconds=likelihood_seconds,
            normalization_seconds=0.0,
            fp_prior_seconds=0.0,
            normalization_value=1.0,
            fp_prior_log_term=0.0,
            fpfit_mu=math.nan,
            fpfit_beta=math.nan,
            fpfit_xi=math.nan,
            fpfit_scatter=math.nan,
            kernel=compiled_model.config.model.name,
            parallel_strategy=compiled_model.parallelism.strategy,
        ),
    )


__all__ = ["log_prob"]
