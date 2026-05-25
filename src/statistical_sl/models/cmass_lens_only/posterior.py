"""CMASS lens-only posterior and model-owned Numba kernels."""

from __future__ import annotations

import math
from time import perf_counter

import numba as nb
import numpy as np

from statistical_sl.numerics.numba.kernels.distributions import normal_pdf
from statistical_sl.numerics.numba.kernels.integration import trapezoid_1d
from statistical_sl.numerics.numba.kernels.population import (
    gaussian_linear_mass_mean,
    sigma_star_linear_gamma_mean,
)
from statistical_sl.numerics.numba.kernels.selection_likelihood import (
    observed_sigma_likelihood,
    sigma_model_from_s2,
)
from statistical_sl.inference.types import CompiledModel
from .assembly import GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE
from statistical_sl.inference.diagnostics import build_timing_blob


THETA_DIMENSION = 9


@nb.njit(cache=True, inline="always")
def unpack_lens_only_theta(theta: np.ndarray) -> tuple[float, ...]:
    """Return the fixed scalar tuple used by the CMASS lens-only posterior."""

    return (
        theta[0],
        theta[1],
        theta[2],
        theta[3],
        theta[4],
        theta[5],
        theta[6],
        theta[7],
        theta[8],
    )


@nb.njit(cache=True, parallel=True, fastmath=True)
def log_likelihood_lenses_only_numba(
    theta: np.ndarray,
    mass_grid_int: np.ndarray,
    dmass_dthetaein_grid_int: np.ndarray,
    s2_grid_int: np.ndarray,
    has_s2: np.ndarray,
    num_sigma: np.ndarray,
    sigma_obs: np.ndarray,
    sigma_err: np.ndarray,
    mstar_grid: np.ndarray,
    mstar_shift11p4: np.ndarray,
    sigma_star_shift9p0_grid: np.ndarray,
    mstar_observation_density: np.ndarray,
    delta_r_grid: np.ndarray,
    gamma_grid_int: np.ndarray,
    gamma_mode_code: int,
) -> float:
    """
    Evaluate the observed-lens-sample likelihood.

    This kernel deliberately excludes every selection-correction term present
    in the default CMASS model: no source-redshift density, no lensing
    cross-section, no lens-finding probability, and no selection normalization.
    The observed velocity-dispersion likelihood remains inside this kernel so
    the full multiplicative likelihood order is visible in the hot path.
    """

    if theta.shape[0] != THETA_DIMENSION:
        return -np.inf

    (
        mu_mstar_lens,
        sigma_mstar_lens,
        mu5_0,
        beta5,
        xi5,
        sigma5,
        mu_gamma_0,
        beta_sigma_star_gamma,
        sigma_gamma,
    ) = unpack_lens_only_theta(theta)

    if sigma_mstar_lens <= 0.0 or sigma5 <= 0.0 or sigma_gamma <= 0.0:
        return -np.inf
    if gamma_mode_code != GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE:
        return -np.inf

    n_lens = mstar_grid.shape[0]
    n_gamma = gamma_grid_int.shape[0]
    n_mstar = mstar_grid.shape[1]
    ll_terms = np.zeros(n_lens, dtype=np.float64)
    valid = np.ones(n_lens, dtype=np.int64)

    for lens_index in nb.prange(n_lens):
        gamma_integrand = np.zeros(n_gamma, dtype=np.float64)
        for gamma_index in range(n_gamma):
            gamma = gamma_grid_int[gamma_index]
            log_enclosed_mass = mass_grid_int[lens_index, gamma_index]
            jacobian = abs(dmass_dthetaein_grid_int[lens_index, gamma_index])
            if jacobian <= 0.0:
                continue

            sigma_model = sigma_model_from_s2(
                s2_grid_int[lens_index, gamma_index],
                log_enclosed_mass,
            )
            sigma_probability = observed_sigma_likelihood(
                lens_index,
                num_sigma,
                has_s2,
                sigma_obs,
                sigma_err,
                sigma_model,
            )
            if sigma_probability <= 0.0:
                continue

            mstar_integrand = np.zeros(n_mstar, dtype=np.float64)
            for mstar_index in range(n_mstar):
                mstar_value = mstar_grid[lens_index, mstar_index]
                mstar_obs_density = mstar_observation_density[lens_index, mstar_index]
                if mstar_obs_density <= 0.0:
                    continue

                mstar_lens_density = normal_pdf(
                    mstar_value,
                    mu_mstar_lens,
                    sigma_mstar_lens,
                )
                if mstar_lens_density <= 0.0:
                    continue

                mu5 = gaussian_linear_mass_mean(
                    mu5_0,
                    beta5,
                    xi5,
                    mstar_shift11p4[lens_index, mstar_index],
                    delta_r_grid[lens_index, mstar_index],
                )
                mu_gamma = sigma_star_linear_gamma_mean(
                    mu_gamma_0,
                    beta_sigma_star_gamma,
                    sigma_star_shift9p0_grid[lens_index, mstar_index],
                )
                mstar_integrand[mstar_index] = (
                    mstar_obs_density
                    * mstar_lens_density
                    * normal_pdf(log_enclosed_mass, mu5, sigma5)
                    * normal_pdf(gamma, mu_gamma, sigma_gamma)
                )

            integrated_mstar = trapezoid_1d(mstar_integrand, mstar_grid[lens_index])
            gamma_integrand[gamma_index] = integrated_mstar * jacobian * sigma_probability

        lens_integral = trapezoid_1d(gamma_integrand, gamma_grid_int)
        if lens_integral <= 0.0 or not math.isfinite(lens_integral):
            valid[lens_index] = 0
            continue
        ll_terms[lens_index] = math.log(lens_integral)

    total = 0.0
    for lens_index in range(n_lens):
        if valid[lens_index] == 0:
            return -np.inf
        total += ll_terms[lens_index]
    return total


def log_prob(theta: np.ndarray, compiled_model: CompiledModel, total_start: float) -> tuple[float, np.void]:
    """
    Evaluate the CMASS lens-only posterior.

    Diagnostic fields retain the shared backend schema. `normalization_value`
    is set to 1.0 because this model has no selection-normalization integral.
    """

    lens_only_context = compiled_model.context
    base = lens_only_context.base

    likelihood_start = perf_counter()
    likelihood_value = log_likelihood_lenses_only_numba(
        theta=theta,
        mass_grid_int=base.mass_grid_int,
        dmass_dthetaein_grid_int=base.dmass_dthetaein_grid_int,
        s2_grid_int=base.s2_grid_int,
        has_s2=base.has_s2,
        num_sigma=base.num_sigma,
        sigma_obs=base.sigma_obs,
        sigma_err=base.sigma_err,
        mstar_grid=base.mstar_grid,
        mstar_shift11p4=base.mstar_shift11p4,
        sigma_star_shift9p0_grid=base.sigma_star_shift9p0_grid,
        mstar_observation_density=lens_only_context.mstar_observation_density,
        delta_r_grid=base.delta_r_grid,
        gamma_grid_int=base.gamma_grid_int,
        gamma_mode_code=base.gamma_mode_code,
    )
    likelihood_seconds = perf_counter() - likelihood_start
    total_seconds = perf_counter() - total_start
    blob = build_timing_blob(
        total_log_prob_seconds=total_seconds,
        likelihood_seconds=likelihood_seconds,
        normalization_seconds=0.0,
        fp_prior_seconds=0.0,
        normalization_value=1.0,
        fp_prior_log_term=0.0,
        fpfit_mu=math.nan,
        fpfit_beta=math.nan,
        fpfit_xi=math.nan,
        fpfit_scatter=math.nan,
        kernel="cmass_lens_only",
        parallel_strategy=compiled_model.parallelism.strategy,
    )

    if not np.isfinite(likelihood_value):
        return -np.inf, blob
    return float(likelihood_value), blob


__all__ = ["log_prob", "log_likelihood_lenses_only_numba", "unpack_lens_only_theta"]
