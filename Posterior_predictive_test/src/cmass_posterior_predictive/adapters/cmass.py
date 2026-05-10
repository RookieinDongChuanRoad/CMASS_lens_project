"""CMASS posterior-predictive registry entry.

This module is still a thin adapter, but it now owns the CMASS-specific
shared-parent diagnostics kernel.  Generic PPT workflow code reads runs,
selects posterior draws, and writes artifacts; this module maps a CMASS
posterior draw to replicated lens-population diagnostics.
"""

from __future__ import annotations

import math
from typing import Any

import numba as nb
import numpy as np

from cmass_lens_inference.canonical_dataset import load_canonical_inference_dataset
from cmass_lens_inference.canonical_context import lens_gamma_axis
from cmass_lens_inference.mass_definition import MassDefinition
from cmass_lens_inference.model_registry import get_model_definition
from cmass_lens_inference.models.cmass.posterior import (
    cmass_gamma_population_mean,
    unpack_cmass_theta,
)
from cmass_lens_inference.numba_backend.kernels.distributions import (
    skewnorm_sample,
    truncnorm_sample,
)
from cmass_lens_inference.numba_backend.kernels.interpolation import (
    interp_cross_section_theta_gamma,
    interp_sigma_unit_clip,
)
from cmass_lens_inference.numba_backend.kernels.lensing import theta_ein_arcsec
from cmass_lens_inference.types import ObservationRecord, ProfileSpec, RuntimeConfig

from ..interfaces import PPCContextBundle, PredictiveDefinition


MODEL_NAME = "cmass"
BACKEND_NAME = "numba_shared_parent"
ARTIFACT_SCHEMA_VERSION = "cmass_ppt_diagnostics_v1"
THETA_SAMPLE_SIZE = 23
SIGMA_SAMPLE_SIZE = 7
SIGMA_RELATIVE_NOISE = 0.0625
SUMMARY_STAT_NAMES = ("median", "std", "p10", "p90")
TREND_CATEGORY_NAMES = ("parent", "detectable", "selected")
LOG10_2PI = math.log10(2.0 * math.pi)
LOG10_4 = math.log10(4.0)
DEFAULT_NUMBA_DIAGNOSTICS_CHUNK_SIZE = 16


def _observations_from_canonical_dataset(dataset) -> list[ObservationRecord]:
    """
    Build legacy-compatible observation records from canonical CMASS input.

    Existing PPT artifact writers still consume ``ObservationRecord`` objects
    for observed theta/sigma summaries and raw-overlay fallbacks.  The canonical
    dataset already stores the required per-lens arrays, so this adapter keeps
    the compatibility conversion close to the CMASS context builder instead of
    leaving it in generic orchestration code.
    """

    observations: list[ObservationRecord] = []
    n_lens = len(dataset.lenses.lens_id)
    for lens_index in range(n_lens):
        num_sigma = int(dataset.lenses.num_sigma[lens_index])
        sigma_count = max(num_sigma, 0)
        sigma_observed = np.asarray(dataset.lenses.sigma_obs[lens_index, :sigma_count], dtype=float)
        sigma_error = np.asarray(dataset.lenses.sigma_err[lens_index, :sigma_count], dtype=float)
        has_s2 = int(dataset.mass_grids.has_s2[lens_index]) == 1
        observations.append(
            ObservationRecord(
                lens_id=str(dataset.lenses.lens_id[lens_index]),
                z_d=float(dataset.lenses.z_d[lens_index]),
                z_s=float(dataset.lenses.z_s[lens_index]),
                log_stellar_mass_obs=float(dataset.lenses.log_mstar_obs[lens_index]),
                log_stellar_mass_err=float(dataset.lenses.log_mstar_err[lens_index]),
                n_observed=float(dataset.lenses.n_obs[lens_index]),
                effective_radius_arcsec=math.nan,
                log_effective_radius_obs=float(dataset.lenses.log_re_obs[lens_index]),
                einstein_radius_arcsec=float(dataset.lenses.theta_e_obs[lens_index]),
                num_sigma=num_sigma,
                sigma_observed=sigma_observed,
                sigma_error=sigma_error,
                gamma_grid_17=np.asarray(lens_gamma_axis(dataset.mass_grids.gamma_grid, lens_index), dtype=float),
                mass_grid_17=np.asarray(dataset.mass_grids.log_enclosed_mass_grid[lens_index], dtype=float),
                dmass_dthetaein_grid_17=np.asarray(dataset.mass_grids.dmass_dthetaein_grid[lens_index], dtype=float),
                s2_grid_17=(
                    np.asarray(dataset.mass_grids.s2_grid[lens_index], dtype=float)
                    if has_s2
                    else None
                ),
            )
        )
    return observations


def build_context(runtime_config: RuntimeConfig) -> PPCContextBundle:
    """
    Build the canonical CMASS context consumed by current PPT diagnostics.

    The numerical context comes from the same `ModelDefinition` path used by
    production inference.  PPT still reconstructs `ObservationRecord` objects
    for legacy artifact writers, but it no longer chooses the CMASS
    preprocessing function itself; that selection belongs to the inference
    model registry.
    """

    if runtime_config.data.inference_dataset_path is None:
        raise ValueError("CMASS predictive diagnostics require data.inference_dataset_path.")

    model_definition = get_model_definition(runtime_config.model.name)
    compiled_model = model_definition.build_compiled_model(runtime_config)
    dataset = load_canonical_inference_dataset(
        runtime_config.data.inference_dataset_path,
        expected_unit_convention=runtime_config.unit_convention,
        expected_h_ref=runtime_config.h_ref,
        expected_profile_name=compiled_model.profile.name,
        expected_mass_definition_label=runtime_config.mass_definition.label,
        required_capabilities=model_definition.required_capabilities,
    )
    return (
        compiled_model.context,
        compiled_model.profile,
        compiled_model.cross_section_grid,
        compiled_model.cosmology,
        None,
        _observations_from_canonical_dataset(dataset),
    )



@nb.njit(cache=True)
def _numba_percentile_from_sorted(sorted_values: np.ndarray, percentile: float) -> float:
    """
    Return a NumPy-compatible linear percentile from an already sorted sample.

    Numba does not support every `np.percentile` mode consistently across
    versions.  The PPC summaries only need NumPy's default linear interpolation,
    so keeping the formula local makes the kernel behavior explicit.
    """

    n_values = sorted_values.shape[0]
    if n_values == 0:
        return math.nan
    if n_values == 1:
        return float(sorted_values[0])
    position = (percentile / 100.0) * (n_values - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    weight = position - lower_index
    return float(sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight)


@nb.njit(cache=True)
def _numba_summary_statistics(values: np.ndarray) -> np.ndarray:
    """Return median, sample standard deviation, p10, and p90 for one replicate."""

    n_values = values.shape[0]
    result = np.empty(4, dtype=np.float64)
    sorted_values = np.sort(values.copy())
    result[0] = _numba_percentile_from_sorted(sorted_values, 50.0)
    if n_values <= 1:
        result[1] = math.nan
    else:
        mean_value = 0.0
        for index in range(n_values):
            mean_value += values[index]
        mean_value /= n_values
        variance_sum = 0.0
        for index in range(n_values):
            diff = values[index] - mean_value
            variance_sum += diff * diff
        result[1] = math.sqrt(variance_sum / (n_values - 1))
    result[2] = _numba_percentile_from_sorted(sorted_values, 10.0)
    result[3] = _numba_percentile_from_sorted(sorted_values, 90.0)
    return result


@nb.njit(cache=True)
def _numba_draw_weighted_index(weights: np.ndarray, total_weight: float, unit_value: float) -> int:
    """
    Draw one parent-population index from non-negative selection weights.

    If the posterior draw produces no selectable systems, the PPC still needs a
    well-defined replicate shape.  The fallback is a uniform draw from the full
    parent pool, which matches the historical vectorized path's uniform-probability
    branch when the selected weight sum is zero.
    """

    n_values = weights.shape[0]
    clipped_unit = min(max(unit_value, 0.0), 1.0 - 1.0e-15)
    if total_weight <= 0.0 or not math.isfinite(total_weight):
        return min(int(clipped_unit * n_values), n_values - 1)

    target = clipped_unit * total_weight
    running_total = 0.0
    last_positive_index = n_values - 1
    for index in range(n_values):
        weight = weights[index]
        if math.isfinite(weight) and weight > 0.0:
            running_total += weight
            last_positive_index = index
            if running_total >= target:
                return index
    return last_positive_index


@nb.njit(cache=True)
def _numba_logistic_discovery_probability(theta_ein: float, theta0: float, loga: float) -> float:
    """
    Return the discovery probability used by the CMASS selection model.

    The exponent clamp mirrors the old vectorized implementation and prevents
    overflow for high-slope posterior draws without changing the saturated
    probability limit.
    """

    exponent = -(10.0**loga) * (theta_ein - theta0)
    if exponent < -60.0:
        exponent = -60.0
    elif exponent > 60.0:
        exponent = 60.0
    return 1.0 / (1.0 + math.exp(exponent))


@nb.njit(cache=True)
def _numba_reduce_population_to_bins(
    x_values: np.ndarray,
    y_values: np.ndarray,
    bin_edges: np.ndarray,
    detectable_weights: np.ndarray,
    selected_weights: np.ndarray,
    parent_means: np.ndarray,
    detectable_means: np.ndarray,
    selected_means: np.ndarray,
    parent_counts: np.ndarray,
    detectable_weight_sums: np.ndarray,
    selected_weight_sums: np.ndarray,
) -> None:
    """
    Reduce one parent population into parent/detectable/selected binned means.

    The reduction is intentionally written as explicit loops rather than a
    generic helper returning temporary masks.  These arrays are large in real
    PPC runs, and avoiding mask materialization keeps the Numba backend close to
    the production likelihood style.
    """

    n_bins = bin_edges.shape[0] - 1
    n_values = x_values.shape[0]
    for bin_index in range(n_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]
        parent_count = 0
        parent_sum = 0.0
        detectable_weight_sum = 0.0
        detectable_value_sum = 0.0
        selected_weight_sum = 0.0
        selected_value_sum = 0.0

        for value_index in range(n_values):
            x_value = x_values[value_index]
            y_value = y_values[value_index]
            if not (math.isfinite(x_value) and math.isfinite(y_value)):
                continue
            in_bin = x_value >= lower and (x_value <= upper if bin_index == n_bins - 1 else x_value < upper)
            if not in_bin:
                continue

            parent_count += 1
            parent_sum += y_value

            detectable_weight = detectable_weights[value_index]
            if math.isfinite(detectable_weight) and detectable_weight > 0.0:
                detectable_weight_sum += detectable_weight
                detectable_value_sum += y_value * detectable_weight

            selected_weight = selected_weights[value_index]
            if math.isfinite(selected_weight) and selected_weight > 0.0:
                selected_weight_sum += selected_weight
                selected_value_sum += y_value * selected_weight

        parent_counts[bin_index] = parent_count
        parent_means[bin_index] = parent_sum / parent_count if parent_count > 0 else math.nan
        detectable_weight_sums[bin_index] = detectable_weight_sum
        detectable_means[bin_index] = (
            detectable_value_sum / detectable_weight_sum
            if detectable_weight_sum > 0.0
            else math.nan
        )
        selected_weight_sums[bin_index] = selected_weight_sum
        selected_means[bin_index] = (
            selected_value_sum / selected_weight_sum
            if selected_weight_sum > 0.0
            else math.nan
        )


@nb.njit(cache=True, parallel=True)
def _shared_parent_diagnostics_numba_chunk(
    theta_chunk: np.ndarray,
    parent_normals: np.ndarray,
    theta_uniforms: np.ndarray,
    sigma_uniforms: np.ndarray,
    sigma_noise: np.ndarray,
    scalar_context: np.ndarray,
    z_grid: np.ndarray,
    chi_kpc_grid: np.ndarray,
    cs_theta_e_axis: np.ndarray,
    cs_gamma_grid: np.ndarray,
    cs_cross_section_grid: np.ndarray,
    sigma_gamma_axis: np.ndarray,
    sigma_zd_axis: np.ndarray,
    sigma_log_re_axis: np.ndarray,
    sigma_n_axis: np.ndarray,
    sigma_unit_grid: np.ndarray,
    mass_bin_edges: np.ndarray,
    sigma_star_bin_edges: np.ndarray,
    log_re_bin_edges: np.ndarray,
    delta_r_bin_edges: np.ndarray,
    use_sersic_index: int,
    gamma_mode_code: int,
    has_n_axis: int,
) -> tuple[np.ndarray, ...]:
    """
    Generate Numba shared-parent diagnostics for one posterior-draw chunk.

    Random numbers are generated outside the kernel and passed in as dense
    arrays.  That keeps reproducibility under Python's `Generator` contract and
    leaves this function responsible only for deterministic physics, selection,
    weighted resampling, and bin reductions.
    """

    n_draws = theta_chunk.shape[0]
    parent_sample_size = parent_normals.shape[1]
    n_mass_bins = mass_bin_edges.shape[0] - 1
    n_sigma_star_bins = sigma_star_bin_edges.shape[0] - 1
    n_log_re_bins = log_re_bin_edges.shape[0] - 1
    n_delta_r_bins = delta_r_bin_edges.shape[0] - 1

    theta_theta_ein = np.empty((n_draws, THETA_SAMPLE_SIZE), dtype=np.float64)
    theta_gamma = np.empty((n_draws, THETA_SAMPLE_SIZE), dtype=np.float64)
    theta_zd = np.empty((n_draws, THETA_SAMPLE_SIZE), dtype=np.float64)
    theta_zs = np.empty((n_draws, THETA_SAMPLE_SIZE), dtype=np.float64)
    theta_mass = np.empty((n_draws, THETA_SAMPLE_SIZE), dtype=np.float64)
    theta_re_kpc = np.empty((n_draws, THETA_SAMPLE_SIZE), dtype=np.float64)
    theta_n = np.empty((n_draws, THETA_SAMPLE_SIZE), dtype=np.float64)

    sigma_sigma = np.empty((n_draws, SIGMA_SAMPLE_SIZE), dtype=np.float64)
    sigma_theta_ein = np.empty((n_draws, SIGMA_SAMPLE_SIZE), dtype=np.float64)
    sigma_gamma = np.empty((n_draws, SIGMA_SAMPLE_SIZE), dtype=np.float64)
    sigma_zd = np.empty((n_draws, SIGMA_SAMPLE_SIZE), dtype=np.float64)
    sigma_zs = np.empty((n_draws, SIGMA_SAMPLE_SIZE), dtype=np.float64)
    sigma_mass = np.empty((n_draws, SIGMA_SAMPLE_SIZE), dtype=np.float64)
    sigma_re_kpc = np.empty((n_draws, SIGMA_SAMPLE_SIZE), dtype=np.float64)
    sigma_n = np.empty((n_draws, SIGMA_SAMPLE_SIZE), dtype=np.float64)

    theta_stats = np.empty((n_draws, len(SUMMARY_STAT_NAMES)), dtype=np.float64)
    sigma_stats = np.empty((n_draws, len(SUMMARY_STAT_NAMES)), dtype=np.float64)

    mass_trend = np.empty((n_draws, len(TREND_CATEGORY_NAMES), n_mass_bins), dtype=np.float64)
    gamma_trend = np.empty((n_draws, len(TREND_CATEGORY_NAMES), n_mass_bins), dtype=np.float64)
    sigma_trend = np.empty((n_draws, len(TREND_CATEGORY_NAMES), n_mass_bins), dtype=np.float64)
    parent_bin_counts = np.empty((n_draws, n_mass_bins), dtype=np.int64)
    detectable_weight_sums = np.empty((n_draws, n_mass_bins), dtype=np.float64)
    selected_weight_sums = np.empty((n_draws, n_mass_bins), dtype=np.float64)

    gamma_logre_trend = np.empty((n_draws, len(TREND_CATEGORY_NAMES), n_log_re_bins), dtype=np.float64)
    gamma_logre_parent_counts = np.empty((n_draws, n_log_re_bins), dtype=np.int64)
    gamma_logre_detectable_sums = np.empty((n_draws, n_log_re_bins), dtype=np.float64)
    gamma_logre_selected_sums = np.empty((n_draws, n_log_re_bins), dtype=np.float64)

    gamma_sigma_star_trend = np.empty((n_draws, len(TREND_CATEGORY_NAMES), n_sigma_star_bins), dtype=np.float64)
    gamma_sigma_star_parent_counts = np.empty((n_draws, n_sigma_star_bins), dtype=np.int64)
    gamma_sigma_star_detectable_sums = np.empty((n_draws, n_sigma_star_bins), dtype=np.float64)
    gamma_sigma_star_selected_sums = np.empty((n_draws, n_sigma_star_bins), dtype=np.float64)

    gamma_delta_r_trend = np.empty((n_draws, len(TREND_CATEGORY_NAMES), n_delta_r_bins), dtype=np.float64)
    gamma_delta_r_parent_counts = np.empty((n_draws, n_delta_r_bins), dtype=np.int64)
    gamma_delta_r_detectable_sums = np.empty((n_draws, n_delta_r_bins), dtype=np.float64)
    gamma_delta_r_selected_sums = np.empty((n_draws, n_delta_r_bins), dtype=np.float64)

    mass_radius_kpc = scalar_context[0]
    n_fixed = scalar_context[1]
    mu_n0 = scalar_context[2]
    beta_n = scalar_context[3]
    sigma_n_population = scalar_context[4]
    mass_function_loc = scalar_context[5]
    mass_function_scale = scalar_context[6]
    mass_function_alpha = scalar_context[7]
    mu_r0 = scalar_context[8]
    beta_r = scalar_context[9]
    sigma_r = scalar_context[10]
    nu_r = scalar_context[11]
    mu_d = scalar_context[12]
    sigma_d = scalar_context[13]
    gamma_trunc_low = scalar_context[14]
    gamma_trunc_high = scalar_context[15]
    stellar_mass_pivot = scalar_context[16]
    mass_log_physical_offset = scalar_context[17]

    for draw_index in nb.prange(n_draws):
        theta = theta_chunk[draw_index]
        (
            mu5_0,
            beta5,
            xi5,
            sigma5,
            mu_gamma_0,
            beta_gamma,
            xi_gamma,
            beta_sigma_star_gamma,
            sigma_gamma_value,
            mu_zs,
            sigma_zs_value,
            theta0,
            loga,
        ) = unpack_cmass_theta(theta, gamma_mode_code)

        log_mstar_values = np.empty(parent_sample_size, dtype=np.float64)
        log_enclosed_mass_values = np.empty(parent_sample_size, dtype=np.float64)
        gamma_values = np.empty(parent_sample_size, dtype=np.float64)
        theta_ein_values = np.empty(parent_sample_size, dtype=np.float64)
        zd_values = np.empty(parent_sample_size, dtype=np.float64)
        zs_values = np.empty(parent_sample_size, dtype=np.float64)
        log_re_values = np.empty(parent_sample_size, dtype=np.float64)
        re_kpc_values = np.empty(parent_sample_size, dtype=np.float64)
        n_values = np.empty(parent_sample_size, dtype=np.float64)
        log_sigma_star_values = np.empty(parent_sample_size, dtype=np.float64)
        delta_r_values = np.empty(parent_sample_size, dtype=np.float64)
        sigma_model_values = np.empty(parent_sample_size, dtype=np.float64)
        detectable_weights = np.empty(parent_sample_size, dtype=np.float64)
        selected_weights = np.empty(parent_sample_size, dtype=np.float64)

        total_selected_weight = 0.0
        for parent_index in range(parent_sample_size):
            normals = parent_normals[draw_index, parent_index]
            zd_value = mu_d + sigma_d * normals[0]
            zs_value = mu_zs + sigma_zs_value * normals[1]
            log_mstar_value = skewnorm_sample(
                mass_function_loc,
                mass_function_scale,
                mass_function_alpha,
                normals[2],
                normals[3],
            )
            mstar_shift = log_mstar_value - stellar_mass_pivot

            if use_sersic_index == 1:
                log_n_value = mu_n0 + beta_n * mstar_shift + sigma_n_population * normals[4]
                n_value = 10.0**log_n_value
                mu_r_value = mu_r0 + beta_r * mstar_shift + nu_r * (
                    math.log10(max(n_value, 1.0e-12)) - LOG10_4
                )
                re_noise = normals[5]
                mass_noise = normals[6]
            else:
                n_value = n_fixed
                mu_r_value = mu_r0 + beta_r * mstar_shift
                re_noise = normals[4]
                mass_noise = normals[5]

            log_re_value = mu_r_value + sigma_r * re_noise
            delta_r_value = log_re_value - mu_r_value
            log_enclosed_mass_value = mu5_0 + beta5 * mstar_shift + xi5 * delta_r_value + sigma5 * mass_noise
            log_sigma_star_value = log_mstar_value - LOG10_2PI - 2.0 * log_re_value
            mu_gamma_value = cmass_gamma_population_mean(
                mu_gamma_0,
                beta_gamma,
                xi_gamma,
                beta_sigma_star_gamma,
                mstar_shift,
                delta_r_value,
                log_sigma_star_value - 9.0,
                gamma_mode_code,
            )
            gamma_value = truncnorm_sample(
                mu_gamma_value,
                sigma_gamma_value,
                gamma_trunc_low,
                gamma_trunc_high,
                normals[7],
            )
            re_kpc_value = 10.0**log_re_value
            theta_ein_value = theta_ein_arcsec(
                zd_value,
                zs_value,
                log_enclosed_mass_value,
                gamma_value,
                z_grid,
                chi_kpc_grid,
                mass_radius_kpc,
                mass_log_physical_offset,
            )
            cross_section = interp_cross_section_theta_gamma(
                theta_ein_value,
                gamma_value,
                cs_theta_e_axis,
                cs_gamma_grid,
                cs_cross_section_grid,
            )
            discovery_probability = _numba_logistic_discovery_probability(theta_ein_value, theta0, loga)
            valid_geometry = (
                math.isfinite(gamma_value)
                and math.isfinite(log_enclosed_mass_value)
                and math.isfinite(re_kpc_value)
                and theta_ein_value > 0.0
                and zs_value > zd_value
                and zd_value > 0.0
                and zs_value > 0.0
                and cross_section > 0.0
            )
            detectable_weight = cross_section if valid_geometry else 0.0
            selected_weight = (
                detectable_weight * discovery_probability
                if valid_geometry and math.isfinite(discovery_probability) and discovery_probability > 0.0
                else 0.0
            )

            sigma_unit = interp_sigma_unit_clip(
                gamma_value,
                zd_value,
                log_re_value,
                n_value,
                sigma_gamma_axis,
                sigma_zd_axis,
                sigma_log_re_axis,
                sigma_n_axis,
                sigma_unit_grid,
                has_n_axis,
            )
            sigma_model = math.sqrt(max(sigma_unit * (10.0**log_enclosed_mass_value), 1.0e-30))

            log_mstar_values[parent_index] = log_mstar_value
            log_enclosed_mass_values[parent_index] = log_enclosed_mass_value
            gamma_values[parent_index] = gamma_value
            theta_ein_values[parent_index] = theta_ein_value
            zd_values[parent_index] = zd_value
            zs_values[parent_index] = zs_value
            log_re_values[parent_index] = log_re_value
            re_kpc_values[parent_index] = re_kpc_value
            n_values[parent_index] = n_value
            log_sigma_star_values[parent_index] = log_sigma_star_value
            delta_r_values[parent_index] = delta_r_value
            sigma_model_values[parent_index] = sigma_model
            detectable_weights[parent_index] = detectable_weight
            selected_weights[parent_index] = selected_weight
            total_selected_weight += selected_weight

        theta_sample_values = np.empty(THETA_SAMPLE_SIZE, dtype=np.float64)
        sigma_sample_values = np.empty(SIGMA_SAMPLE_SIZE, dtype=np.float64)
        for sample_index in range(THETA_SAMPLE_SIZE):
            selected_index = _numba_draw_weighted_index(
                selected_weights,
                total_selected_weight,
                theta_uniforms[draw_index, sample_index],
            )
            theta_theta_ein[draw_index, sample_index] = theta_ein_values[selected_index]
            theta_gamma[draw_index, sample_index] = gamma_values[selected_index]
            theta_zd[draw_index, sample_index] = zd_values[selected_index]
            theta_zs[draw_index, sample_index] = zs_values[selected_index]
            theta_mass[draw_index, sample_index] = log_enclosed_mass_values[selected_index]
            theta_re_kpc[draw_index, sample_index] = re_kpc_values[selected_index]
            theta_n[draw_index, sample_index] = n_values[selected_index]
            theta_sample_values[sample_index] = theta_ein_values[selected_index]

        for sample_index in range(SIGMA_SAMPLE_SIZE):
            selected_index = _numba_draw_weighted_index(
                selected_weights,
                total_selected_weight,
                sigma_uniforms[draw_index, sample_index],
            )
            sigma_value = sigma_model_values[selected_index]
            sigma_replicate = sigma_value + SIGMA_RELATIVE_NOISE * sigma_value * sigma_noise[draw_index, sample_index]
            sigma_sigma[draw_index, sample_index] = sigma_replicate
            sigma_theta_ein[draw_index, sample_index] = theta_ein_values[selected_index]
            sigma_gamma[draw_index, sample_index] = gamma_values[selected_index]
            sigma_zd[draw_index, sample_index] = zd_values[selected_index]
            sigma_zs[draw_index, sample_index] = zs_values[selected_index]
            sigma_mass[draw_index, sample_index] = log_enclosed_mass_values[selected_index]
            sigma_re_kpc[draw_index, sample_index] = re_kpc_values[selected_index]
            sigma_n[draw_index, sample_index] = n_values[selected_index]
            sigma_sample_values[sample_index] = sigma_replicate

        theta_stats[draw_index] = _numba_summary_statistics(theta_sample_values)
        sigma_stats[draw_index] = _numba_summary_statistics(sigma_sample_values)

        _numba_reduce_population_to_bins(
            log_mstar_values,
            log_enclosed_mass_values,
            mass_bin_edges,
            detectable_weights,
            selected_weights,
            mass_trend[draw_index, 0],
            mass_trend[draw_index, 1],
            mass_trend[draw_index, 2],
            parent_bin_counts[draw_index],
            detectable_weight_sums[draw_index],
            selected_weight_sums[draw_index],
        )
        _numba_reduce_population_to_bins(
            log_mstar_values,
            gamma_values,
            mass_bin_edges,
            detectable_weights,
            selected_weights,
            gamma_trend[draw_index, 0],
            gamma_trend[draw_index, 1],
            gamma_trend[draw_index, 2],
            parent_bin_counts[draw_index],
            detectable_weight_sums[draw_index],
            selected_weight_sums[draw_index],
        )
        _numba_reduce_population_to_bins(
            log_mstar_values,
            sigma_model_values,
            mass_bin_edges,
            detectable_weights,
            selected_weights,
            sigma_trend[draw_index, 0],
            sigma_trend[draw_index, 1],
            sigma_trend[draw_index, 2],
            parent_bin_counts[draw_index],
            detectable_weight_sums[draw_index],
            selected_weight_sums[draw_index],
        )
        _numba_reduce_population_to_bins(
            log_re_values,
            gamma_values,
            log_re_bin_edges,
            detectable_weights,
            selected_weights,
            gamma_logre_trend[draw_index, 0],
            gamma_logre_trend[draw_index, 1],
            gamma_logre_trend[draw_index, 2],
            gamma_logre_parent_counts[draw_index],
            gamma_logre_detectable_sums[draw_index],
            gamma_logre_selected_sums[draw_index],
        )
        _numba_reduce_population_to_bins(
            log_sigma_star_values,
            gamma_values,
            sigma_star_bin_edges,
            detectable_weights,
            selected_weights,
            gamma_sigma_star_trend[draw_index, 0],
            gamma_sigma_star_trend[draw_index, 1],
            gamma_sigma_star_trend[draw_index, 2],
            gamma_sigma_star_parent_counts[draw_index],
            gamma_sigma_star_detectable_sums[draw_index],
            gamma_sigma_star_selected_sums[draw_index],
        )
        _numba_reduce_population_to_bins(
            delta_r_values,
            gamma_values,
            delta_r_bin_edges,
            detectable_weights,
            selected_weights,
            gamma_delta_r_trend[draw_index, 0],
            gamma_delta_r_trend[draw_index, 1],
            gamma_delta_r_trend[draw_index, 2],
            gamma_delta_r_parent_counts[draw_index],
            gamma_delta_r_detectable_sums[draw_index],
            gamma_delta_r_selected_sums[draw_index],
        )

    return (
        theta_theta_ein,
        theta_gamma,
        theta_zd,
        theta_zs,
        theta_mass,
        theta_re_kpc,
        theta_n,
        sigma_sigma,
        sigma_theta_ein,
        sigma_gamma,
        sigma_zd,
        sigma_zs,
        sigma_mass,
        sigma_re_kpc,
        sigma_n,
        theta_stats,
        sigma_stats,
        mass_trend,
        gamma_trend,
        sigma_trend,
        parent_bin_counts,
        detectable_weight_sums,
        selected_weight_sums,
        gamma_logre_trend,
        gamma_logre_parent_counts,
        gamma_logre_detectable_sums,
        gamma_logre_selected_sums,
        gamma_sigma_star_trend,
        gamma_sigma_star_parent_counts,
        gamma_sigma_star_detectable_sums,
        gamma_sigma_star_selected_sums,
        gamma_delta_r_trend,
        gamma_delta_r_parent_counts,
        gamma_delta_r_detectable_sums,
        gamma_delta_r_selected_sums,
    )


def _diagnostics_scalar_context(context) -> np.ndarray:
    """Pack scalar context values consumed by the Numba diagnostics kernel."""

    return np.asarray(
        [
            context.mass_radius_kpc,
            context.n_fixed,
            context.mu_n0,
            context.beta_n,
            context.sigma_n,
            context.mass_function_loc,
            context.mass_function_scale,
            context.mass_function_alpha,
            context.mu_r0,
            context.beta_r,
            context.sigma_r,
            context.nu_r,
            context.mu_d,
            context.sigma_d,
            context.gamma_trunc_low,
            context.gamma_trunc_high,
            context.stellar_mass_pivot,
            context.mass_log_physical_offset,
        ],
        dtype=float,
    )


def _sigma_table_numba_arrays(sigma_table: SigmaUnitTable) -> dict[str, np.ndarray | int]:
    """
    Convert a loaded sigma table into the 4D shape expected by Numba kernels.

    PPC accepts both observed-aperture bundles with a real redshift axis and
    within-Re-style tables that are independent of redshift.  The shared
    interpolation primitive is simpler and faster when every table is presented
    as `(gamma, zd, logRe, n)`, so this helper injects singleton compatibility
    axes where the physical table omits them.
    """

    values = np.asarray(sigma_table.values, dtype=np.float64)
    if sigma_table.zd_axis is None:
        zd_axis = np.asarray([0.0], dtype=np.float64)
        values = values[:, None, ...]
    else:
        zd_axis = np.asarray(sigma_table.zd_axis, dtype=np.float64)
    if sigma_table.n_axis is None:
        n_axis = np.asarray([0.0], dtype=np.float64)
        values = values[..., None]
        has_n_axis = 0
    else:
        n_axis = np.asarray(sigma_table.n_axis, dtype=np.float64)
        has_n_axis = 1
    if values.ndim != 4:
        raise ValueError(
            "Sigma table must normalize to a 4D (gamma, zd, logRe, n) grid for the Numba PPC backend; "
            f"got shape {values.shape}."
        )
    return {
        "gamma_axis": np.asarray(sigma_table.gamma_axis, dtype=np.float64),
        "zd_axis": zd_axis,
        "log_re_axis": np.asarray(sigma_table.log_re_kpc_axis, dtype=np.float64),
        "n_axis": n_axis,
        "values": np.ascontiguousarray(values, dtype=np.float64),
        "has_n_axis": has_n_axis,
    }


_NUMBA_DIAGNOSTIC_ARRAY_NAMES = (
    "theta_theta_ein",
    "theta_gamma",
    "theta_zd",
    "theta_zs",
    "theta_mass",
    "theta_re_kpc",
    "theta_n",
    "sigma_sigma",
    "sigma_theta_ein",
    "sigma_gamma",
    "sigma_zd",
    "sigma_zs",
    "sigma_mass",
    "sigma_re_kpc",
    "sigma_n",
    "theta_stats",
    "sigma_stats",
    "mass_trend",
    "gamma_trend",
    "sigma_trend",
    "parent_bin_counts",
    "detectable_weight_sums",
    "selected_weight_sums",
    "gamma_logre_trend",
    "gamma_logre_parent_bin_counts",
    "gamma_logre_detectable_weight_sums",
    "gamma_logre_selected_weight_sums",
    "gamma_sigma_star_trend",
    "gamma_sigma_star_parent_bin_counts",
    "gamma_sigma_star_detectable_weight_sums",
    "gamma_sigma_star_selected_weight_sums",
    "gamma_delta_r_trend",
    "gamma_delta_r_parent_bin_counts",
    "gamma_delta_r_detectable_weight_sums",
    "gamma_delta_r_selected_weight_sums",
)


def _run_shared_parent_diagnostics_numba(
    posterior_draws: np.ndarray,
    profile: ProfileSpec,
    context,
    mass_definition: MassDefinition,
    sigma_table: SigmaUnitTable,
    mass_bin_edges: np.ndarray,
    sigma_star_bin_edges: np.ndarray,
    log_re_bin_edges: np.ndarray,
    delta_r_bin_edges: np.ndarray,
    parent_sample_size: int,
    random_seed: int,
) -> dict[str, Any]:
    """
    Execute the shared-parent Numba diagnostics kernel and adapt outputs.

    The returned structure mirrors the old PPC/trend chunk contracts so the
    artifact-writing code can stay mostly unchanged while the hot numerical
    path uses the same Numba primitive family as production inference.
    """

    del profile
    n_draws = int(posterior_draws.shape[0])
    rng = np.random.default_rng(int(random_seed))
    sigma_arrays = _sigma_table_numba_arrays(sigma_table)
    chunk_outputs: list[tuple[np.ndarray, ...]] = []
    chunk_size = max(1, min(DEFAULT_NUMBA_DIAGNOSTICS_CHUNK_SIZE, n_draws))

    for chunk_start in range(0, n_draws, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_draws)
        current_chunk_size = chunk_end - chunk_start
        parent_normals = rng.normal(
            size=(current_chunk_size, int(parent_sample_size), 8),
        ).astype(np.float64, copy=False)
        theta_uniforms = rng.random(size=(current_chunk_size, THETA_SAMPLE_SIZE)).astype(np.float64, copy=False)
        sigma_uniforms = rng.random(size=(current_chunk_size, SIGMA_SAMPLE_SIZE)).astype(np.float64, copy=False)
        sigma_noise = rng.normal(size=(current_chunk_size, SIGMA_SAMPLE_SIZE)).astype(np.float64, copy=False)
        chunk_outputs.append(
            _shared_parent_diagnostics_numba_chunk(
                np.ascontiguousarray(posterior_draws[chunk_start:chunk_end], dtype=np.float64),
                np.ascontiguousarray(parent_normals, dtype=np.float64),
                np.ascontiguousarray(theta_uniforms, dtype=np.float64),
                np.ascontiguousarray(sigma_uniforms, dtype=np.float64),
                np.ascontiguousarray(sigma_noise, dtype=np.float64),
                np.ascontiguousarray(_diagnostics_scalar_context(context), dtype=np.float64),
                np.ascontiguousarray(context.z_grid, dtype=np.float64),
                np.ascontiguousarray(context.chi_kpc_grid, dtype=np.float64),
                np.ascontiguousarray(context.cs_theta_e_axis, dtype=np.float64),
                np.ascontiguousarray(context.cs_gamma_grid, dtype=np.float64),
                np.ascontiguousarray(context.cs_cross_section_grid, dtype=np.float64),
                np.ascontiguousarray(sigma_arrays["gamma_axis"], dtype=np.float64),
                np.ascontiguousarray(sigma_arrays["zd_axis"], dtype=np.float64),
                np.ascontiguousarray(sigma_arrays["log_re_axis"], dtype=np.float64),
                np.ascontiguousarray(sigma_arrays["n_axis"], dtype=np.float64),
                np.ascontiguousarray(sigma_arrays["values"], dtype=np.float64),
                np.ascontiguousarray(mass_bin_edges, dtype=np.float64),
                np.ascontiguousarray(sigma_star_bin_edges, dtype=np.float64),
                np.ascontiguousarray(log_re_bin_edges, dtype=np.float64),
                np.ascontiguousarray(delta_r_bin_edges, dtype=np.float64),
                int(context.use_sersic_index),
                int(context.gamma_mode_code),
                int(sigma_arrays["has_n_axis"]),
            )
        )

    arrays = {
        array_name: np.concatenate(
            [chunk[index] for chunk in chunk_outputs],
            axis=0,
        )
        for index, array_name in enumerate(_NUMBA_DIAGNOSTIC_ARRAY_NAMES)
    }
    mass_label = mass_definition.label
    theta_latent = {
        "theta_ein": arrays["theta_theta_ein"],
        "gamma": arrays["theta_gamma"],
        "zd": arrays["theta_zd"],
        "zs": arrays["theta_zs"],
        mass_label: arrays["theta_mass"],
        "re_kpc": arrays["theta_re_kpc"],
        "n": arrays["theta_n"],
    }
    sigma_latent = {
        "sigma": arrays["sigma_sigma"],
        "theta_ein": arrays["sigma_theta_ein"],
        "gamma": arrays["sigma_gamma"],
        "zd": arrays["sigma_zd"],
        "zs": arrays["sigma_zs"],
        mass_label: arrays["sigma_mass"],
        "re_kpc": arrays["sigma_re_kpc"],
        "n": arrays["sigma_n"],
    }
    theta_replicated_stats = {
        name: arrays["theta_stats"][:, index]
        for index, name in enumerate(SUMMARY_STAT_NAMES)
    }
    sigma_replicated_stats = {
        name: arrays["sigma_stats"][:, index]
        for index, name in enumerate(SUMMARY_STAT_NAMES)
    }

    def trend_payload(array_name: str) -> dict[str, np.ndarray]:
        return {
            category_name: arrays[array_name][:, category_index, :]
            for category_index, category_name in enumerate(TREND_CATEGORY_NAMES)
        }

    trend_draws = {
        mass_label: trend_payload("mass_trend"),
        "gamma": trend_payload("gamma_trend"),
        "sigma_ap": trend_payload("sigma_trend"),
    }
    return {
        "theta_latent": theta_latent,
        "sigma_latent": sigma_latent,
        "theta_replicated_stats": theta_replicated_stats,
        "sigma_replicated_stats": sigma_replicated_stats,
        "trend_draws": trend_draws,
        "parent_bin_counts_draws": arrays["parent_bin_counts"],
        "detectable_weight_sums_draws": arrays["detectable_weight_sums"],
        "selected_weight_sums_draws": arrays["selected_weight_sums"],
        "gamma_vs_logre_draws": trend_payload("gamma_logre_trend"),
        "gamma_vs_logre_parent_bin_counts_draws": arrays["gamma_logre_parent_bin_counts"],
        "gamma_vs_logre_detectable_weight_sums_draws": arrays["gamma_logre_detectable_weight_sums"],
        "gamma_vs_logre_selected_weight_sums_draws": arrays["gamma_logre_selected_weight_sums"],
        "gamma_vs_sigma_star_draws": trend_payload("gamma_sigma_star_trend"),
        "gamma_vs_sigma_star_parent_bin_counts_draws": arrays["gamma_sigma_star_parent_bin_counts"],
        "gamma_vs_sigma_star_detectable_weight_sums_draws": arrays["gamma_sigma_star_detectable_weight_sums"],
        "gamma_vs_sigma_star_selected_weight_sums_draws": arrays["gamma_sigma_star_selected_weight_sums"],
        "gamma_vs_delta_r_draws": trend_payload("gamma_delta_r_trend"),
        "gamma_vs_delta_r_parent_bin_counts_draws": arrays["gamma_delta_r_parent_bin_counts"],
        "gamma_vs_delta_r_detectable_weight_sums_draws": arrays["gamma_delta_r_detectable_weight_sums"],
        "gamma_vs_delta_r_selected_weight_sums_draws": arrays["gamma_delta_r_selected_weight_sums"],
    }


def _build_cmass_trend_panel_order(mass_definition: MassDefinition) -> tuple[str, ...]:
    """
    Return the CMASS five-panel trend artifact order for the active mass label.

    The mass panel name depends on the configured mass aperture (`m5`, `m10`,
    etc.), while the remaining panels are the current CMASS diagnostic schema.
    Keeping this in the adapter prevents generic orchestration from hard-coding
    CMASS Fig. 8 semantics as the default for future models.
    """

    return (
        mass_definition.label,
        "gamma",
        "sigma_ap",
        "gamma_vs_sigma_star",
        "gamma_vs_logre_kpc",
    )


def get_predictive_definition() -> PredictiveDefinition:
    """
    Return the CMASS predictive definition exposed to generic PPT workflow code.

    ``sigma_table`` remains declared as an external input because the current
    CMASS diagnostics compare observed-aperture velocity dispersions against a
    profile/mass-definition-aware Jeans interpolation table.
    """

    return PredictiveDefinition(
        model_name=MODEL_NAME,
        backend=BACKEND_NAME,
        supported_diagnostics=("posterior_diagnostics", "posterior_predictive", "posterior_trends"),
        required_external_inputs=("sigma_table",),
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        build_context=build_context,
        run_diagnostics=_run_shared_parent_diagnostics_numba,
        trend_category_names=TREND_CATEGORY_NAMES,
        build_trend_panel_order=_build_cmass_trend_panel_order,
    )


__all__ = ["get_predictive_definition"]
