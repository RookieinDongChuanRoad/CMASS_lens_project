"""Sonnenfeld/SLACS posterior-predictive registry entries.

The current hook is intentionally narrower than the mature CMASS adapter: it
uses the Sonnenfeld inference registry and canonical context, then materializes
the common diagnostic arrays expected by the existing PPT artifact writer.  It
does not reuse CMASS posterior helpers or CMASS gamma-mode logic.
"""

from __future__ import annotations

import math
from typing import Any

import numba as nb
import numpy as np

from cmass_lens_inference.canonical_context import lens_gamma_axis
from cmass_lens_inference.canonical_dataset import load_canonical_inference_dataset
from cmass_lens_inference.mass_definition import MassDefinition
from cmass_lens_inference.model_registry import get_model_definition
from cmass_lens_inference.numba_backend.kernels.interpolation import (
    interp_cross_section_theta_gamma,
    interp_sigma_unit_clip,
)
from cmass_lens_inference.numba_backend.kernels.lensing import theta_ein_arcsec
from cmass_lens_inference.numba_backend.kernels.selection import theta_e_est_from_sigma_proxy
from cmass_lens_inference.numba_backend.kernels.selection_likelihood import (
    cross_section_find_weight,
    sigma_model_from_s2,
)
from cmass_lens_inference.parallel import apply_thread_limits
from cmass_lens_inference.types import ObservationRecord, RuntimeConfig

from ..interfaces import DiagnosticsExecution, PPCContextBundle, PredictiveDefinition


BASE_MODEL_NAMES = ("sonnenfeld2024_slacs", "sonnenfeld2024_slacs_hunit")
SIGMA_STAR_GAMMA_MODEL_NAMES = (
    "sonnenfeld2024_slacs_sigma_star_gamma",
    "sonnenfeld2024_slacs_sigma_star_gamma_hunit",
)
MODEL_NAMES = BASE_MODEL_NAMES + SIGMA_STAR_GAMMA_MODEL_NAMES
BACKEND_NAME = "numba_sonnenfeld_parent"
ARTIFACT_SCHEMA_VERSION = "sonnenfeld2024_slacs_ppt_diagnostics_v1"
TREND_CATEGORY_NAMES = ("parent", "detectable", "selected")
SUMMARY_STAT_NAMES = ("median", "std", "p10", "p90")
THETA_SAMPLE_SIZE = 23
SIGMA_SAMPLE_SIZE = 7
LOG10_2PI = math.log10(2.0 * math.pi)
DEFAULT_SONNENFELD_NUMBA_DIAGNOSTICS_CHUNK_SIZE = 16


def _observations_from_canonical_dataset(dataset) -> list[ObservationRecord]:
    """Build observation records from canonical Sonnenfeld input."""

    observations: list[ObservationRecord] = []
    n_lens = len(dataset.lenses.lens_id)
    for lens_index in range(n_lens):
        num_sigma = int(dataset.lenses.num_sigma[lens_index])
        sigma_count = max(num_sigma, 0)
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
                sigma_observed=np.asarray(dataset.lenses.sigma_obs[lens_index, :sigma_count], dtype=float),
                sigma_error=np.asarray(dataset.lenses.sigma_err[lens_index, :sigma_count], dtype=float),
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
    """Build the Sonnenfeld predictive context through the inference registry."""

    if runtime_config.data.inference_dataset_path is None:
        raise ValueError("Sonnenfeld predictive diagnostics require data.inference_dataset_path.")

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
    Return NumPy's default linear percentile for an already sorted vector.

    The diagnostics payload exposes small replicated-sample summaries.  Keeping
    the interpolation formula inside the adapter avoids a dependency on
    version-specific `np.percentile` support in Numba while preserving the old
    Python summary semantics.
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
    """
    Return median, sample standard deviation, p10, and p90 for one replicate.

    The implementation is written in scalar loops so the full Sonnenfeld
    diagnostic route can remain inside the compiled kernel.
    """

    n_values = values.shape[0]
    result = np.empty(4, dtype=np.float64)
    for index in range(n_values):
        if math.isnan(values[index]):
            # The pre-Numba Python path used `np.percentile` and `np.std`
            # directly.  Those routines propagate any NaN in the replicated
            # sample, so the compiled helper must not silently sort NaNs to the
            # end and return finite percentiles.
            result[0] = math.nan
            result[1] = math.nan
            result[2] = math.nan
            result[3] = math.nan
            return result
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
    Draw one index from finite positive weights using a pre-generated uniform.

    Python owns random-number generation so chunking remains reproducible under
    `np.random.Generator`.  The compiled kernel only consumes uniforms and
    applies the same fallback as the historical path: if no selected weight is
    available, draw uniformly from the parent population.
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

    The old Python helper built boolean masks for each bin.  The Numba route
    performs the same reduction explicitly to keep large real diagnostics from
    materializing mask arrays inside every posterior draw.
    """

    n_bins = bin_edges.shape[0] - 1
    n_values = x_values.shape[0]
    for bin_index in range(n_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]
        parent_count = 0.0
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

            parent_count += 1.0
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
        parent_means[bin_index] = parent_sum / parent_count if parent_count > 0.0 else math.nan
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


def _sonnenfeld_diagnostics_context_arrays(context) -> dict[str, np.ndarray]:
    """
    Pack Python context fields into arrays accepted by the Numba diagnostics.

    The compiled kernel should not know about the dataclass or test
    `SimpleNamespace` used to hold context.  This helper is the explicit
    adapter boundary: it converts every scalar, parent-population array, and
    interpolation grid consumed by the old loop into stable NumPy inputs.
    """

    return {
        "scalar_context": np.asarray(
            [
                context.n_fixed,
                context.use_sersic_index,
                context.mstar_pivot,
                context.gamma_trunc_low,
                context.gamma_trunc_high,
                context.source_lens_redshift_gap,
                context.mass_radius_kpc,
                context.mass_log_physical_offset,
                context.sigma_proxy_fractional_scatter,
            ],
            dtype=np.float64,
        ),
        "parent_sample_zd": np.ascontiguousarray(context.parent_sample_zd, dtype=np.float64),
        "parent_sample_mstar": np.ascontiguousarray(context.parent_sample_mstar, dtype=np.float64),
        "parent_sample_log_re": np.ascontiguousarray(context.parent_sample_log_re, dtype=np.float64),
        "parent_sample_delta_r": np.ascontiguousarray(context.parent_sample_delta_r, dtype=np.float64),
        "base_normals": np.ascontiguousarray(context.base_normals, dtype=np.float64),
        "z_grid": np.ascontiguousarray(context.z_grid, dtype=np.float64),
        "chi_kpc_grid": np.ascontiguousarray(context.chi_kpc_grid, dtype=np.float64),
        "population_gamma_axis": np.ascontiguousarray(context.population_gamma_axis, dtype=np.float64),
        "population_zd_axis": np.ascontiguousarray(context.population_zd_axis, dtype=np.float64),
        "population_log_re_kpc_axis": np.ascontiguousarray(context.population_log_re_kpc_axis, dtype=np.float64),
        "population_n_axis": np.ascontiguousarray(context.population_n_axis, dtype=np.float64),
        "population_sigma_unit_grid": np.ascontiguousarray(context.population_sigma_unit_grid, dtype=np.float64),
        "cs_theta_e_axis": np.ascontiguousarray(context.cs_theta_e_axis, dtype=np.float64),
        "cs_gamma_axis": np.ascontiguousarray(context.cs_gamma_axis, dtype=np.float64),
        "cs_cross_section_grid": np.ascontiguousarray(context.cs_cross_section_grid, dtype=np.float64),
    }


_SONNENFELD_DIAGNOSTIC_ARRAY_NAMES = (
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


@nb.njit(cache=True, parallel=True)
def _sonnenfeld_parent_diagnostics_numba_chunk(
    theta_chunk: np.ndarray,
    parent_indices: np.ndarray,
    theta_uniforms: np.ndarray,
    sigma_uniforms: np.ndarray,
    scalar_context: np.ndarray,
    parent_sample_zd: np.ndarray,
    parent_sample_mstar: np.ndarray,
    parent_sample_log_re: np.ndarray,
    parent_sample_delta_r: np.ndarray,
    base_normals: np.ndarray,
    z_grid: np.ndarray,
    chi_kpc_grid: np.ndarray,
    population_gamma_axis: np.ndarray,
    population_zd_axis: np.ndarray,
    population_log_re_kpc_axis: np.ndarray,
    population_n_axis: np.ndarray,
    population_sigma_unit_grid: np.ndarray,
    cs_theta_e_axis: np.ndarray,
    cs_gamma_axis: np.ndarray,
    cs_cross_section_grid: np.ndarray,
    mass_bin_edges: np.ndarray,
    sigma_star_bin_edges: np.ndarray,
    log_re_bin_edges: np.ndarray,
    delta_r_bin_edges: np.ndarray,
    is_sigma_star_gamma_model: int,
) -> tuple[np.ndarray, ...]:
    """
    Generate Sonnenfeld parent-population diagnostics for one posterior chunk.

    The loop over posterior draws is parallelized with `nb.prange`.  Each draw
    receives pre-sampled parent indices and uniforms from the Python wrapper, so
    this function owns deterministic physics, selection weighting, weighted
    replication, and trend reductions without calling back into Python.
    """

    n_draws = theta_chunk.shape[0]
    parent_sample_size = parent_indices.shape[1]
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
    parent_bin_counts = np.empty((n_draws, n_mass_bins), dtype=np.float64)
    detectable_weight_sums = np.empty((n_draws, n_mass_bins), dtype=np.float64)
    selected_weight_sums = np.empty((n_draws, n_mass_bins), dtype=np.float64)

    gamma_logre_trend = np.empty((n_draws, len(TREND_CATEGORY_NAMES), n_log_re_bins), dtype=np.float64)
    gamma_logre_parent_counts = np.empty((n_draws, n_log_re_bins), dtype=np.float64)
    gamma_logre_detectable_sums = np.empty((n_draws, n_log_re_bins), dtype=np.float64)
    gamma_logre_selected_sums = np.empty((n_draws, n_log_re_bins), dtype=np.float64)

    gamma_sigma_star_trend = np.empty((n_draws, len(TREND_CATEGORY_NAMES), n_sigma_star_bins), dtype=np.float64)
    gamma_sigma_star_parent_counts = np.empty((n_draws, n_sigma_star_bins), dtype=np.float64)
    gamma_sigma_star_detectable_sums = np.empty((n_draws, n_sigma_star_bins), dtype=np.float64)
    gamma_sigma_star_selected_sums = np.empty((n_draws, n_sigma_star_bins), dtype=np.float64)

    gamma_delta_r_trend = np.empty((n_draws, len(TREND_CATEGORY_NAMES), n_delta_r_bins), dtype=np.float64)
    gamma_delta_r_parent_counts = np.empty((n_draws, n_delta_r_bins), dtype=np.float64)
    gamma_delta_r_detectable_sums = np.empty((n_draws, n_delta_r_bins), dtype=np.float64)
    gamma_delta_r_selected_sums = np.empty((n_draws, n_delta_r_bins), dtype=np.float64)

    n_fixed = scalar_context[0]
    use_sersic_index = int(scalar_context[1])
    mstar_pivot = scalar_context[2]
    gamma_trunc_low = scalar_context[3]
    gamma_trunc_high = scalar_context[4]
    source_lens_redshift_gap = scalar_context[5]
    mass_radius_kpc = scalar_context[6]
    mass_log_physical_offset = scalar_context[7]
    sigma_proxy_fractional_scatter = scalar_context[8]

    for draw_index in nb.prange(n_draws):
        theta = theta_chunk[draw_index]
        sigma5 = max(float(theta[3]), 1.0e-8)
        if is_sigma_star_gamma_model == 1:
            sigma_gamma_value = max(float(theta[6]), 1.0e-8)
            mu_zs = float(theta[7])
            sigma_zs_value = max(float(theta[8]), 1.0e-8)
            theta0 = float(theta[9])
            loga = float(theta[10])
        else:
            sigma_gamma_value = max(float(theta[7]), 1.0e-8)
            mu_zs = float(theta[8])
            sigma_zs_value = max(float(theta[9]), 1.0e-8)
            theta0 = float(theta[10])
            loga = float(theta[11])

        zd_values = np.empty(parent_sample_size, dtype=np.float64)
        zs_values = np.empty(parent_sample_size, dtype=np.float64)
        log_mstar_values = np.empty(parent_sample_size, dtype=np.float64)
        log_mass_values = np.empty(parent_sample_size, dtype=np.float64)
        gamma_values = np.empty(parent_sample_size, dtype=np.float64)
        theta_ein_values = np.empty(parent_sample_size, dtype=np.float64)
        log_re_values = np.empty(parent_sample_size, dtype=np.float64)
        re_kpc_values = np.empty(parent_sample_size, dtype=np.float64)
        n_values = np.empty(parent_sample_size, dtype=np.float64)
        log_sigma_star_values = np.empty(parent_sample_size, dtype=np.float64)
        delta_r_values = np.empty(parent_sample_size, dtype=np.float64)
        sigma_model_values = np.empty(parent_sample_size, dtype=np.float64)
        detectable_weights = np.empty(parent_sample_size, dtype=np.float64)
        selected_weights = np.empty(parent_sample_size, dtype=np.float64)

        total_selected_weight = 0.0
        for local_parent_index in range(parent_sample_size):
            source_parent_index = parent_indices[draw_index, local_parent_index]
            normals = base_normals[source_parent_index]
            zd_value = parent_sample_zd[source_parent_index]
            log_mstar_value = parent_sample_mstar[source_parent_index]
            log_re_value = parent_sample_log_re[source_parent_index]
            delta_r_value = parent_sample_delta_r[source_parent_index]
            n_value = n_fixed
            if use_sersic_index == 1:
                n_value = max(4.0 + 0.4 * normals[7], 0.5)

            mstar_shift = log_mstar_value - mstar_pivot
            log_mass_value = (
                float(theta[0])
                + float(theta[1]) * mstar_shift
                + float(theta[2]) * delta_r_value
                + sigma5 * normals[3]
            )
            if is_sigma_star_gamma_model == 1:
                sigma_star_shift9p0 = log_mstar_value - LOG10_2PI - 2.0 * log_re_value - 9.0
                gamma_mean = float(theta[4]) + float(theta[5]) * sigma_star_shift9p0
            else:
                gamma_mean = float(theta[4]) + float(theta[5]) * mstar_shift + float(theta[6]) * delta_r_value

            gamma_value = gamma_mean + sigma_gamma_value * normals[4]
            if gamma_value < gamma_trunc_low:
                gamma_value = gamma_trunc_low
            elif gamma_value > gamma_trunc_high:
                gamma_value = gamma_trunc_high

            zs_value = mu_zs + sigma_zs_value * normals[5]
            minimum_zs = zd_value + source_lens_redshift_gap + 1.0e-3
            if zs_value < minimum_zs:
                zs_value = minimum_zs

            theta_ein_value = theta_ein_arcsec(
                zd_value,
                zs_value,
                log_mass_value,
                gamma_value,
                z_grid,
                chi_kpc_grid,
                mass_radius_kpc,
                mass_log_physical_offset,
            )
            sigma_unit = interp_sigma_unit_clip(
                gamma_value,
                zd_value,
                log_re_value,
                n_value,
                population_gamma_axis,
                population_zd_axis,
                population_log_re_kpc_axis,
                population_n_axis,
                population_sigma_unit_grid,
                1,
            )
            sigma_model_value = 0.0
            cross_section_weight = 0.0
            selection_weight = 0.0
            if theta_ein_value > 0.0 and sigma_unit > 0.0:
                cross_section_candidate = interp_cross_section_theta_gamma(
                    theta_ein_value,
                    gamma_value,
                    cs_theta_e_axis,
                    cs_gamma_axis,
                    cs_cross_section_grid,
                )
                if math.isfinite(cross_section_candidate) and cross_section_candidate > 0.0:
                    # "Detectable" in the Sonnenfeld Fig. 8-like products
                    # matches the reference no-pfind posterior predictive
                    # route: finite-fibre lensing cross-section only.  The
                    # extra discovery probability belongs to the full
                    # selected-lens curve below.
                    cross_section_weight = cross_section_candidate
                sigma_model_value = sigma_model_from_s2(sigma_unit, log_mass_value)
                sigma_proxy = sigma_model_value * (1.0 + sigma_proxy_fractional_scatter * normals[6])
                theta_est = theta_e_est_from_sigma_proxy(
                    sigma_proxy,
                    zd_value,
                    zs_value,
                    z_grid,
                    chi_kpc_grid,
                )
                selection_weight = cross_section_find_weight(
                    theta_ein_value,
                    gamma_value,
                    theta_est,
                    theta0,
                    loga,
                    cs_theta_e_axis,
                    cs_gamma_axis,
                    cs_cross_section_grid,
                )

            log_mstar_values[local_parent_index] = log_mstar_value
            log_mass_values[local_parent_index] = log_mass_value
            gamma_values[local_parent_index] = gamma_value
            theta_ein_values[local_parent_index] = theta_ein_value
            zd_values[local_parent_index] = zd_value
            zs_values[local_parent_index] = zs_value
            log_re_values[local_parent_index] = log_re_value
            re_kpc_values[local_parent_index] = 10.0**log_re_value
            n_values[local_parent_index] = n_value
            log_sigma_star_values[local_parent_index] = log_mstar_value - LOG10_2PI - 2.0 * log_re_value
            delta_r_values[local_parent_index] = delta_r_value
            sigma_model_values[local_parent_index] = sigma_model_value
            detectable_weights[local_parent_index] = cross_section_weight
            selected_weight_value = selection_weight if math.isfinite(selection_weight) and selection_weight > 0.0 else 0.0
            selected_weights[local_parent_index] = selected_weight_value
            total_selected_weight += selected_weight_value

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
            theta_mass[draw_index, sample_index] = log_mass_values[selected_index]
            theta_re_kpc[draw_index, sample_index] = re_kpc_values[selected_index]
            theta_n[draw_index, sample_index] = n_values[selected_index]
            theta_sample_values[sample_index] = theta_ein_values[selected_index]

        for sample_index in range(SIGMA_SAMPLE_SIZE):
            selected_index = _numba_draw_weighted_index(
                selected_weights,
                total_selected_weight,
                sigma_uniforms[draw_index, sample_index],
            )
            sigma_sigma[draw_index, sample_index] = sigma_model_values[selected_index]
            sigma_theta_ein[draw_index, sample_index] = theta_ein_values[selected_index]
            sigma_gamma[draw_index, sample_index] = gamma_values[selected_index]
            sigma_zd[draw_index, sample_index] = zd_values[selected_index]
            sigma_zs[draw_index, sample_index] = zs_values[selected_index]
            sigma_mass[draw_index, sample_index] = log_mass_values[selected_index]
            sigma_re_kpc[draw_index, sample_index] = re_kpc_values[selected_index]
            sigma_n[draw_index, sample_index] = n_values[selected_index]
            sigma_sample_values[sample_index] = sigma_model_values[selected_index]

        theta_stats[draw_index] = _numba_summary_statistics(theta_sample_values)
        sigma_stats[draw_index] = _numba_summary_statistics(sigma_sample_values)

        scratch_counts = np.empty(n_mass_bins, dtype=np.float64)
        scratch_detectable_sums = np.empty(n_mass_bins, dtype=np.float64)
        scratch_selected_sums = np.empty(n_mass_bins, dtype=np.float64)
        _numba_reduce_population_to_bins(
            log_mstar_values,
            log_mass_values,
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
            scratch_counts,
            scratch_detectable_sums,
            scratch_selected_sums,
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
            scratch_counts,
            scratch_detectable_sums,
            scratch_selected_sums,
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


def _run_sonnenfeld_parent_diagnostics(
    posterior_draws: np.ndarray,
    profile,
    context,
    mass_definition: MassDefinition,
    sigma_table,
    mass_bin_edges: np.ndarray,
    sigma_star_bin_edges: np.ndarray,
    log_re_bin_edges: np.ndarray,
    delta_r_bin_edges: np.ndarray,
    parent_sample_size: int,
    random_seed: int,
    model_name: str = BASE_MODEL_NAMES[0],
    *,
    execution: DiagnosticsExecution | None = None,
) -> dict[str, Any]:
    """
    Execute Sonnenfeld diagnostics through the adapter-owned Numba kernel.

    The wrapper owns orchestration that is intentionally outside Numba:
    validating model-specific theta dimensionality, applying the resolved
    thread cap, chunking posterior draws, generating parent-resampling indices,
    and adapting dense kernel outputs back to the existing payload schema.
    """

    del profile, sigma_table
    if execution is not None:
        # The generic PPT runner records the execution policy, but the model
        # adapter owns Numba kernels.  Apply the cap here so artifact metadata
        # and actual thread usage describe the same computation.
        apply_thread_limits(int(execution.kernel_threads_per_process))

    is_sigma_star_gamma_model = model_name in SIGMA_STAR_GAMMA_MODEL_NAMES
    expected_theta_size = 11 if is_sigma_star_gamma_model else 12
    posterior_draws_array = np.asarray(posterior_draws, dtype=np.float64)
    if posterior_draws_array.ndim != 2 or posterior_draws_array.shape[1] != expected_theta_size:
        received = posterior_draws_array.shape[1] if posterior_draws_array.ndim == 2 else posterior_draws_array.shape
        raise ValueError(
            f"Sonnenfeld predictive diagnostics for '{model_name}' expected "
            f"{expected_theta_size} parameters, received {received}."
        )

    context_arrays = _sonnenfeld_diagnostics_context_arrays(context)
    available_parent = int(context_arrays["parent_sample_mstar"].shape[0])
    if available_parent <= 0:
        raise ValueError("Sonnenfeld predictive diagnostics require at least one parent sample.")
    n_parent = max(1, min(int(parent_sample_size), available_parent))

    n_draws = int(posterior_draws_array.shape[0])
    rng = np.random.default_rng(int(random_seed))
    chunk_outputs: list[tuple[np.ndarray, ...]] = []
    chunk_size = max(1, min(DEFAULT_SONNENFELD_NUMBA_DIAGNOSTICS_CHUNK_SIZE, max(n_draws, 1)))
    chunk_starts = range(0, n_draws, chunk_size) if n_draws > 0 else (0,)

    for chunk_start in chunk_starts:
        chunk_end = min(chunk_start + chunk_size, n_draws)
        current_chunk_size = chunk_end - chunk_start
        parent_indices = np.empty((current_chunk_size, n_parent), dtype=np.int64)
        for draw_offset in range(current_chunk_size):
            parent_indices[draw_offset] = rng.choice(
                available_parent,
                size=n_parent,
                replace=available_parent < n_parent,
            )
        theta_uniforms = rng.random(size=(current_chunk_size, THETA_SAMPLE_SIZE)).astype(np.float64, copy=False)
        sigma_uniforms = rng.random(size=(current_chunk_size, SIGMA_SAMPLE_SIZE)).astype(np.float64, copy=False)
        chunk_outputs.append(
            _sonnenfeld_parent_diagnostics_numba_chunk(
                np.ascontiguousarray(posterior_draws_array[chunk_start:chunk_end], dtype=np.float64),
                np.ascontiguousarray(parent_indices, dtype=np.int64),
                np.ascontiguousarray(theta_uniforms, dtype=np.float64),
                np.ascontiguousarray(sigma_uniforms, dtype=np.float64),
                np.ascontiguousarray(context_arrays["scalar_context"], dtype=np.float64),
                np.ascontiguousarray(context_arrays["parent_sample_zd"], dtype=np.float64),
                np.ascontiguousarray(context_arrays["parent_sample_mstar"], dtype=np.float64),
                np.ascontiguousarray(context_arrays["parent_sample_log_re"], dtype=np.float64),
                np.ascontiguousarray(context_arrays["parent_sample_delta_r"], dtype=np.float64),
                np.ascontiguousarray(context_arrays["base_normals"], dtype=np.float64),
                np.ascontiguousarray(context_arrays["z_grid"], dtype=np.float64),
                np.ascontiguousarray(context_arrays["chi_kpc_grid"], dtype=np.float64),
                np.ascontiguousarray(context_arrays["population_gamma_axis"], dtype=np.float64),
                np.ascontiguousarray(context_arrays["population_zd_axis"], dtype=np.float64),
                np.ascontiguousarray(context_arrays["population_log_re_kpc_axis"], dtype=np.float64),
                np.ascontiguousarray(context_arrays["population_n_axis"], dtype=np.float64),
                np.ascontiguousarray(context_arrays["population_sigma_unit_grid"], dtype=np.float64),
                np.ascontiguousarray(context_arrays["cs_theta_e_axis"], dtype=np.float64),
                np.ascontiguousarray(context_arrays["cs_gamma_axis"], dtype=np.float64),
                np.ascontiguousarray(context_arrays["cs_cross_section_grid"], dtype=np.float64),
                np.ascontiguousarray(mass_bin_edges, dtype=np.float64),
                np.ascontiguousarray(sigma_star_bin_edges, dtype=np.float64),
                np.ascontiguousarray(log_re_bin_edges, dtype=np.float64),
                np.ascontiguousarray(delta_r_bin_edges, dtype=np.float64),
                1 if is_sigma_star_gamma_model else 0,
            )
        )

    arrays = {
        array_name: np.concatenate(
            [chunk[index] for chunk in chunk_outputs],
            axis=0,
        )
        for index, array_name in enumerate(_SONNENFELD_DIAGNOSTIC_ARRAY_NAMES)
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


def _build_sonnenfeld_trend_panel_order(mass_definition: MassDefinition) -> tuple[str, ...]:
    """Return the declared Sonnenfeld trend-panel contract."""

    del mass_definition
    return ("theta_ein", "sigma_ap", "gamma", "gamma_vs_logre_kpc")


def get_predictive_definition(model_name: str) -> PredictiveDefinition:
    """Return a Sonnenfeld predictive definition for one concrete registry name."""

    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported Sonnenfeld predictive model '{model_name}'.")

    def run_model_diagnostics(
        posterior_draws: np.ndarray,
        profile,
        context,
        mass_definition: MassDefinition,
        sigma_table,
        mass_bin_edges: np.ndarray,
        sigma_star_bin_edges: np.ndarray,
        log_re_bin_edges: np.ndarray,
        delta_r_bin_edges: np.ndarray,
        parent_sample_size: int,
        random_seed: int,
        *,
        execution: DiagnosticsExecution | None = None,
    ) -> dict[str, Any]:
        """
        Bind the concrete model name into the shared Sonnenfeld diagnostics.

        The generic PPT runner intentionally passes only scientific arrays and
        context objects to the model hook.  Capturing ``model_name`` here keeps
        the 12D paper-gamma and 11D sigma-star-gamma theta semantics inside the
        model-specific adapter instead of leaking another branch into the
        generic workflow.
        """

        return _run_sonnenfeld_parent_diagnostics(
            posterior_draws=posterior_draws,
            profile=profile,
            context=context,
            mass_definition=mass_definition,
            sigma_table=sigma_table,
            mass_bin_edges=mass_bin_edges,
            sigma_star_bin_edges=sigma_star_bin_edges,
            log_re_bin_edges=log_re_bin_edges,
            delta_r_bin_edges=delta_r_bin_edges,
            parent_sample_size=parent_sample_size,
            random_seed=random_seed,
            model_name=model_name,
            execution=execution,
        )

    return PredictiveDefinition(
        model_name=model_name,
        backend=BACKEND_NAME,
        supported_diagnostics=("posterior_diagnostics", "posterior_predictive", "posterior_trends"),
        required_external_inputs=(),
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        build_context=build_context,
        run_diagnostics=run_model_diagnostics,
        trend_category_names=TREND_CATEGORY_NAMES,
        build_trend_panel_order=_build_sonnenfeld_trend_panel_order,
    )


__all__ = ["MODEL_NAMES", "get_predictive_definition"]
