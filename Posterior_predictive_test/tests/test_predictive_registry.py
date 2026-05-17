"""Tests for model-aware posterior-predictive dispatch.

The current refactor goal is to stop treating CMASS as an implicit global
inside the generic PPT workflow.  These tests pin the smallest dispatch
contract first: the PPT package should expose a predictive registry, CMASS
should register the existing Numba shared-parent diagnostics, and unsupported
models should fail with a model-specific error before any CMASS context builder
is touched.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest


SONNENFELD_TOP_LEVEL_KEYS = (
    "theta_latent",
    "sigma_latent",
    "theta_replicated_stats",
    "sigma_replicated_stats",
    "trend_draws",
    "parent_bin_counts_draws",
    "detectable_weight_sums_draws",
    "selected_weight_sums_draws",
    "gamma_vs_logre_draws",
    "gamma_vs_logre_parent_bin_counts_draws",
    "gamma_vs_logre_detectable_weight_sums_draws",
    "gamma_vs_logre_selected_weight_sums_draws",
    "gamma_vs_sigma_star_draws",
    "gamma_vs_sigma_star_parent_bin_counts_draws",
    "gamma_vs_sigma_star_detectable_weight_sums_draws",
    "gamma_vs_sigma_star_selected_weight_sums_draws",
    "gamma_vs_delta_r_draws",
    "gamma_vs_delta_r_parent_bin_counts_draws",
    "gamma_vs_delta_r_detectable_weight_sums_draws",
    "gamma_vs_delta_r_selected_weight_sums_draws",
)
SONNENFELD_THETA_KEYS = ("theta_ein", "gamma", "zd", "zs", "m5", "re_kpc", "n")
SONNENFELD_SIGMA_KEYS = ("sigma",) + SONNENFELD_THETA_KEYS
SONNENFELD_TREND_QUANTITY_KEYS = ("m5", "gamma", "sigma_ap")
SONNENFELD_TREND_CATEGORY_KEYS = ("parent", "detectable", "selected")
SONNENFELD_STAT_KEYS = ("median", "std", "p10", "p90")


def _fake_sonnenfeld_context() -> SimpleNamespace:
    """
    Build a minimal context with exactly the fields the Sonnenfeld adapter uses.

    The arrays are deliberately tiny but keep all axes non-degenerate except
    the fixed Sersic-index axis.  That lets the tests exercise context packing,
    interpolation calls, weighted sampling, and trend reduction without
    depending on a full canonical HDF5 fixture.
    """

    population_sigma_unit_grid = np.fromfunction(
        lambda gamma_index, zd_index, re_index, n_index: (
            0.75
            + 0.07 * gamma_index
            + 0.03 * zd_index
            + 0.02 * re_index
            + 0.01 * n_index
        ),
        (3, 3, 3, 3),
        dtype=np.float64,
    )
    cs_cross_section_grid = np.fromfunction(
        lambda theta_index, gamma_index: 0.2 + 0.45 * theta_index + 0.08 * gamma_index,
        (5, 3),
        dtype=np.float64,
    )

    return SimpleNamespace(
        parent_sample_zd=np.asarray([0.25, 0.35, 0.45, 0.55], dtype=np.float64),
        parent_sample_mstar=np.asarray([11.0, 11.2, 11.4, 11.6], dtype=np.float64),
        parent_sample_log_re=np.asarray([0.45, 0.50, 0.55, 0.60], dtype=np.float64),
        parent_sample_delta_r=np.asarray([-0.05, 0.00, 0.05, 0.10], dtype=np.float64),
        base_normals=np.asarray(
            [
                [0.15, -0.10, 0.20, 0.30, -0.25, 0.40, 0.10, -0.50],
                [-0.30, 0.20, -0.10, -0.20, 0.35, -0.15, -0.05, 0.75],
                [0.45, -0.35, 0.15, 0.10, -0.45, 0.25, 0.20, 1.00],
                [-0.20, 0.50, -0.25, -0.10, 0.55, -0.35, -0.15, -0.75],
            ],
            dtype=np.float64,
        ),
        n_fixed=4.0,
        use_sersic_index=1,
        mstar_pivot=11.3,
        gamma_trunc_low=1.2,
        gamma_trunc_high=2.8,
        source_lens_redshift_gap=0.1,
        mass_radius_kpc=5.0,
        mass_log_physical_offset=0.0,
        sigma_proxy_fractional_scatter=0.03,
        z_grid=np.asarray([0.0, 0.5, 1.0, 2.0], dtype=np.float64),
        chi_kpc_grid=np.asarray([0.0, 1.0e6, 2.0e6, 4.0e6], dtype=np.float64),
        population_gamma_axis=np.asarray([1.2, 2.0, 2.8], dtype=np.float64),
        population_zd_axis=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        population_log_re_kpc_axis=np.asarray([0.3, 0.6, 0.9], dtype=np.float64),
        population_n_axis=np.asarray([3.5, 4.0, 4.5], dtype=np.float64),
        population_sigma_unit_grid=np.ascontiguousarray(population_sigma_unit_grid, dtype=np.float64),
        cs_theta_e_axis=np.asarray([0.0, 10.0, 100.0, 1000.0, 1.0e6], dtype=np.float64),
        cs_gamma_axis=np.asarray([1.2, 2.0, 2.8], dtype=np.float64),
        cs_cross_section_grid=np.ascontiguousarray(cs_cross_section_grid, dtype=np.float64),
    )


def _fake_sonnenfeld_posterior_draws_by_model() -> dict[str, np.ndarray]:
    """Return tiny posterior matrices for both Sonnenfeld theta conventions."""

    return {
        "sonnenfeld2024_slacs": np.asarray(
            [
                [11.3, 0.1, 0.05, 0.02, 2.0, 0.1, 0.03, 0.08, 1.2, 0.1, 0.5, 0.0],
                [11.4, 0.0, 0.02, 0.03, 2.1, 0.0, 0.02, 0.10, 1.3, 0.2, 0.6, -0.1],
            ],
            dtype=np.float64,
        ),
        "sonnenfeld2024_slacs_sigma_star_gamma": np.asarray(
            [
                [11.3, 0.1, 0.05, 0.02, 2.0, 0.1, 0.08, 1.2, 0.1, 0.5, 0.0],
                [11.4, 0.0, 0.02, 0.03, 2.1, 0.0, 0.10, 1.3, 0.2, 0.6, -0.1],
            ],
            dtype=np.float64,
        ),
    }


def _reference_summary_statistics(values: np.ndarray) -> np.ndarray:
    """Match the historical Python summary behavior, including NaN propagation."""

    if np.isnan(values).any():
        return np.full(4, np.nan, dtype=np.float64)
    return np.asarray(
        [
            np.percentile(values, 50.0),
            np.std(values, ddof=1) if values.size > 1 else np.nan,
            np.percentile(values, 10.0),
            np.percentile(values, 90.0),
        ],
        dtype=np.float64,
    )


def _reference_draw_weighted_index(weights: np.ndarray, unit_value: float) -> int:
    """Draw one weighted index using the same finite-positive fallback rule."""

    finite_weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    total_weight = float(np.sum(finite_weights))
    clipped_unit = min(max(float(unit_value), 0.0), 1.0 - 1.0e-15)
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        return min(int(clipped_unit * weights.size), weights.size - 1)

    target = clipped_unit * total_weight
    running_total = 0.0
    last_positive_index = weights.size - 1
    for index, weight in enumerate(finite_weights):
        if weight > 0.0:
            running_total += float(weight)
            last_positive_index = index
            if running_total >= target:
                return index
    return last_positive_index


def _reference_reduce_population_to_bins(
    x_values: np.ndarray,
    y_values: np.ndarray,
    bin_edges: np.ndarray,
    detectable_weights: np.ndarray,
    selected_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return parent/detectable/selected binned means plus the three count/sum arrays.

    This mirrors the Numba reducer for the fake-context parity tests.  It keeps
    the finite-positive selected-weight filtering explicit so regressions in
    selection-weight handling produce direct numerical mismatches.
    """

    n_bins = bin_edges.size - 1
    parent_means = np.full(n_bins, np.nan, dtype=np.float64)
    detectable_means = np.full(n_bins, np.nan, dtype=np.float64)
    selected_means = np.full(n_bins, np.nan, dtype=np.float64)
    parent_counts = np.zeros(n_bins, dtype=np.float64)
    detectable_sums = np.zeros(n_bins, dtype=np.float64)
    selected_sums = np.zeros(n_bins, dtype=np.float64)
    for bin_index in range(n_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]
        if bin_index == n_bins - 1:
            in_bin = (x_values >= lower) & (x_values <= upper)
        else:
            in_bin = (x_values >= lower) & (x_values < upper)
        finite = in_bin & np.isfinite(y_values)
        parent_counts[bin_index] = float(np.count_nonzero(finite))
        if np.any(finite):
            parent_means[bin_index] = float(np.mean(y_values[finite]))

        detectable = finite & np.isfinite(detectable_weights) & (detectable_weights > 0.0)
        if np.any(detectable):
            local_weights = detectable_weights[detectable]
            detectable_sums[bin_index] = float(np.sum(local_weights))
            detectable_means[bin_index] = float(np.average(y_values[detectable], weights=local_weights))

        selected = finite & np.isfinite(selected_weights) & (selected_weights > 0.0)
        if np.any(selected):
            local_weights = selected_weights[selected]
            selected_sums[bin_index] = float(np.sum(local_weights))
            selected_means[bin_index] = float(np.average(y_values[selected], weights=local_weights))
    return parent_means, detectable_means, selected_means, parent_counts, detectable_sums, selected_sums


def _reference_sonnenfeld_payload(
    *,
    sonnenfeld,
    model_name: str,
    posterior_draws: np.ndarray,
    context: SimpleNamespace,
    mass_bin_edges: np.ndarray,
    sigma_star_bin_edges: np.ndarray,
    log_re_bin_edges: np.ndarray,
    delta_r_bin_edges: np.ndarray,
    parent_sample_size: int,
    random_seed: int,
) -> dict[str, object]:
    """
    Compute a small Python reference payload for the fake Sonnenfeld context.

    The reference intentionally follows wrapper RNG order: parent indices are
    generated per draw, then theta uniforms, then sigma uniforms.  It calls the
    same production scalar kernels for lensing, sigma interpolation, and
    selection, so the comparison isolates adapter-owned control flow, theta
    unpacking, weighted sampling, and payload mapping.
    """

    from cmass_lens_inference.numba_backend.kernels.interpolation import interp_sigma_unit_clip
    from cmass_lens_inference.numba_backend.kernels.lensing import theta_ein_arcsec
    from cmass_lens_inference.numba_backend.kernels.selection import theta_e_est_from_sigma_proxy
    from cmass_lens_inference.numba_backend.kernels.selection_likelihood import (
        cross_section_find_weight,
        sigma_model_from_s2,
    )

    available_parent = int(context.parent_sample_mstar.shape[0])
    n_parent = max(1, min(int(parent_sample_size), available_parent))
    n_draws = posterior_draws.shape[0]
    rng = np.random.default_rng(int(random_seed))
    parent_indices = np.empty((n_draws, n_parent), dtype=np.int64)
    for draw_index in range(n_draws):
        parent_indices[draw_index] = rng.choice(
            available_parent,
            size=n_parent,
            replace=available_parent < n_parent,
        )
    theta_uniforms = rng.random(size=(n_draws, sonnenfeld.THETA_SAMPLE_SIZE))
    sigma_uniforms = rng.random(size=(n_draws, sonnenfeld.SIGMA_SAMPLE_SIZE))

    theta_latent = {key: np.zeros((n_draws, sonnenfeld.THETA_SAMPLE_SIZE), dtype=np.float64) for key in SONNENFELD_THETA_KEYS}
    sigma_latent = {key: np.zeros((n_draws, sonnenfeld.SIGMA_SAMPLE_SIZE), dtype=np.float64) for key in SONNENFELD_SIGMA_KEYS}
    theta_stats = {key: np.zeros(n_draws, dtype=np.float64) for key in SONNENFELD_STAT_KEYS}
    sigma_stats = {key: np.zeros(n_draws, dtype=np.float64) for key in SONNENFELD_STAT_KEYS}
    trend_draws = {
        quantity: {category: np.zeros((n_draws, mass_bin_edges.size - 1), dtype=np.float64) for category in SONNENFELD_TREND_CATEGORY_KEYS}
        for quantity in SONNENFELD_TREND_QUANTITY_KEYS
    }
    gamma_vs_logre_draws = {
        category: np.zeros((n_draws, log_re_bin_edges.size - 1), dtype=np.float64)
        for category in SONNENFELD_TREND_CATEGORY_KEYS
    }
    gamma_vs_sigma_star_draws = {
        category: np.zeros((n_draws, sigma_star_bin_edges.size - 1), dtype=np.float64)
        for category in SONNENFELD_TREND_CATEGORY_KEYS
    }
    gamma_vs_delta_r_draws = {
        category: np.zeros((n_draws, delta_r_bin_edges.size - 1), dtype=np.float64)
        for category in SONNENFELD_TREND_CATEGORY_KEYS
    }
    parent_counts = np.zeros((n_draws, mass_bin_edges.size - 1), dtype=np.float64)
    detectable_sums = np.zeros_like(parent_counts)
    selected_sums = np.zeros_like(parent_counts)
    gamma_logre_counts = np.zeros((n_draws, log_re_bin_edges.size - 1), dtype=np.float64)
    gamma_logre_detectable = np.zeros_like(gamma_logre_counts)
    gamma_logre_selected = np.zeros_like(gamma_logre_counts)
    gamma_sigma_star_counts = np.zeros((n_draws, sigma_star_bin_edges.size - 1), dtype=np.float64)
    gamma_sigma_star_detectable = np.zeros_like(gamma_sigma_star_counts)
    gamma_sigma_star_selected = np.zeros_like(gamma_sigma_star_counts)
    gamma_delta_r_counts = np.zeros((n_draws, delta_r_bin_edges.size - 1), dtype=np.float64)
    gamma_delta_r_detectable = np.zeros_like(gamma_delta_r_counts)
    gamma_delta_r_selected = np.zeros_like(gamma_delta_r_counts)
    is_sigma_star_gamma = model_name in {
        "sonnenfeld2024_slacs_sigma_star_gamma",
        "sonnenfeld2024_slacs_sigma_star_gamma_hunit",
    }

    for draw_index, theta in enumerate(posterior_draws):
        selected_parent = parent_indices[draw_index]
        normals = context.base_normals[selected_parent]
        zd = context.parent_sample_zd[selected_parent]
        log_mstar = context.parent_sample_mstar[selected_parent]
        log_re = context.parent_sample_log_re[selected_parent]
        delta_r = context.parent_sample_delta_r[selected_parent]
        n_values = np.full(n_parent, float(context.n_fixed), dtype=np.float64)
        if int(context.use_sersic_index) == 1:
            n_values = np.maximum(4.0 + 0.4 * normals[:, 7], 0.5)

        mstar_shift = log_mstar - float(context.mstar_pivot)
        sigma5 = max(float(theta[3]), 1.0e-8)
        log_mass = float(theta[0]) + float(theta[1]) * mstar_shift + float(theta[2]) * delta_r + sigma5 * normals[:, 3]
        if is_sigma_star_gamma:
            sigma_gamma = max(float(theta[6]), 1.0e-8)
            mu_zs = float(theta[7])
            sigma_zs = max(float(theta[8]), 1.0e-8)
            theta0 = float(theta[9])
            loga = float(theta[10])
            sigma_star_shift9p0 = log_mstar - sonnenfeld.LOG10_2PI - 2.0 * log_re - 9.0
            gamma_mean = float(theta[4]) + float(theta[5]) * sigma_star_shift9p0
        else:
            sigma_gamma = max(float(theta[7]), 1.0e-8)
            mu_zs = float(theta[8])
            sigma_zs = max(float(theta[9]), 1.0e-8)
            theta0 = float(theta[10])
            loga = float(theta[11])
            gamma_mean = float(theta[4]) + float(theta[5]) * mstar_shift + float(theta[6]) * delta_r
        gamma = np.clip(gamma_mean + sigma_gamma * normals[:, 4], context.gamma_trunc_low, context.gamma_trunc_high)
        zs = np.maximum(mu_zs + sigma_zs * normals[:, 5], zd + float(context.source_lens_redshift_gap) + 1.0e-3)

        theta_ein = np.zeros(n_parent, dtype=np.float64)
        sigma_model = np.zeros(n_parent, dtype=np.float64)
        selected_weights = np.zeros(n_parent, dtype=np.float64)
        for parent_index in range(n_parent):
            theta_ein[parent_index] = theta_ein_arcsec(
                zd[parent_index],
                zs[parent_index],
                log_mass[parent_index],
                gamma[parent_index],
                context.z_grid,
                context.chi_kpc_grid,
                float(context.mass_radius_kpc),
                float(context.mass_log_physical_offset),
            )
            sigma_unit = interp_sigma_unit_clip(
                gamma[parent_index],
                zd[parent_index],
                log_re[parent_index],
                n_values[parent_index],
                context.population_gamma_axis,
                context.population_zd_axis,
                context.population_log_re_kpc_axis,
                context.population_n_axis,
                context.population_sigma_unit_grid,
                1,
            )
            if theta_ein[parent_index] <= 0.0 or sigma_unit <= 0.0:
                continue
            sigma_model[parent_index] = sigma_model_from_s2(sigma_unit, log_mass[parent_index])
            sigma_proxy = sigma_model[parent_index] * (
                1.0 + float(context.sigma_proxy_fractional_scatter) * normals[parent_index, 6]
            )
            theta_est = theta_e_est_from_sigma_proxy(
                sigma_proxy,
                zd[parent_index],
                zs[parent_index],
                context.z_grid,
                context.chi_kpc_grid,
            )
            raw_weight = cross_section_find_weight(
                theta_ein[parent_index],
                gamma[parent_index],
                theta_est,
                theta0,
                loga,
                context.cs_theta_e_axis,
                context.cs_gamma_axis,
                context.cs_cross_section_grid,
            )
            selected_weights[parent_index] = raw_weight if np.isfinite(raw_weight) and raw_weight > 0.0 else 0.0

        theta_indices = np.asarray(
            [_reference_draw_weighted_index(selected_weights, unit_value) for unit_value in theta_uniforms[draw_index]],
            dtype=np.int64,
        )
        sigma_indices = np.asarray(
            [_reference_draw_weighted_index(selected_weights, unit_value) for unit_value in sigma_uniforms[draw_index]],
            dtype=np.int64,
        )
        source_arrays = {
            "theta_ein": theta_ein,
            "gamma": gamma,
            "zd": zd,
            "zs": zs,
            "m5": log_mass,
            "re_kpc": np.power(10.0, log_re),
            "n": n_values,
        }
        for key, values in source_arrays.items():
            theta_latent[key][draw_index] = values[theta_indices]
            sigma_latent[key][draw_index] = values[sigma_indices]
        sigma_latent["sigma"][draw_index] = sigma_model[sigma_indices]

        theta_summary = _reference_summary_statistics(theta_ein[theta_indices])
        sigma_summary = _reference_summary_statistics(sigma_model[sigma_indices])
        for stat_index, stat_name in enumerate(SONNENFELD_STAT_KEYS):
            theta_stats[stat_name][draw_index] = theta_summary[stat_index]
            sigma_stats[stat_name][draw_index] = sigma_summary[stat_index]

        detectable_weights = np.where(theta_ein > 0.0, 1.0, 0.0)
        for quantity_name, y_values in (("m5", log_mass), ("gamma", gamma), ("sigma_ap", sigma_model)):
            parent, detectable, selected, counts, det_sums, sel_sums = _reference_reduce_population_to_bins(
                log_mstar,
                y_values,
                mass_bin_edges,
                detectable_weights,
                selected_weights,
            )
            trend_draws[quantity_name]["parent"][draw_index] = parent
            trend_draws[quantity_name]["detectable"][draw_index] = detectable
            trend_draws[quantity_name]["selected"][draw_index] = selected
            if quantity_name == "m5":
                parent_counts[draw_index] = counts
                detectable_sums[draw_index] = det_sums
                selected_sums[draw_index] = sel_sums

        parent, detectable, selected, counts, det_sums, sel_sums = _reference_reduce_population_to_bins(
            log_re,
            gamma,
            log_re_bin_edges,
            detectable_weights,
            selected_weights,
        )
        gamma_vs_logre_draws["parent"][draw_index] = parent
        gamma_vs_logre_draws["detectable"][draw_index] = detectable
        gamma_vs_logre_draws["selected"][draw_index] = selected
        gamma_logre_counts[draw_index] = counts
        gamma_logre_detectable[draw_index] = det_sums
        gamma_logre_selected[draw_index] = sel_sums

        log_sigma_star = log_mstar - sonnenfeld.LOG10_2PI - 2.0 * log_re
        parent, detectable, selected, counts, det_sums, sel_sums = _reference_reduce_population_to_bins(
            log_sigma_star,
            gamma,
            sigma_star_bin_edges,
            detectable_weights,
            selected_weights,
        )
        gamma_vs_sigma_star_draws["parent"][draw_index] = parent
        gamma_vs_sigma_star_draws["detectable"][draw_index] = detectable
        gamma_vs_sigma_star_draws["selected"][draw_index] = selected
        gamma_sigma_star_counts[draw_index] = counts
        gamma_sigma_star_detectable[draw_index] = det_sums
        gamma_sigma_star_selected[draw_index] = sel_sums

        parent, detectable, selected, counts, det_sums, sel_sums = _reference_reduce_population_to_bins(
            delta_r,
            gamma,
            delta_r_bin_edges,
            detectable_weights,
            selected_weights,
        )
        gamma_vs_delta_r_draws["parent"][draw_index] = parent
        gamma_vs_delta_r_draws["detectable"][draw_index] = detectable
        gamma_vs_delta_r_draws["selected"][draw_index] = selected
        gamma_delta_r_counts[draw_index] = counts
        gamma_delta_r_detectable[draw_index] = det_sums
        gamma_delta_r_selected[draw_index] = sel_sums

    return {
        "theta_latent": theta_latent,
        "sigma_latent": sigma_latent,
        "theta_replicated_stats": theta_stats,
        "sigma_replicated_stats": sigma_stats,
        "trend_draws": trend_draws,
        "parent_bin_counts_draws": parent_counts,
        "detectable_weight_sums_draws": detectable_sums,
        "selected_weight_sums_draws": selected_sums,
        "gamma_vs_logre_draws": gamma_vs_logre_draws,
        "gamma_vs_logre_parent_bin_counts_draws": gamma_logre_counts,
        "gamma_vs_logre_detectable_weight_sums_draws": gamma_logre_detectable,
        "gamma_vs_logre_selected_weight_sums_draws": gamma_logre_selected,
        "gamma_vs_sigma_star_draws": gamma_vs_sigma_star_draws,
        "gamma_vs_sigma_star_parent_bin_counts_draws": gamma_sigma_star_counts,
        "gamma_vs_sigma_star_detectable_weight_sums_draws": gamma_sigma_star_detectable,
        "gamma_vs_sigma_star_selected_weight_sums_draws": gamma_sigma_star_selected,
        "gamma_vs_delta_r_draws": gamma_vs_delta_r_draws,
        "gamma_vs_delta_r_parent_bin_counts_draws": gamma_delta_r_counts,
        "gamma_vs_delta_r_detectable_weight_sums_draws": gamma_delta_r_detectable,
        "gamma_vs_delta_r_selected_weight_sums_draws": gamma_delta_r_selected,
    }


def _assert_sonnenfeld_payload_schema(result: dict[str, object], *, n_draws: int, sonnenfeld) -> None:
    """Pin the public Sonnenfeld diagnostics schema, including dict key order."""

    assert tuple(result) == SONNENFELD_TOP_LEVEL_KEYS
    assert tuple(result["theta_latent"]) == SONNENFELD_THETA_KEYS
    assert tuple(result["sigma_latent"]) == SONNENFELD_SIGMA_KEYS
    assert tuple(result["theta_replicated_stats"]) == SONNENFELD_STAT_KEYS
    assert tuple(result["sigma_replicated_stats"]) == SONNENFELD_STAT_KEYS
    assert tuple(result["trend_draws"]) == SONNENFELD_TREND_QUANTITY_KEYS
    for quantity_payload in result["trend_draws"].values():
        assert tuple(quantity_payload) == SONNENFELD_TREND_CATEGORY_KEYS
    for gamma_payload_name in (
        "gamma_vs_logre_draws",
        "gamma_vs_sigma_star_draws",
        "gamma_vs_delta_r_draws",
    ):
        assert tuple(result[gamma_payload_name]) == SONNENFELD_TREND_CATEGORY_KEYS

    for array in result["theta_latent"].values():
        assert array.shape == (n_draws, sonnenfeld.THETA_SAMPLE_SIZE)
    for array in result["sigma_latent"].values():
        assert array.shape == (n_draws, sonnenfeld.SIGMA_SAMPLE_SIZE)
    for array in result["theta_replicated_stats"].values():
        assert array.shape == (n_draws,)
    for array in result["sigma_replicated_stats"].values():
        assert array.shape == (n_draws,)

    for quantity_payload in result["trend_draws"].values():
        for array in quantity_payload.values():
            assert array.shape == (n_draws, 2)
    for count_name in (
        "parent_bin_counts_draws",
        "detectable_weight_sums_draws",
        "selected_weight_sums_draws",
        "gamma_vs_logre_parent_bin_counts_draws",
        "gamma_vs_logre_detectable_weight_sums_draws",
        "gamma_vs_logre_selected_weight_sums_draws",
        "gamma_vs_sigma_star_parent_bin_counts_draws",
        "gamma_vs_sigma_star_detectable_weight_sums_draws",
        "gamma_vs_sigma_star_selected_weight_sums_draws",
        "gamma_vs_delta_r_parent_bin_counts_draws",
        "gamma_vs_delta_r_detectable_weight_sums_draws",
        "gamma_vs_delta_r_selected_weight_sums_draws",
    ):
        assert result[count_name].shape == (n_draws, 2)
    for gamma_payload_name in (
        "gamma_vs_logre_draws",
        "gamma_vs_sigma_star_draws",
        "gamma_vs_delta_r_draws",
    ):
        for array in result[gamma_payload_name].values():
            assert array.shape == (n_draws, 2)


def test_cmass_predictive_registry_entry_declares_existing_diagnostics() -> None:
    """CMASS should register the current Numba diagnostics through a thin contract."""

    from lensing_posterior_predictive.registry import get_predictive_definition

    definition = get_predictive_definition("cmass")

    assert definition.model_name == "cmass"
    assert definition.backend == "numba_shared_parent"
    assert definition.required_external_inputs == ("sigma_table",)
    assert "posterior_diagnostics" in definition.supported_diagnostics
    assert callable(definition.build_context)
    assert definition.trend_category_names == ("parent", "detectable", "selected")
    assert definition.build_trend_panel_order(type("Mass", (), {"label": "m10"})()) == (
        "m10",
        "gamma",
        "sigma_ap",
        "gamma_vs_sigma_star",
        "gamma_vs_logre_kpc",
    )


def test_unsupported_model_predictive_dispatch_fails_before_cmass_fallback() -> None:
    """Unsupported models should fail explicitly instead of silently using CMASS logic."""

    from lensing_posterior_predictive.registry import (
        UnsupportedPredictiveModelError,
        get_predictive_definition,
    )

    with pytest.raises(UnsupportedPredictiveModelError, match="toy_hierarchical"):
        get_predictive_definition("toy_hierarchical")


@pytest.mark.parametrize(
    "model_name",
    [
        "sonnenfeld2024_slacs",
        "sonnenfeld2024_slacs_hunit",
        "sonnenfeld2024_slacs_sigma_star_gamma",
        "sonnenfeld2024_slacs_sigma_star_gamma_hunit",
    ],
)
def test_sonnenfeld_predictive_registry_entry_declares_independent_schema(model_name: str) -> None:
    """Sonnenfeld models should expose a non-CMASS predictive contract."""

    from lensing_posterior_predictive.registry import get_predictive_definition

    definition = get_predictive_definition(model_name)

    assert definition.model_name == model_name
    assert definition.backend == "numba_sonnenfeld_parent"
    assert definition.required_external_inputs == ()
    assert definition.artifact_schema_version == "sonnenfeld2024_slacs_ppt_diagnostics_v1"
    assert "posterior_diagnostics" in definition.supported_diagnostics
    assert definition.build_trend_panel_order(type("Mass", (), {"label": "m5"})()) == (
        "theta_ein",
        "sigma_ap",
        "gamma",
        "gamma_vs_logre_kpc",
    )


def test_predictive_definition_diagnostics_hook_accepts_execution_context() -> None:
    """Every model adapter should receive resolved PPC diagnostics execution metadata."""

    from lensing_posterior_predictive.registry import get_predictive_definition

    for model_name in (
        "cmass",
        "sonnenfeld2024_slacs",
        "sonnenfeld2024_slacs_hunit",
        "sonnenfeld2024_slacs_sigma_star_gamma",
        "sonnenfeld2024_slacs_sigma_star_gamma_hunit",
    ):
        definition = get_predictive_definition(model_name)
        signature = inspect.signature(definition.run_diagnostics)
        assert "execution" in signature.parameters
        param = signature.parameters["execution"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None
        assert definition.backend.startswith("numba")


def test_sonnenfeld_diagnostics_use_numba_kernel_and_preserve_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Sonnenfeld diagnostics should run through the adapter-owned Numba hot path.

    The fake context deliberately contains only the fields consumed by the
    diagnostics adapter.  That keeps the test focused on the model boundary:
    parent resampling, compiled-kernel execution, thread-limit handoff, and the
    public payload shapes expected by downstream artifact writers.
    """

    from lensing_posterior_predictive.adapters import sonnenfeld
    from lensing_posterior_predictive.interfaces import DiagnosticsExecution

    thread_limit_calls: list[int] = []
    monkeypatch.setattr(
        sonnenfeld,
        "apply_thread_limits",
        lambda thread_count: thread_limit_calls.append(int(thread_count)),
    )
    kernel_calls: list[tuple[int, int]] = []
    original_kernel = sonnenfeld._sonnenfeld_parent_diagnostics_numba_chunk

    def spy_kernel(*args, **kwargs):
        theta_chunk = args[0]
        parent_indices = args[1]
        kernel_calls.append((int(theta_chunk.shape[0]), int(parent_indices.shape[1])))
        return original_kernel(*args, **kwargs)

    monkeypatch.setattr(sonnenfeld, "_sonnenfeld_parent_diagnostics_numba_chunk", spy_kernel)

    context = _fake_sonnenfeld_context()
    posterior_draws_by_model = _fake_sonnenfeld_posterior_draws_by_model()
    execution = DiagnosticsExecution(
        strategy="kernel_only",
        cpu_count=2,
        reserve_cores=0,
        compute_budget=2,
        requested_worker_processes=2,
        worker_processes=0,
        kernel_threads_per_process=2,
    )
    mass_bin_edges = np.asarray([10.8, 11.3, 11.8], dtype=np.float64)
    sigma_star_bin_edges = np.asarray([9.6, 10.0, 10.4], dtype=np.float64)
    log_re_bin_edges = np.asarray([0.3, 0.5, 0.7], dtype=np.float64)
    delta_r_bin_edges = np.asarray([-0.1, 0.05, 0.2], dtype=np.float64)

    for model_name, posterior_draws in posterior_draws_by_model.items():
        result = sonnenfeld._run_sonnenfeld_parent_diagnostics(
            posterior_draws=posterior_draws,
            profile=None,
            context=context,
            mass_definition=SimpleNamespace(label="m5"),
            sigma_table=None,
            mass_bin_edges=mass_bin_edges,
            sigma_star_bin_edges=sigma_star_bin_edges,
            log_re_bin_edges=log_re_bin_edges,
            delta_r_bin_edges=delta_r_bin_edges,
            parent_sample_size=3,
            random_seed=17,
            model_name=model_name,
            execution=execution,
        )
        reference = _reference_sonnenfeld_payload(
            sonnenfeld=sonnenfeld,
            model_name=model_name,
            posterior_draws=posterior_draws,
            context=context,
            mass_bin_edges=mass_bin_edges,
            sigma_star_bin_edges=sigma_star_bin_edges,
            log_re_bin_edges=log_re_bin_edges,
            delta_r_bin_edges=delta_r_bin_edges,
            parent_sample_size=3,
            random_seed=17,
        )

        _assert_sonnenfeld_payload_schema(result, n_draws=2, sonnenfeld=sonnenfeld)
        for latent_key in SONNENFELD_THETA_KEYS:
            np.testing.assert_allclose(result["theta_latent"][latent_key], reference["theta_latent"][latent_key])
        for latent_key in SONNENFELD_SIGMA_KEYS:
            np.testing.assert_allclose(result["sigma_latent"][latent_key], reference["sigma_latent"][latent_key])
        for stat_name in SONNENFELD_STAT_KEYS:
            np.testing.assert_allclose(result["theta_replicated_stats"][stat_name], reference["theta_replicated_stats"][stat_name])
            np.testing.assert_allclose(result["sigma_replicated_stats"][stat_name], reference["sigma_replicated_stats"][stat_name])
        for quantity_name in SONNENFELD_TREND_QUANTITY_KEYS:
            for category_name in SONNENFELD_TREND_CATEGORY_KEYS:
                np.testing.assert_allclose(
                    result["trend_draws"][quantity_name][category_name],
                    reference["trend_draws"][quantity_name][category_name],
                )
        for gamma_payload_name in (
            "gamma_vs_logre_draws",
            "gamma_vs_sigma_star_draws",
            "gamma_vs_delta_r_draws",
        ):
            for category_name in SONNENFELD_TREND_CATEGORY_KEYS:
                np.testing.assert_allclose(
                    result[gamma_payload_name][category_name],
                    reference[gamma_payload_name][category_name],
                )
        for array_name in (
            "parent_bin_counts_draws",
            "detectable_weight_sums_draws",
            "selected_weight_sums_draws",
            "gamma_vs_logre_parent_bin_counts_draws",
            "gamma_vs_logre_detectable_weight_sums_draws",
            "gamma_vs_logre_selected_weight_sums_draws",
            "gamma_vs_sigma_star_parent_bin_counts_draws",
            "gamma_vs_sigma_star_detectable_weight_sums_draws",
            "gamma_vs_sigma_star_selected_weight_sums_draws",
            "gamma_vs_delta_r_parent_bin_counts_draws",
            "gamma_vs_delta_r_detectable_weight_sums_draws",
            "gamma_vs_delta_r_selected_weight_sums_draws",
        ):
            np.testing.assert_allclose(result[array_name], reference[array_name])

    source = inspect.getsource(sonnenfeld._run_sonnenfeld_parent_diagnostics)
    assert "for draw_index, theta in enumerate" not in source
    assert thread_limit_calls == [2, 2]
    assert kernel_calls == [(2, 3), (2, 3)]


def test_sonnenfeld_numba_summary_statistics_propagates_nan_values() -> None:
    """The Numba summary helper should preserve old NumPy NaN propagation."""

    from lensing_posterior_predictive.adapters import sonnenfeld

    result = sonnenfeld._numba_summary_statistics(np.asarray([1.0, np.nan, 3.0], dtype=np.float64))

    np.testing.assert_allclose(result, np.full(4, np.nan, dtype=np.float64))


def test_ppc_context_builder_uses_predictive_registry_instead_of_model_name_branch() -> None:
    """The generic context builder should dispatch through the predictive registry."""

    from lensing_posterior_predictive import predictive

    source = inspect.getsource(predictive._build_ppc_context)

    assert "get_predictive_definition" in source
    assert 'runtime_config.model.name != "cmass"' not in source


def test_generic_predictive_workflow_does_not_import_cmass_posterior_helpers() -> None:
    """CMASS posterior helpers should live behind the model-specific boundary."""

    from lensing_posterior_predictive import predictive

    source = inspect.getsource(predictive)

    assert "from cmass_lens_inference.models.cmass.posterior import" not in source
    assert "cmass_gamma_population_mean" not in source
    assert "unpack_cmass_theta" not in source


def test_diagnostics_runner_is_model_definition_owned() -> None:
    """The joint diagnostics workflow should call the active predictive definition."""

    from lensing_posterior_predictive import predictive

    source = inspect.getsource(predictive.run_posterior_diagnostics)

    assert "predictive_definition.run_diagnostics" in source
    assert "_run_shared_parent_diagnostics_numba(" not in source


def test_cmass_context_builder_reuses_inference_model_registry() -> None:
    """CMASS predictive context construction should reuse the inference registry."""

    from lensing_posterior_predictive.adapters import cmass

    source = inspect.getsource(cmass.build_context)

    assert "get_model_definition" in source
    assert ".build_compiled_model(" in source
    assert "load_cmass_canonical_dataset" not in source
    assert "build_cmass_context_from_canonical_dataset" not in source


def test_legacy_raw_config_parser_is_cmass_only() -> None:
    """Raw pre-registry snapshots should not become a non-CMASS fallback path."""

    from pathlib import Path

    from lensing_posterior_predictive.legacy import load_legacy_ppc_runtime_config

    with pytest.raises(ValueError, match="only supports model.name='cmass'"):
        load_legacy_ppc_runtime_config(
            Path("config_snapshot.yaml"),
            {"model": {"name": "sonnenfeld2024_slacs"}},
        )


def test_legacy_parser_is_quarantined_outside_generic_predictive_module() -> None:
    """Generic predictive workflow should not define the legacy CMASS parser."""

    from lensing_posterior_predictive import predictive

    source = inspect.getsource(predictive)

    assert "def _legacy_ppc_parameter_order" not in source
    assert "def _load_legacy_ppc_runtime_config" not in source
