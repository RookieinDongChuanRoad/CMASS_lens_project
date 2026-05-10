"""Sonnenfeld/SLACS posterior-predictive registry entries.

The current hook is intentionally narrower than the mature CMASS adapter: it
uses the Sonnenfeld inference registry and canonical context, then materializes
the common diagnostic arrays expected by the existing PPT artifact writer.  It
does not reuse CMASS posterior helpers or CMASS gamma-mode logic.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from cmass_lens_inference.canonical_context import lens_gamma_axis
from cmass_lens_inference.canonical_dataset import load_canonical_inference_dataset
from cmass_lens_inference.mass_definition import MassDefinition
from cmass_lens_inference.model_registry import get_model_definition
from cmass_lens_inference.numba_backend.kernels.interpolation import interp_sigma_unit_clip
from cmass_lens_inference.numba_backend.kernels.lensing import theta_ein_arcsec
from cmass_lens_inference.numba_backend.kernels.selection import theta_e_est_from_sigma_proxy
from cmass_lens_inference.numba_backend.kernels.selection_likelihood import (
    cross_section_find_weight,
    sigma_model_from_s2,
)
from cmass_lens_inference.types import ObservationRecord, RuntimeConfig

from ..interfaces import PPCContextBundle, PredictiveDefinition


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


def _summary_statistics(values: np.ndarray) -> np.ndarray:
    """Return median, sample std, p10, and p90 for one replicated sample."""

    if values.size == 0:
        return np.full(4, np.nan, dtype=float)
    std = float(np.std(values, ddof=1)) if values.size > 1 else math.nan
    return np.asarray(
        [
            np.percentile(values, 50.0),
            std,
            np.percentile(values, 10.0),
            np.percentile(values, 90.0),
        ],
        dtype=float,
    )


def _bin_weighted_means(
    x_values: np.ndarray,
    y_values: np.ndarray,
    bin_edges: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return binned means, unweighted counts, and weight sums."""

    n_bins = bin_edges.size - 1
    means = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=float)
    weight_sums = np.zeros(n_bins, dtype=float)
    for bin_index in range(n_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]
        if bin_index == n_bins - 1:
            mask = (x_values >= lower) & (x_values <= upper)
        else:
            mask = (x_values >= lower) & (x_values < upper)
        finite_mask = mask & np.isfinite(y_values)
        counts[bin_index] = float(np.count_nonzero(finite_mask))
        if weights is None:
            if np.any(finite_mask):
                means[bin_index] = float(np.mean(y_values[finite_mask]))
                weight_sums[bin_index] = counts[bin_index]
            continue
        local_weights = np.asarray(weights[finite_mask], dtype=float)
        positive = np.isfinite(local_weights) & (local_weights > 0.0)
        if np.any(positive):
            means[bin_index] = float(np.average(y_values[finite_mask][positive], weights=local_weights[positive]))
            weight_sums[bin_index] = float(np.sum(local_weights[positive]))
    return means, counts, weight_sums


def _draw_weighted_indices(
    rng: np.random.Generator,
    weights: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    """Draw replicated indices using selection weights with a finite fallback."""

    finite_weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    total_weight = float(np.sum(finite_weights))
    n_values = weights.size
    if total_weight <= 0.0:
        return rng.integers(0, n_values, size=sample_size)
    return rng.choice(n_values, size=sample_size, replace=True, p=finite_weights / total_weight)


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
) -> dict[str, Any]:
    """Materialize a minimal Sonnenfeld parent-population diagnostics payload."""

    del profile, sigma_table
    is_sigma_star_gamma_model = model_name in SIGMA_STAR_GAMMA_MODEL_NAMES
    rng = np.random.default_rng(int(random_seed))
    n_draws = int(posterior_draws.shape[0])
    available_parent = int(context.parent_sample_mstar.shape[0])
    n_parent = max(1, min(int(parent_sample_size), available_parent))
    mass_label = mass_definition.label

    theta_latent_keys = ("theta_ein", "gamma", "zd", "zs", mass_label, "re_kpc", "n")
    sigma_latent_keys = ("sigma",) + theta_latent_keys
    theta_latent = {key: np.zeros((n_draws, THETA_SAMPLE_SIZE), dtype=float) for key in theta_latent_keys}
    sigma_latent = {key: np.zeros((n_draws, SIGMA_SAMPLE_SIZE), dtype=float) for key in sigma_latent_keys}
    theta_stats = {name: np.zeros(n_draws, dtype=float) for name in SUMMARY_STAT_NAMES}
    sigma_stats = {name: np.zeros(n_draws, dtype=float) for name in SUMMARY_STAT_NAMES}

    def empty_trend() -> dict[str, np.ndarray]:
        return {category: np.full((n_draws, mass_bin_edges.size - 1), np.nan, dtype=float) for category in TREND_CATEGORY_NAMES}

    trend_draws = {
        mass_label: empty_trend(),
        "gamma": empty_trend(),
        "sigma_ap": empty_trend(),
    }
    parent_counts = np.zeros((n_draws, mass_bin_edges.size - 1), dtype=float)
    detectable_weight_sums = np.zeros_like(parent_counts)
    selected_weight_sums = np.zeros_like(parent_counts)

    def empty_gamma_trend(edges: np.ndarray) -> dict[str, np.ndarray]:
        return {category: np.full((n_draws, edges.size - 1), np.nan, dtype=float) for category in TREND_CATEGORY_NAMES}

    gamma_vs_logre = empty_gamma_trend(log_re_bin_edges)
    gamma_vs_sigma_star = empty_gamma_trend(sigma_star_bin_edges)
    gamma_vs_delta_r = empty_gamma_trend(delta_r_bin_edges)
    gamma_logre_counts = np.zeros((n_draws, log_re_bin_edges.size - 1), dtype=float)
    gamma_logre_detectable = np.zeros_like(gamma_logre_counts)
    gamma_logre_selected = np.zeros_like(gamma_logre_counts)
    gamma_sigma_counts = np.zeros((n_draws, sigma_star_bin_edges.size - 1), dtype=float)
    gamma_sigma_detectable = np.zeros_like(gamma_sigma_counts)
    gamma_sigma_selected = np.zeros_like(gamma_sigma_counts)
    gamma_delta_counts = np.zeros((n_draws, delta_r_bin_edges.size - 1), dtype=float)
    gamma_delta_detectable = np.zeros_like(gamma_delta_counts)
    gamma_delta_selected = np.zeros_like(gamma_delta_counts)

    for draw_index, theta in enumerate(np.asarray(posterior_draws, dtype=float)):
        expected_theta_size = 11 if is_sigma_star_gamma_model else 12
        if theta.shape[0] != expected_theta_size:
            raise ValueError(
                f"Sonnenfeld predictive diagnostics for '{model_name}' expected "
                f"{expected_theta_size} parameters, received {theta.shape[0]}."
            )
        parent_indices = rng.choice(available_parent, size=n_parent, replace=available_parent < n_parent)
        normals = np.asarray(context.base_normals[parent_indices], dtype=float)
        zd = np.asarray(context.parent_sample_zd[parent_indices], dtype=float)
        mstar = np.asarray(context.parent_sample_mstar[parent_indices], dtype=float)
        log_re = np.asarray(context.parent_sample_log_re[parent_indices], dtype=float)
        delta_r = np.asarray(context.parent_sample_delta_r[parent_indices], dtype=float)
        n_values = np.full(n_parent, float(context.n_fixed), dtype=float)
        if int(context.use_sersic_index) == 1:
            n_values = np.maximum(4.0 + 0.4 * normals[:, 7], 0.5)

        mstar_shift = mstar - float(context.mstar_pivot)
        sigma5 = max(float(theta[3]), 1.0e-8)
        log_mass = float(theta[0]) + float(theta[1]) * mstar_shift + float(theta[2]) * delta_r + sigma5 * normals[:, 3]
        if is_sigma_star_gamma_model:
            # The sigma-star peer model changes only the gamma population
            # relation.  The inference posterior defines the predictor as
            # log10(Sigma_*) - 9, where Sigma_* = M_* / (2 pi R_e^2), using the
            # same active unit convention already packed into the context.
            sigma_gamma = max(float(theta[6]), 1.0e-8)
            sigma_zs = max(float(theta[8]), 1.0e-8)
            sigma_star_shift9p0 = mstar - LOG10_2PI - 2.0 * log_re - 9.0
            gamma_mean = float(theta[4]) + float(theta[5]) * sigma_star_shift9p0
            mu_zs = float(theta[7])
            theta0 = float(theta[9])
            loga = float(theta[10])
        else:
            # Original Sonnenfeld keeps the paper-native two-predictor gamma
            # relation, gamma ~ M_* shift + size residual.  Keep this branch
            # byte-for-byte close to the existing semantic contract so adding
            # the peer model does not alter old diagnostics.
            sigma_gamma = max(float(theta[7]), 1.0e-8)
            sigma_zs = max(float(theta[9]), 1.0e-8)
            gamma_mean = float(theta[4]) + float(theta[5]) * mstar_shift + float(theta[6]) * delta_r
            mu_zs = float(theta[8])
            theta0 = float(theta[10])
            loga = float(theta[11])
        gamma = np.clip(
            gamma_mean + sigma_gamma * normals[:, 4],
            float(context.gamma_trunc_low),
            float(context.gamma_trunc_high),
        )
        zs = np.maximum(mu_zs + sigma_zs * normals[:, 5], zd + float(context.source_lens_redshift_gap) + 1.0e-3)

        theta_e = np.zeros(n_parent, dtype=float)
        sigma_model = np.zeros(n_parent, dtype=float)
        selection_weight = np.zeros(n_parent, dtype=float)
        for sample_index in range(n_parent):
            theta_e[sample_index] = theta_ein_arcsec(
                zd[sample_index],
                zs[sample_index],
                log_mass[sample_index],
                gamma[sample_index],
                context.z_grid,
                context.chi_kpc_grid,
                float(context.mass_radius_kpc),
                float(context.mass_log_physical_offset),
            )
            sigma_unit = interp_sigma_unit_clip(
                gamma[sample_index],
                zd[sample_index],
                log_re[sample_index],
                n_values[sample_index],
                context.population_gamma_axis,
                context.population_zd_axis,
                context.population_log_re_kpc_axis,
                context.population_n_axis,
                context.population_sigma_unit_grid,
                1,
            )
            if theta_e[sample_index] <= 0.0 or sigma_unit <= 0.0:
                continue
            sigma_model[sample_index] = sigma_model_from_s2(sigma_unit, log_mass[sample_index])
            sigma_proxy = sigma_model[sample_index] * (1.0 + float(context.sigma_proxy_fractional_scatter) * normals[sample_index, 6])
            theta_est = theta_e_est_from_sigma_proxy(
                sigma_proxy,
                zd[sample_index],
                zs[sample_index],
                context.z_grid,
                context.chi_kpc_grid,
            )
            selection_weight[sample_index] = cross_section_find_weight(
                theta_e[sample_index],
                gamma[sample_index],
                theta_est,
                theta0,
                loga,
                context.cs_theta_e_axis,
                context.cs_gamma_axis,
                context.cs_cross_section_grid,
            )

        theta_indices = _draw_weighted_indices(rng, selection_weight, THETA_SAMPLE_SIZE)
        sigma_indices = _draw_weighted_indices(rng, selection_weight, SIGMA_SAMPLE_SIZE)
        source_arrays = {
            "theta_ein": theta_e,
            "gamma": gamma,
            "zd": zd,
            "zs": zs,
            mass_label: log_mass,
            "re_kpc": np.power(10.0, log_re),
            "n": n_values,
        }
        for key, values in source_arrays.items():
            theta_latent[key][draw_index] = values[theta_indices]
            sigma_latent[key][draw_index] = values[sigma_indices]
        sigma_latent["sigma"][draw_index] = sigma_model[sigma_indices]
        theta_summary = _summary_statistics(theta_e[theta_indices])
        sigma_summary = _summary_statistics(sigma_model[sigma_indices])
        for stat_index, stat_name in enumerate(SUMMARY_STAT_NAMES):
            theta_stats[stat_name][draw_index] = theta_summary[stat_index]
            sigma_stats[stat_name][draw_index] = sigma_summary[stat_index]

        detectable_weight = np.where(theta_e > 0.0, 1.0, 0.0)
        for quantity_name, y_values in ((mass_label, log_mass), ("gamma", gamma), ("sigma_ap", sigma_model)):
            parent, counts, _ = _bin_weighted_means(mstar, y_values, mass_bin_edges, None)
            detectable, _, detectable_sums = _bin_weighted_means(mstar, y_values, mass_bin_edges, detectable_weight)
            selected, _, selected_sums = _bin_weighted_means(mstar, y_values, mass_bin_edges, selection_weight)
            trend_draws[quantity_name]["parent"][draw_index] = parent
            trend_draws[quantity_name]["detectable"][draw_index] = detectable
            trend_draws[quantity_name]["selected"][draw_index] = selected
            if quantity_name == mass_label:
                parent_counts[draw_index] = counts
                detectable_weight_sums[draw_index] = detectable_sums
                selected_weight_sums[draw_index] = selected_sums

        sigma_star = mstar - LOG10_2PI - 2.0 * log_re
        for target, edges, counts_store, det_store, sel_store, x_values in (
            (gamma_vs_logre, log_re_bin_edges, gamma_logre_counts, gamma_logre_detectable, gamma_logre_selected, log_re),
            (gamma_vs_sigma_star, sigma_star_bin_edges, gamma_sigma_counts, gamma_sigma_detectable, gamma_sigma_selected, sigma_star),
            (gamma_vs_delta_r, delta_r_bin_edges, gamma_delta_counts, gamma_delta_detectable, gamma_delta_selected, delta_r),
        ):
            parent, counts, _ = _bin_weighted_means(x_values, gamma, edges, None)
            detectable, _, det_sums = _bin_weighted_means(x_values, gamma, edges, detectable_weight)
            selected, _, sel_sums = _bin_weighted_means(x_values, gamma, edges, selection_weight)
            target["parent"][draw_index] = parent
            target["detectable"][draw_index] = detectable
            target["selected"][draw_index] = selected
            counts_store[draw_index] = counts
            det_store[draw_index] = det_sums
            sel_store[draw_index] = sel_sums

    return {
        "theta_latent": theta_latent,
        "sigma_latent": sigma_latent,
        "theta_replicated_stats": theta_stats,
        "sigma_replicated_stats": sigma_stats,
        "trend_draws": trend_draws,
        "parent_bin_counts_draws": parent_counts,
        "detectable_weight_sums_draws": detectable_weight_sums,
        "selected_weight_sums_draws": selected_weight_sums,
        "gamma_vs_logre_draws": gamma_vs_logre,
        "gamma_vs_logre_parent_bin_counts_draws": gamma_logre_counts,
        "gamma_vs_logre_detectable_weight_sums_draws": gamma_logre_detectable,
        "gamma_vs_logre_selected_weight_sums_draws": gamma_logre_selected,
        "gamma_vs_sigma_star_draws": gamma_vs_sigma_star,
        "gamma_vs_sigma_star_parent_bin_counts_draws": gamma_sigma_counts,
        "gamma_vs_sigma_star_detectable_weight_sums_draws": gamma_sigma_detectable,
        "gamma_vs_sigma_star_selected_weight_sums_draws": gamma_sigma_selected,
        "gamma_vs_delta_r_draws": gamma_vs_delta_r,
        "gamma_vs_delta_r_parent_bin_counts_draws": gamma_delta_counts,
        "gamma_vs_delta_r_detectable_weight_sums_draws": gamma_delta_detectable,
        "gamma_vs_delta_r_selected_weight_sums_draws": gamma_delta_selected,
    }


def _build_sonnenfeld_trend_panel_order(mass_definition: MassDefinition) -> tuple[str, ...]:
    """Return the declared Sonnenfeld trend-panel contract."""

    del mass_definition
    return ("theta_ein", "sigma_ap", "gamma", "gamma_vs_logre_kpc")


def get_predictive_definition(model_name: str) -> PredictiveDefinition:
    """Return a Sonnenfeld predictive definition for one concrete registry name."""

    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported Sonnenfeld predictive model '{model_name}'.")

    def run_model_diagnostics(**kwargs) -> dict[str, Any]:
        """
        Bind the concrete model name into the shared Sonnenfeld diagnostics.

        The generic PPT runner intentionally passes only scientific arrays and
        context objects to the model hook.  Capturing ``model_name`` here keeps
        the 12D paper-gamma and 11D sigma-star-gamma theta semantics inside the
        model-specific adapter instead of leaking another branch into the
        generic workflow.
        """

        return _run_sonnenfeld_parent_diagnostics(model_name=model_name, **kwargs)

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
