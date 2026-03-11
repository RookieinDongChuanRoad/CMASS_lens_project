"""
Notebook-vs-pipeline comparison helpers.

This module exists for one narrow purpose: compare the historical notebook
posterior-predictive workflow against the current local pipeline on the same
posterior chain, while deliberately matching the notebook's scientific
contract as closely as possible.

Why this module is separate from `posterior_predictive.py`:
- the production PPC code has already moved to a different public contract
  (`23/7` sample sizes, `mean` statistics, optional sigma noise, and the
  newer sigma-table loader)
- this comparison task is a scientific debugging exercise, not a production
  analysis step
- forcing notebook quirks into the production module would make both paths
  harder to maintain

The comparison contract implemented here is the user-approved corrected
version of the notebook's "Inference with all 22 galaxies" workflow:
- use the same post-burn-in chain for both engines
- use `22` theta_E lenses and `7` sigma lenses
- use `num_parents = 10000` unless explicitly overridden
- use the notebook-native sigma interpolator:
  `RegularGridInterpolator((z_grid, logRe_grid, logn_grid, gamma_grid), s2_grid, ...)`
- use sigmoid selection
- compare `median/std/p10/p90`
- do *not* add extra sigma noise
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .compiled_context import build_compiled_context
from .config import load_runtime_config
from .io import load_cross_section_grid
from .posterior_predictive import _draw_candidate_population, _draw_replicated_lenses
from .types import DataConfig, NotebookComparisonResult


NOTEBOOK_SELECTION_FUNCTION = "sigmoid"
NOTEBOOK_SUMMARY_STAT_NAMES = ("median", "std", "p10", "p90")


@dataclass(frozen=True)
class NotebookSigmaInterpolator:
    """
    Thin wrapper around the notebook-native sigma interpolation table.

    The wrapper exists for two reasons:
    1. The tests need an explicit object whose API documents the expected
       notebook axis order.
    2. The production `SigmaUnitTable` reorders axes to fit the pipeline's
       internal conventions. This comparison must *not* do that.
    """

    z_axis: np.ndarray
    log_re_axis: np.ndarray
    log_n_axis: np.ndarray
    gamma_axis: np.ndarray
    values: np.ndarray
    interpolator: RegularGridInterpolator

    def evaluate(
        self,
        zd: np.ndarray,
        log_re: np.ndarray,
        log_n: np.ndarray,
        gamma: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the notebook table in the notebook's original axis order.

        Important: we intentionally do not transpose or reorder the axes.
        The whole point of this comparison is to isolate implementation
        differences while holding the notebook interpolation contract fixed.
        """

        query_points = np.column_stack((zd, log_re, log_n, gamma))
        return np.asarray(self.interpolator(query_points), dtype=float)


def load_notebook_sigma_interpolator(table_path: str | Path) -> NotebookSigmaInterpolator:
    """
    Load the notebook-native sigma table without any pipeline-side axis remap.

    Expected schema:
    - `z_grid`
    - `logRe_grid`
    - `logn_grid`
    - `gamma_grid`
    - `s2_grid`
    """

    resolved_path = Path(table_path).expanduser().resolve()
    with h5py.File(resolved_path, "r") as handle:
        z_axis = np.asarray(handle["z_grid"], dtype=float)
        log_re_axis = np.asarray(handle["logRe_grid"], dtype=float)
        log_n_axis = np.asarray(handle["logn_grid"], dtype=float)
        gamma_axis = np.asarray(handle["gamma_grid"], dtype=float)
        values = np.asarray(handle["s2_grid"], dtype=float)

    interpolator = RegularGridInterpolator(
        (z_axis, log_re_axis, log_n_axis, gamma_axis),
        values,
        bounds_error=False,
        fill_value=None,
    )
    return NotebookSigmaInterpolator(
        z_axis=z_axis,
        log_re_axis=log_re_axis,
        log_n_axis=log_n_axis,
        gamma_axis=gamma_axis,
        values=values,
        interpolator=interpolator,
    )


def map_notebook_theta_to_pipeline_theta(notebook_theta: np.ndarray) -> np.ndarray:
    """
    Convert notebook chain order into the pipeline's canonical parameter order.

    The first ten parameters match. The only known difference is that the
    notebook chain stores `(loga, theta0)` while the current pipeline expects
    `(theta0, loga)`.
    """

    theta = np.asarray(notebook_theta, dtype=float)
    if theta.shape != (12,):
        raise ValueError("Notebook theta must contain exactly 12 parameters.")

    pipeline_theta = theta.copy()
    pipeline_theta[10] = theta[11]
    pipeline_theta[11] = theta[10]
    return pipeline_theta


def compute_sigma_model_from_notebook_interpolator(
    sigma_interpolator: NotebookSigmaInterpolator,
    zd: np.ndarray,
    re_kpc: np.ndarray,
    n_values: np.ndarray,
    gamma: np.ndarray,
    m5: np.ndarray,
    *,
    add_noise: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute sigma using the notebook's raw `s2_grid` contract.

    Parameters
    ----------
    sigma_interpolator:
        The notebook-native interpolator that expects `(z, logRe, logn, gamma)`.
    zd, re_kpc, n_values, gamma, m5:
        Per-lens latent quantities. `re_kpc` and `n_values` are converted to
        `log10` before interpolation because that is how the notebook table is
        defined.
    add_noise:
        Kept explicit for clarity. The notebook-matched comparison path must
        pass `False`, while tests can still exercise the optional branch.
    rng:
        Random generator used only if `add_noise=True`.
    """

    s2_values = sigma_interpolator.evaluate(
        zd=np.asarray(zd, dtype=float),
        log_re=np.log10(np.maximum(np.asarray(re_kpc, dtype=float), 1.0e-12)),
        log_n=np.log10(np.maximum(np.asarray(n_values, dtype=float), 1.0e-12)),
        gamma=np.asarray(gamma, dtype=float),
    )
    sigma_model = np.sqrt(np.maximum(s2_values * np.power(10.0, np.asarray(m5, dtype=float)), 0.0))
    if not add_noise:
        return sigma_model

    # This branch is intentionally present only for completeness and tests.
    # The notebook-matched comparison uses noiseless sigma by contract.
    return rng.normal(loc=sigma_model, scale=0.0625 * np.maximum(sigma_model, 1.0e-12))


def _load_population_model_module(population_model_path: Path) -> ModuleType:
    """
    Import the notebook's `Population_model.py` without requiring installation.

    The notebook module relies on sibling imports from the same directory, so
    we temporarily prepend that directory to `sys.path` while loading it.
    """

    resolved_path = population_model_path.expanduser().resolve()
    module_dir = str(resolved_path.parent)
    spec = importlib.util.spec_from_file_location("notebook_population_model", resolved_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Population_model module from '{resolved_path}'.")

    module = importlib.util.module_from_spec(spec)
    original_sys_path = list(sys.path)
    try:
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_sys_path
    return module


def _load_flattened_chain(chain_path: Path, discard: int, max_samples: int | None) -> np.ndarray:
    """
    Load the flattened post-burn-in chain in the notebook's own access pattern.

    The notebook used `backend.get_chain(flat=True, discard=1000)`. We keep
    that access pattern instead of the production PPC sampler's "sample with
    replacement" behavior because the comparison question is about historical
    workflow differences on the same explicit set of posterior samples.
    """

    backend = emcee.backends.HDFBackend(str(chain_path))
    chain = np.asarray(backend.get_chain(flat=True, discard=discard), dtype=float)
    if chain.shape[0] == 0:
        raise ValueError(f"Discard={discard} removes all posterior samples from '{chain_path}'.")
    if max_samples is not None:
        chain = chain[: int(max_samples)]
    return chain


def _notebook_style_summary(values: np.ndarray) -> dict[str, float]:
    """Compute the corrected notebook summary statistics for one replicated sample."""

    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "p10": float(np.percentile(array, 10.0)),
        "p90": float(np.percentile(array, 90.0)),
    }


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    """Summarize one replicated-statistic distribution across posterior draws."""

    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "p10": float(np.percentile(array, 10.0)),
        "p50": float(np.percentile(array, 50.0)),
        "p90": float(np.percentile(array, 90.0)),
    }


def _build_pipeline_context(
    pipeline_config_path: Path,
    cross_section_path: Path,
    observation_path: Path | None,
) -> tuple[Any, Any]:
    """
    Build the local pipeline context used by the notebook-matched engine.

    We override the data-file locations instead of mutating the original YAML.
    That keeps the comparison self-contained and ensures the output manifest can
    state exactly which external notebook assets were used.
    """

    runtime_config = load_runtime_config(pipeline_config_path)
    overridden_data = DataConfig(
        observation_path=runtime_config.data.observation_path if observation_path is None else observation_path,
        cross_section_path=cross_section_path,
    )
    runtime_config = replace(runtime_config, data=overridden_data)
    compiled_context, profile_spec, _, _, _, _ = build_compiled_context(runtime_config)
    return compiled_context, profile_spec


def _draw_notebook_candidates(
    population_model: ModuleType,
    theta: np.ndarray,
    cross_section_gamma: np.ndarray,
    cross_section_values: np.ndarray,
    num_parents: int,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Draw one notebook-style parent population for a single posterior sample.

    The corrected comparison seeds NumPy *before* parent generation so the
    baseline becomes deterministic per posterior index. This is more stable
    than the original notebook's mix of global RNG state and joblib process
    scheduling, while preserving the notebook's actual population generator.
    """

    np.random.seed(seed)

    eta_5 = theta[0:4]
    eta_gamma = theta[4:8]
    eta_s = theta[8:10]
    eta_f = theta[10:12]

    _, log_n, log_re, m5, gamma = population_model.draw_sample_test(num_parents, eta_5, eta_gamma)
    zd = population_model.zd_generator(num_parents)
    zs = population_model.zs_generator(eta_s, num_parents)
    theta_ein = population_model.theta_ein(zd, zs, m5, gamma)
    selection_area = population_model.g_thetae_gamma5(theta_ein, gamma, cross_section_gamma, cross_section_values)
    discovery_probability = population_model.Pfind_sigmoid_thetae_etaf(theta_ein, eta_f)

    valid = (
        np.isfinite(theta_ein)
        & np.isfinite(gamma)
        & np.isfinite(m5)
        & np.isfinite(log_re)
        & np.isfinite(log_n)
        & (theta_ein > 0.0)
        & (zs > zd)
        & (selection_area > 0.0)
        & (discovery_probability > 0.0)
    )
    weights = np.where(valid, selection_area * discovery_probability, 0.0)
    return {
        "theta_ein": np.asarray(theta_ein, dtype=float),
        "gamma": np.asarray(gamma, dtype=float),
        "zd": np.asarray(zd, dtype=float),
        "zs": np.asarray(zs, dtype=float),
        "m5": np.asarray(m5, dtype=float),
        "log_re": np.asarray(log_re, dtype=float),
        "log_n": np.asarray(log_n, dtype=float),
        "weights": np.asarray(weights, dtype=float),
    }


def _draw_weighted_sample(
    candidates: dict[str, np.ndarray],
    sample_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Draw one selected-lens sample from notebook-style weighted candidates.

    The real notebook used `replace=False`. We preserve that whenever possible,
    but fall back to replacement for tiny synthetic fixtures so the test suite
    can exercise the workflow without constructing huge parent pools.
    """

    positive_indices = np.flatnonzero(candidates["weights"] > 0.0)
    if positive_indices.size == 0:
        raise ValueError("Notebook candidate population contains no positive selection weights.")

    probabilities = candidates["weights"][positive_indices]
    probabilities = probabilities / probabilities.sum()
    replace_choice = positive_indices.size < sample_size
    chosen = rng.choice(positive_indices, size=sample_size, replace=replace_choice, p=probabilities)
    return {name: values[chosen] for name, values in candidates.items() if name != "weights"}


def _run_notebook_corrected_engine(
    theta: np.ndarray,
    population_model: ModuleType,
    sigma_interpolator: NotebookSigmaInterpolator,
    cross_section_gamma: np.ndarray,
    cross_section_values: np.ndarray,
    num_parents: int,
    theta_sample_size: int,
    sigma_sample_size: int,
    *,
    seed: int,
) -> dict[str, dict[str, float]]:
    """
    Execute one corrected notebook replicate.

    The notebook multiplied weights by `norm = P_SL_norm_mc(...)`, but then
    normalized those weights before sampling. Because `norm` is a scalar for
    the given posterior draw, it cancels exactly and does not affect the
    selected-lens distribution. Omitting it here avoids unnecessary expensive
    work while leaving the sampling law unchanged.
    """

    candidates = _draw_notebook_candidates(
        population_model=population_model,
        theta=theta,
        cross_section_gamma=cross_section_gamma,
        cross_section_values=cross_section_values,
        num_parents=num_parents,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    theta_sample = _draw_weighted_sample(candidates, theta_sample_size, rng)
    sigma_sample = _draw_weighted_sample(candidates, sigma_sample_size, rng)
    sigma_values = compute_sigma_model_from_notebook_interpolator(
        sigma_interpolator=sigma_interpolator,
        zd=sigma_sample["zd"],
        re_kpc=np.power(10.0, sigma_sample["log_re"]),
        n_values=np.power(10.0, sigma_sample["log_n"]),
        gamma=sigma_sample["gamma"],
        m5=sigma_sample["m5"],
        add_noise=False,
        rng=rng,
    )
    return {
        "theta_ein": _notebook_style_summary(theta_sample["theta_ein"]),
        "sigma": _notebook_style_summary(sigma_values),
    }


def _run_pipeline_matched_engine(
    notebook_theta: np.ndarray,
    compiled_context: Any,
    profile_spec: Any,
    sigma_interpolator: NotebookSigmaInterpolator,
    num_parents: int,
    theta_sample_size: int,
    sigma_sample_size: int,
    *,
    seed: int,
) -> dict[str, dict[str, float]]:
    """
    Execute one local-pipeline replicate with notebook-matched downstream rules.

    This path deliberately reuses the current pipeline's latent population
    generator and selection implementation, but forces the notebook's
    comparison contract downstream:
    - same parent-pool size as the notebook
    - same `22/7` selected sample sizes
    - same sigma interpolator and query order
    - same noiseless sigma definition
    - same `median/std/p10/p90` summaries
    """

    pipeline_theta = map_notebook_theta_to_pipeline_theta(notebook_theta)
    rng = np.random.default_rng(seed)
    candidates = _draw_candidate_population(
        theta=pipeline_theta,
        profile=profile_spec,
        context=compiled_context,
        rng=rng,
        candidate_pool_size=num_parents,
    )
    theta_sample = _draw_replicated_lenses(candidates, theta_sample_size, rng)
    sigma_sample = _draw_replicated_lenses(candidates, sigma_sample_size, rng)
    sigma_values = compute_sigma_model_from_notebook_interpolator(
        sigma_interpolator=sigma_interpolator,
        zd=sigma_sample["zd"],
        re_kpc=sigma_sample["re_kpc"],
        n_values=sigma_sample["n"],
        gamma=sigma_sample["gamma"],
        m5=sigma_sample["m5"],
        add_noise=False,
        rng=rng,
    )
    return {
        "theta_ein": _notebook_style_summary(theta_sample["theta_ein"]),
        "sigma": _notebook_style_summary(sigma_values),
    }


def _summaries_to_arrays(engine_results: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, np.ndarray]]:
    """Convert per-sample dictionaries into vectorized arrays for output and plotting."""

    grouped: dict[str, dict[str, list[float]]] = {
        "theta_ein": {name: [] for name in NOTEBOOK_SUMMARY_STAT_NAMES},
        "sigma": {name: [] for name in NOTEBOOK_SUMMARY_STAT_NAMES},
    }
    for result in engine_results:
        for quantity_name in ("theta_ein", "sigma"):
            for stat_name in NOTEBOOK_SUMMARY_STAT_NAMES:
                grouped[quantity_name][stat_name].append(float(result[quantity_name][stat_name]))

    return {
        quantity_name: {
            stat_name: np.asarray(values, dtype=float)
            for stat_name, values in stat_group.items()
        }
        for quantity_name, stat_group in grouped.items()
    }


def _summarize_engine_distributions(engine_arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, dict[str, float]]]:
    """Summarize the distribution of replicated summary statistics."""

    return {
        quantity_name: {
            stat_name: _distribution_summary(stat_values)
            for stat_name, stat_values in quantity_group.items()
        }
        for quantity_name, quantity_group in engine_arrays.items()
    }


def _compute_paired_differences(
    notebook_arrays: dict[str, dict[str, np.ndarray]],
    pipeline_arrays: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, dict[str, float]]]]:
    """Compute paired per-sample differences and their summaries."""

    difference_arrays: dict[str, dict[str, np.ndarray]] = {"theta_ein": {}, "sigma": {}}
    difference_summary: dict[str, dict[str, dict[str, float]]] = {"theta_ein": {}, "sigma": {}}
    for quantity_name in ("theta_ein", "sigma"):
        for stat_name in NOTEBOOK_SUMMARY_STAT_NAMES:
            delta = pipeline_arrays[quantity_name][stat_name] - notebook_arrays[quantity_name][stat_name]
            difference_arrays[quantity_name][stat_name] = delta
            difference_summary[quantity_name][stat_name] = _distribution_summary(delta)
    return difference_arrays, difference_summary


def _write_comparison_figure(
    figure_path: Path,
    notebook_arrays: dict[str, dict[str, np.ndarray]],
    pipeline_arrays: dict[str, dict[str, np.ndarray]],
) -> None:
    """Render a compact overlay figure for the two engines."""

    figure, axes = plt.subplots(2, 4, figsize=(15, 7))
    quantity_specs = (
        ("theta_ein", "Theta_E"),
        ("sigma", "Sigma"),
    )
    for row_index, (quantity_name, label_prefix) in enumerate(quantity_specs):
        for column_index, stat_name in enumerate(NOTEBOOK_SUMMARY_STAT_NAMES):
            axis = axes[row_index, column_index]
            notebook_values = notebook_arrays[quantity_name][stat_name]
            pipeline_values = pipeline_arrays[quantity_name][stat_name]
            finite_values = np.concatenate((notebook_values, pipeline_values))
            if finite_values.size == 0:
                x_min, x_max = -1.0, 1.0
            else:
                x_min = float(np.percentile(finite_values, 2.5))
                x_max = float(np.percentile(finite_values, 97.5))
                if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
                    x_min = float(np.min(finite_values)) - 0.5
                    x_max = float(np.max(finite_values)) + 0.5
            axis.hist(
                notebook_values,
                bins=20,
                range=(x_min, x_max),
                alpha=0.55,
                color="#b08d57",
                edgecolor="#5b4327",
                label="Notebook corrected",
            )
            axis.hist(
                pipeline_values,
                bins=20,
                range=(x_min, x_max),
                alpha=0.45,
                color="#6a8caf",
                edgecolor="#28445a",
                label="Pipeline matched",
            )
            axis.set_title(f"{label_prefix} {stat_name}", fontsize=10)
            axis.tick_params(labelsize=8)
            if row_index == 0 and column_index == 0:
                axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)


def run_notebook_pipeline_comparison(
    *,
    chain_path: str | Path,
    pipeline_config_path: str | Path,
    population_model_path: str | Path,
    sigma_table_path: str | Path,
    cross_section_path: str | Path,
    output_dir: str | Path,
    discard: int = 1000,
    max_samples: int | None = None,
    num_parents: int = 10000,
    theta_sample_size: int = 22,
    sigma_sample_size: int = 7,
    random_seed: int = 20260310,
    observation_path: str | Path | None = None,
) -> NotebookComparisonResult:
    """
    Run the apples-to-apples notebook-vs-pipeline comparison.

    The implementation is intentionally verbose because the scientific question
    here is not merely "did two numbers differ?" but "which implementation
    choice generated the difference?" Each artifact therefore records the chain,
    sigma table, selection contract, sample sizes, and paired-difference
    summaries explicitly.
    """

    resolved_chain_path = Path(chain_path).expanduser().resolve()
    resolved_config_path = Path(pipeline_config_path).expanduser().resolve()
    resolved_population_model_path = Path(population_model_path).expanduser().resolve()
    resolved_sigma_table_path = Path(sigma_table_path).expanduser().resolve()
    resolved_cross_section_path = Path(cross_section_path).expanduser().resolve()
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_observation_path = None if observation_path is None else Path(observation_path).expanduser().resolve()

    posterior_chain = _load_flattened_chain(resolved_chain_path, discard=discard, max_samples=max_samples)
    notebook_sigma_interpolator = load_notebook_sigma_interpolator(resolved_sigma_table_path)
    population_model = _load_population_model_module(resolved_population_model_path)
    cross_section_grid = load_cross_section_grid(resolved_cross_section_path)
    compiled_context, profile_spec = _build_pipeline_context(
        resolved_config_path,
        resolved_cross_section_path,
        resolved_observation_path,
    )
    if getattr(profile_spec, "name", None) != "sersic":
        raise ValueError("Notebook comparison currently only supports the sersic profile workflow.")

    notebook_results: list[dict[str, dict[str, float]]] = []
    pipeline_results: list[dict[str, dict[str, float]]] = []

    for sample_index, notebook_theta in enumerate(posterior_chain):
        sample_seed = int(random_seed + sample_index)
        notebook_results.append(
            _run_notebook_corrected_engine(
                theta=notebook_theta,
                population_model=population_model,
                sigma_interpolator=notebook_sigma_interpolator,
                cross_section_gamma=cross_section_grid.gamma_grid,
                cross_section_values=cross_section_grid.cs_over_theta_ein,
                num_parents=num_parents,
                theta_sample_size=theta_sample_size,
                sigma_sample_size=sigma_sample_size,
                seed=sample_seed,
            )
        )
        pipeline_results.append(
            _run_pipeline_matched_engine(
                notebook_theta=notebook_theta,
                compiled_context=compiled_context,
                profile_spec=profile_spec,
                sigma_interpolator=notebook_sigma_interpolator,
                num_parents=num_parents,
                theta_sample_size=theta_sample_size,
                sigma_sample_size=sigma_sample_size,
                seed=sample_seed,
            )
        )

    notebook_arrays = _summaries_to_arrays(notebook_results)
    pipeline_arrays = _summaries_to_arrays(pipeline_results)
    difference_arrays, difference_summary = _compute_paired_differences(notebook_arrays, pipeline_arrays)

    summary_payload = {
        "chain_path": str(resolved_chain_path),
        "pipeline_config_path": str(resolved_config_path),
        "population_model_path": str(resolved_population_model_path),
        "sigma_table_path": str(resolved_sigma_table_path),
        "cross_section_path": str(resolved_cross_section_path),
        "observation_path": None if resolved_observation_path is None else str(resolved_observation_path),
        "posterior_sample_count": int(posterior_chain.shape[0]),
        "discard": int(discard),
        "num_parents": int(num_parents),
        "sample_sizes": {"theta_ein": int(theta_sample_size), "sigma": int(sigma_sample_size)},
        "selection_function": NOTEBOOK_SELECTION_FUNCTION,
        "statistics": list(NOTEBOOK_SUMMARY_STAT_NAMES),
        "sigma_definition": "sqrt(s2 * 10**m5)",
        "sigma_noise": "disabled",
        "interpolator_contract": "RegularGridInterpolator((z_grid, logRe_grid, logn_grid, gamma_grid), s2_grid, bounds_error=False, fill_value=None)",
        "notebook_baseline": _summarize_engine_distributions(notebook_arrays),
        "pipeline_matched": _summarize_engine_distributions(pipeline_arrays),
        "paired_differences": difference_summary,
    }
    (resolved_output_dir / "comparison_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest_payload = {
        "chain_path": str(resolved_chain_path),
        "posterior_sample_count": int(posterior_chain.shape[0]),
        "discard": int(discard),
        "random_seed_base": int(random_seed),
        "num_parents": int(num_parents),
        "sample_sizes": {"theta_ein": int(theta_sample_size), "sigma": int(sigma_sample_size)},
        "selection_function": NOTEBOOK_SELECTION_FUNCTION,
        "sigma_noise_enabled": False,
        "population_model_path": str(resolved_population_model_path),
        "sigma_table_path": str(resolved_sigma_table_path),
        "cross_section_path": str(resolved_cross_section_path),
        "pipeline_config_path": str(resolved_config_path),
        "observation_path_override": None if resolved_observation_path is None else str(resolved_observation_path),
        "notebook_parameter_order": [
            "mu5_0",
            "beta5",
            "xi5",
            "sigma5",
            "mu_gamma_0",
            "beta_gamma",
            "xi_gamma",
            "sigma_gamma",
            "mu_zs",
            "sigma_zs",
            "loga",
            "theta0",
        ],
        "pipeline_parameter_order": [
            "mu5_0",
            "beta5",
            "xi5",
            "sigma5",
            "mu_gamma_0",
            "beta_gamma",
            "xi_gamma",
            "sigma_gamma",
            "mu_zs",
            "sigma_zs",
            "theta0",
            "loga",
        ],
    }
    (resolved_output_dir / "run_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    npz_payload: dict[str, np.ndarray] = {}
    for engine_name, engine_arrays in (("notebook", notebook_arrays), ("pipeline", pipeline_arrays), ("difference", difference_arrays)):
        for quantity_name, stat_group in engine_arrays.items():
            for stat_name, values in stat_group.items():
                npz_payload[f"{engine_name}_{quantity_name}_{stat_name}"] = np.asarray(values, dtype=float)
    np.savez(resolved_output_dir / "paired_differences.npz", **npz_payload)

    _write_comparison_figure(
        resolved_output_dir / "comparison_overview.png",
        notebook_arrays=notebook_arrays,
        pipeline_arrays=pipeline_arrays,
    )

    return NotebookComparisonResult(
        result_dir=resolved_output_dir,
        status="completed",
        chain_path=resolved_chain_path,
        sigma_table_path=resolved_sigma_table_path,
        population_model_path=resolved_population_model_path,
        posterior_sample_count=int(posterior_chain.shape[0]),
        discard=int(discard),
        sample_sizes={"theta_ein": int(theta_sample_size), "sigma": int(sigma_sample_size)},
        metadata={
            "pipeline_config_path": str(resolved_config_path),
            "cross_section_path": str(resolved_cross_section_path),
            "observation_path": None if resolved_observation_path is None else str(resolved_observation_path),
        },
    )
