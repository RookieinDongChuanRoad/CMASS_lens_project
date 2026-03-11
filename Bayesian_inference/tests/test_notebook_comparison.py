"""
Tests for the notebook-vs-pipeline comparison helpers.

The comparison workflow is intentionally separate from the production PPC CLI:
it exists to answer a scientific debugging question about why a historical
notebook and the current pipeline behave differently on the same chain. These
tests lock the comparison contract so future refactors do not silently change
the baseline definition.
"""

from __future__ import annotations

import json
from pathlib import Path

import emcee
import h5py
import numpy as np

from cmass_lens_inference.model import LOG_PROB_BLOB_DTYPE


NOTEBOOK_PARAMETER_ORDER = (
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
)


def _write_notebook_sigma_table(path: Path) -> Path:
    """Create a notebook-style HDF5 sigma table with axis-order-sensitive values."""

    z_axis = np.array([0.50, 0.60], dtype=float)
    log_re_axis = np.array([0.70, 0.80], dtype=float)
    log_n_axis = np.array([0.30, 0.40], dtype=float)
    gamma_axis = np.array([1.80, 2.00], dtype=float)

    values = np.zeros((z_axis.size, log_re_axis.size, log_n_axis.size, gamma_axis.size), dtype=float)
    for z_index in range(z_axis.size):
        for log_re_index in range(log_re_axis.size):
            for log_n_index in range(log_n_axis.size):
                for gamma_index in range(gamma_axis.size):
                    values[z_index, log_re_index, log_n_index, gamma_index] = (
                        1000.0 * z_index
                        + 100.0 * log_re_index
                        + 10.0 * log_n_index
                        + gamma_index
                        + 1.0
                    )

    with h5py.File(path, "w") as handle:
        handle.create_dataset("z_grid", data=z_axis)
        handle.create_dataset("logRe_grid", data=log_re_axis)
        handle.create_dataset("logn_grid", data=log_n_axis)
        handle.create_dataset("gamma_grid", data=gamma_axis)
        handle.create_dataset("s2_grid", data=values)
    return path


def _seed_chain_backend(path: Path, parameter_center: np.ndarray, n_steps: int = 4, n_walkers: int = 6) -> Path:
    """Create a tiny real `emcee` backend chain for comparison integration tests."""

    backend = emcee.backends.HDFBackend(str(path))
    backend.reset(n_walkers, parameter_center.shape[0])
    blobs = np.zeros(n_walkers, dtype=LOG_PROB_BLOB_DTYPE)
    blobs["parallel_strategy"] = b"off"
    backend.grow(n_steps, blobs)
    random_state = np.random.RandomState(123).get_state()

    for step_index in range(n_steps):
        coords = np.tile(parameter_center, (n_walkers, 1))
        coords += 1.0e-3 * step_index
        coords += np.linspace(0.0, 1.0e-4, n_walkers, dtype=float)[:, None]
        log_prob = np.full(n_walkers, -10.0 + 0.05 * step_index, dtype=float)
        state = emcee.State(coords, log_prob=log_prob, blobs=blobs.copy(), random_state=random_state)
        backend.save_step(state, np.ones(n_walkers, dtype=bool))
    return path


def _write_fake_population_model(path: Path) -> Path:
    """Create a tiny deterministic module that mimics the notebook API surface."""

    path.write_text(
        """
import numpy as np


def draw_sample_test(size, eta_5, eta_gamma):
    base = np.linspace(0.0, 1.0, size)
    log_mstar = 11.1 + 0.05 * base
    log_n = 0.45 + 0.03 * base
    log_re = 0.75 + 0.02 * base
    m5 = np.full(size, eta_5[0], dtype=float)
    gamma = np.clip(np.full(size, eta_gamma[0], dtype=float), 1.25, 2.75)
    return log_mstar, log_n, log_re, m5, gamma


def zd_generator(size):
    return np.linspace(0.50, 0.60, size)


def zs_generator(eta_s, size):
    return np.linspace(1.30, 1.60, size)


def theta_ein(zl, zs, m5, gamma):
    return 0.9 + 0.02 * np.arange(zl.size, dtype=float)


def g_thetae_gamma5(theta_ein, gamma, g_gamma_grid, cs_over_theta_ein_grid):
    return np.ones_like(theta_ein, dtype=float)


def Pfind_sigmoid_thetae_etaf(theta_ein, eta_f):
    return np.full_like(theta_ein, 0.5, dtype=float)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_load_notebook_sigma_interpolator_preserves_notebook_axis_order(tmp_path: Path) -> None:
    """The notebook sigma loader must keep the `(z, logRe, logn, gamma)` axis order."""

    from cmass_lens_inference.notebook_comparison import load_notebook_sigma_interpolator

    table_path = _write_notebook_sigma_table(tmp_path / "jeans_sers_grid.h5")
    interpolator = load_notebook_sigma_interpolator(table_path)

    actual = interpolator.evaluate(
        zd=np.array([0.60]),
        log_re=np.array([0.80]),
        log_n=np.array([0.40]),
        gamma=np.array([2.00]),
    )

    assert actual.shape == (1,)
    assert actual[0] == 1112.0


def test_map_notebook_theta_to_pipeline_theta_swaps_the_final_two_columns() -> None:
    """Notebook chains store `(loga, theta0)` while the pipeline expects `(theta0, loga)`."""

    from cmass_lens_inference.notebook_comparison import map_notebook_theta_to_pipeline_theta

    notebook_theta = np.arange(len(NOTEBOOK_PARAMETER_ORDER), dtype=float)
    pipeline_theta = map_notebook_theta_to_pipeline_theta(notebook_theta)

    assert pipeline_theta.shape == notebook_theta.shape
    assert np.allclose(pipeline_theta[:10], notebook_theta[:10])
    assert pipeline_theta[10] == notebook_theta[11]
    assert pipeline_theta[11] == notebook_theta[10]


def test_compute_sigma_model_from_notebook_interpolator_uses_z_logre_logn_gamma_order_without_noise(
    tmp_path: Path,
) -> None:
    """The comparison helper must query sigma using notebook order and return noiseless `sigma_model`."""

    from cmass_lens_inference.notebook_comparison import (
        compute_sigma_model_from_notebook_interpolator,
        load_notebook_sigma_interpolator,
    )

    table_path = _write_notebook_sigma_table(tmp_path / "jeans_sers_grid.h5")
    interpolator = load_notebook_sigma_interpolator(table_path)

    sigma_model = compute_sigma_model_from_notebook_interpolator(
        sigma_interpolator=interpolator,
        zd=np.array([0.60]),
        re_kpc=np.array([10.0**0.80]),
        n_values=np.array([10.0**0.40]),
        gamma=np.array([2.00]),
        m5=np.array([0.0]),
        add_noise=False,
        rng=np.random.default_rng(7),
    )

    assert sigma_model.shape == (1,)
    assert np.allclose(sigma_model[0], np.sqrt(1112.0))


def test_run_notebook_pipeline_comparison_generates_expected_artifacts(
    tmp_path: Path,
    synthetic_config_path: Path,
    synthetic_cross_section_file: Path,
) -> None:
    """A tiny end-to-end comparison should write the agreed artifact set."""

    from cmass_lens_inference.notebook_comparison import run_notebook_pipeline_comparison

    sigma_table_path = _write_notebook_sigma_table(tmp_path / "jeans_sers_grid.h5")
    population_model_path = _write_fake_population_model(tmp_path / "Population_model.py")
    chain_path = _seed_chain_backend(
        tmp_path / "full_0103.h5",
        parameter_center=np.array([11.32, 0.59, -0.11, 0.06, 1.99, 0.10, -0.67, 0.149, 1.8, 0.215, 0.9, 0.7]),
    )

    result = run_notebook_pipeline_comparison(
        chain_path=chain_path,
        pipeline_config_path=synthetic_config_path,
        population_model_path=population_model_path,
        sigma_table_path=sigma_table_path,
        cross_section_path=synthetic_cross_section_file,
        output_dir=tmp_path / "comparison_output",
        discard=1,
        max_samples=3,
        num_parents=16,
        theta_sample_size=22,
        sigma_sample_size=7,
    )

    assert result.result_dir.exists()
    assert (result.result_dir / "comparison_summary.json").exists()
    assert (result.result_dir / "paired_differences.npz").exists()
    assert (result.result_dir / "comparison_overview.png").exists()
    assert (result.result_dir / "run_manifest.json").exists()

    summary = json.loads((result.result_dir / "comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["posterior_sample_count"] == 3
    assert summary["sample_sizes"]["theta_ein"] == 22
    assert summary["sample_sizes"]["sigma"] == 7
    assert "notebook_baseline" in summary
    assert "pipeline_matched" in summary
    assert "paired_differences" in summary
