"""
Integration tests for the posterior predictive test workflow.

These tests intentionally exercise the public PPC contract end-to-end:
- read a completed run directory and its `chain.h5`
- discard burn-in and draw posterior hyper-parameter samples
- generate independent replicated samples for `theta_E` (23 lenses) and
  `sigma` (7 lenses)
- consume a Jeans interpolation table
- write machine-readable artifacts plus a summary figure

The scientific numbers here are synthetic. The goal is to lock in behavior,
file contracts, and data-flow shape before the production implementation is
written.
"""

from __future__ import annotations

import json
import inspect
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from cmass_lens_inference.model import LOG_PROB_BLOB_DTYPE


PARAMETER_ORDER = (
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
)


def _write_observation_file(path: Path, profile_name: str) -> Path:
    """
    Create a 23-lens synthetic observation file with exactly 7 sigma lenses.

    Why this fixture is larger than the rest of the synthetic test data:
    - the PPC requirements are phrased around a 23-lens `theta_E` sample and a
      7-lens `sigma` sample
    - shrinking those counts in tests would stop the tests from guarding the
      most important output-shape contracts
    """

    gamma_grid = np.linspace(1.2, 2.8, 17)
    with h5py.File(path, "w") as handle:
        for lens_index in range(23):
            group = handle.create_group(f"lens-{lens_index:04d}")
            z_d = 0.46 + 0.01 * lens_index
            z_s = 1.40 + 0.03 * lens_index
            group.attrs["zd"] = z_d
            group.attrs["zs"] = z_s
            group.attrs["logmchab"] = 11.1 + 0.01 * (lens_index % 5)
            group.attrs["logmchab_err"] = 0.05
            group.attrs["nser"] = 3.2 + 0.2 * (lens_index % 6)
            group.attrs["re_arcsec"] = 0.8 + 0.04 * lens_index
            group.attrs["rein_arcsec"] = 0.9 + 0.03 * lens_index
            group.attrs["sigma_crit"] = 2.0e9 + 5.0e7 * lens_index
            group.attrs["r_ein_kpc"] = 5.0 + 0.1 * lens_index
            if profile_name == "devauc":
                group.attrs["logmchab_deV"] = group.attrs["logmchab"] + 0.08
                group.attrs["reff_deV"] = 0.9 + 0.03 * lens_index

            if lens_index < 7:
                group.attrs["num_sigma"] = 1 if lens_index < 4 else 2
                if lens_index < 4:
                    group.attrs["sigma"] = np.asarray([250.0 + 5.0 * lens_index], dtype=float)
                    group.attrs["sigma_err"] = np.asarray([12.0 + lens_index], dtype=float)
                else:
                    group.attrs["sigma"] = np.asarray(
                        [260.0 + 4.0 * lens_index, 262.0 + 4.0 * lens_index],
                        dtype=float,
                    )
                    group.attrs["sigma_err"] = np.asarray([14.0, 16.0], dtype=float)
            else:
                group.attrs["num_sigma"] = 0

            m5_grid = np.linspace(11.6, 10.8, 17) - 0.01 * lens_index
            dm5_dthetaein_grid = np.linspace(-1.9, -1.0, 17)
            group.create_dataset("gamma_grid", data=gamma_grid)
            group.create_dataset("m5_grid", data=m5_grid)
            group.create_dataset("dm5_dthetaein_grid", data=dm5_dthetaein_grid)
    return path


def _write_cross_section_file(path: Path) -> Path:
    """Create a tiny but schema-compatible cross-section file."""

    with h5py.File(path, "w") as handle:
        group = handle.create_group("compressed_grids")
        group.create_dataset("gamma_grids", data=np.linspace(1.2, 2.8, 31))
        group.create_dataset("cs_over_theta_ein_grid", data=np.linspace(0.7, 1.3, 31))
    return path


def _write_config(
    path: Path,
    profile_name: str,
    observation_path: Path,
    cross_section_path: Path,
    output_root: Path,
) -> Path:
    """Create a YAML config snapshot that mirrors a completed run."""

    config = {
        "profile": {"name": profile_name},
        "data": {
            "observation_path": str(observation_path),
            "cross_section_path": str(cross_section_path),
        },
        "sampling": {
            "n_walkers": 24,
            "n_steps": 5,
            "warmup": 2,
            "random_seed": 7,
            "initial_center": {
                "mu5_0": 11.32,
                "beta5": 0.59,
                "xi5": -0.11,
                "sigma5": 0.06,
                "mu_gamma_0": 1.99,
                "beta_gamma": 0.10,
                "xi_gamma": -0.67,
                "sigma_gamma": 0.149,
                "mu_zs": 1.8,
                "sigma_zs": 0.215,
                "theta0": 0.93,
                "loga": 1.0,
            },
            "initial_jitter_scale": 1.0e-3,
        },
        "integration": {
            "gamma_points": 200,
            "mstar_points": 200,
            "normalization_samples": 128,
        },
        "runtime": {
            "distance_table_max_z": 5.0,
            "distance_table_size": 8001,
            "checkpoint_every": 1,
            "parallel_strategy": "off",
            "progress": False,
            "progress_summary_every": 1,
            "show_stage_timing": True,
            "disable_hdf5_file_locking": False,
            "num_threads": 0,
            "reserve_cores": 2,
        },
        "output": {
            "root_dir": str(output_root),
            "run_label": "synthetic_ppc",
            "overwrite_latest": True,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _seed_backend(chain_path: Path, base_theta: np.ndarray, n_steps: int = 5, n_walkers: int = 24) -> None:
    """Create a small but real backend chain for PPC tests."""

    backend = emcee.backends.HDFBackend(str(chain_path))
    backend.reset(n_walkers, base_theta.shape[0])
    blobs = np.zeros(n_walkers, dtype=LOG_PROB_BLOB_DTYPE)
    blobs["parallel_strategy"] = b"off"
    backend.grow(n_steps, blobs)
    random_state = np.random.RandomState(321).get_state()

    for step_index in range(n_steps):
        coords = np.tile(base_theta, (n_walkers, 1))
        coords += 1.0e-3 * step_index
        coords += np.linspace(0.0, 1.0e-4, n_walkers, dtype=float)[:, None]
        log_prob = np.full(n_walkers, -10.0 + 0.1 * step_index, dtype=float)
        state = emcee.State(coords, log_prob=log_prob, blobs=blobs.copy(), random_state=random_state)
        backend.save_step(state, np.ones(n_walkers, dtype=bool))


def _write_sigma_table(path: Path, profile_name: str) -> Path:
    """Create a synthetic sigma-unit interpolation table for one profile."""

    gamma_axis = np.linspace(1.2, 2.8, 9)
    zd_axis = np.linspace(0.43, 0.82, 7)
    log_re_axis = np.linspace(0.45, 1.40, 8)

    if profile_name == "devauc":
        gamma_mesh, zd_mesh, log_re_mesh = np.meshgrid(
            gamma_axis,
            zd_axis,
            log_re_axis,
            indexing="ij",
        )
        values = 0.04 + 0.005 * gamma_mesh + 0.003 * zd_mesh + 0.002 * log_re_mesh
        np.savez(
            path,
            profile_name=profile_name,
            gamma_axis=gamma_axis,
            zd_axis=zd_axis,
            log_re_kpc_axis=log_re_axis,
            s_unit_grid=values,
        )
        return path

    n_axis = np.linspace(2.5, 10.5, 6)
    gamma_mesh, zd_mesh, log_re_mesh, n_mesh = np.meshgrid(
        gamma_axis,
        zd_axis,
        log_re_axis,
        n_axis,
        indexing="ij",
    )
    values = 0.03 + 0.004 * gamma_mesh + 0.002 * zd_mesh + 0.002 * log_re_mesh + 0.001 * n_mesh
    np.savez(
        path,
        profile_name=profile_name,
        gamma_axis=gamma_axis,
        zd_axis=zd_axis,
        log_re_kpc_axis=log_re_axis,
        n_axis=n_axis,
        s_unit_grid=values,
    )
    return path


def _write_sigma_table_hdf5(path: Path, profile_name: str) -> Path:
    """Create a synthetic HDF5 sigma-unit interpolation table for one profile."""

    gamma_axis = np.linspace(1.2, 2.8, 9)
    zd_axis = np.linspace(0.43, 0.82, 7)
    log_re_axis = np.linspace(0.45, 1.40, 8)

    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "sigma_unit_hdf5_v1"
        handle.attrs["quantity_name"] = "S_unit"
        handle.attrs["units"] = "km2 s-2 per 10**m5"
        handle.create_dataset("profile_name", data=np.bytes_(profile_name))
        handle.create_dataset("gamma_axis", data=gamma_axis)
        handle.create_dataset("zd_axis", data=zd_axis)
        handle.create_dataset("log_re_kpc_axis", data=log_re_axis)

        if profile_name == "devauc":
            gamma_mesh, zd_mesh, log_re_mesh = np.meshgrid(
                gamma_axis,
                zd_axis,
                log_re_axis,
                indexing="ij",
            )
            values = 0.04 + 0.005 * gamma_mesh + 0.003 * zd_mesh + 0.002 * log_re_mesh
            handle.create_dataset("s_unit_grid", data=values)
            return path

        n_axis = np.linspace(2.5, 10.5, 6)
        gamma_mesh, zd_mesh, log_re_mesh, n_mesh = np.meshgrid(
            gamma_axis,
            zd_axis,
            log_re_axis,
            n_axis,
            indexing="ij",
        )
        values = 0.03 + 0.004 * gamma_mesh + 0.002 * zd_mesh + 0.002 * log_re_mesh + 0.001 * n_mesh
        handle.create_dataset("n_axis", data=n_axis)
        handle.create_dataset("s_unit_grid", data=values)
    return path


def _write_external_sigma_table_hdf5(path: Path, profile_name: str) -> Path:
    """
    Create an HDF5 sigma table under the fixed external filenames.

    Why this fixture exists:
    - the monitored production paths are expected to be overwritten in place
      under fixed filenames
    - the monitoring workflow should validate the same explicit schema that
      `prepare_intepolation_grids` now writes into `data/external`
    - the filename contract and the HDF5 dataset contract are both important
      for the production watcher
    """

    gamma_axis = np.linspace(1.2, 2.8, 9)
    zd_axis = np.linspace(0.43, 0.82, 7)
    log_re_axis = np.linspace(0.45, 1.40, 8)

    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "sigma_unit_hdf5_v1"
        handle.attrs["quantity_name"] = "S_unit"
        handle.attrs["units"] = "km2 s-2 per 10**m5"
        handle.create_dataset("profile_name", data=np.bytes_(profile_name))
        handle.create_dataset("gamma_axis", data=gamma_axis)
        handle.create_dataset("zd_axis", data=zd_axis)
        handle.create_dataset("log_re_kpc_axis", data=log_re_axis)

        if profile_name == "devauc":
            gamma_mesh, zd_mesh, log_re_mesh = np.meshgrid(
                gamma_axis,
                zd_axis,
                log_re_axis,
                indexing="ij",
            )
            values = 0.04 + 0.005 * gamma_mesh + 0.003 * zd_mesh + 0.002 * log_re_mesh
            handle.create_dataset("s_unit_grid", data=values)
            return path

        n_axis = np.linspace(2.5, 10.5, 6)
        gamma_mesh, zd_mesh, log_re_mesh, n_mesh = np.meshgrid(
            gamma_axis,
            zd_axis,
            log_re_axis,
            n_axis,
            indexing="ij",
        )
        values = 0.03 + 0.004 * gamma_mesh + 0.002 * zd_mesh + 0.002 * log_re_mesh + 0.001 * n_mesh
        handle.create_dataset("n_axis", data=n_axis)
        handle.create_dataset("s_unit_grid", data=values)
    return path


def _touch_with_mtime(path: Path, timestamp: datetime) -> None:
    """Force a deterministic modification time on a file for monitor tests."""

    unix_time = timestamp.timestamp()
    path.touch(exist_ok=True)
    path.chmod(0o644)
    Path(path).touch()
    import os

    os.utime(path, (unix_time, unix_time))


def _build_completed_run(tmp_path: Path, profile_name: str, n_steps: int = 5) -> tuple[Path, Path]:
    """Build a minimal completed run directory plus its sigma table."""

    data_dir = tmp_path / "data"
    run_dir = tmp_path / "runs" / profile_name / f"20260309_140000_{profile_name}_synthetic_ppc"
    data_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    observation_path = _write_observation_file(data_dir / f"{profile_name}_observations.hdf5", profile_name=profile_name)
    cross_section_path = _write_cross_section_file(data_dir / "cs_grid_power.h5")
    config_path = _write_config(
        run_dir / "config_snapshot.yaml",
        profile_name=profile_name,
        observation_path=observation_path,
        cross_section_path=cross_section_path,
        output_root=tmp_path / "outputs",
    )
    base_theta = np.array([11.32, 0.59, -0.11, 0.06, 1.99, 0.10, -0.67, 0.149, 1.8, 0.215, 0.93, 1.0], dtype=float)
    _seed_backend(run_dir / "chain.h5", base_theta=base_theta, n_steps=n_steps)
    sigma_table_path = _write_sigma_table(tmp_path / f"{profile_name}_sigma_table.npz", profile_name=profile_name)
    return run_dir, sigma_table_path


def test_run_posterior_predictive_generates_expected_artifacts_for_sersic(tmp_path: Path) -> None:
    """
    The PPC API should consume a completed run and write all required outputs.

    This test guards the main user-visible contract:
    - burn-in is derived from the stored config snapshot when `auto` is used
    - 23-lens and 7-lens replicated samples are both generated
    - latent arrays are preserved in the `.npz` output
    """

    from cmass_posterior_predictive.predictive import run_posterior_predictive

    run_dir, sigma_table_path = _build_completed_run(tmp_path, profile_name="sersic")
    output_root = tmp_path / "ppc_output"

    result = run_posterior_predictive(
        run_dir=str(run_dir),
        sigma_table_path=str(sigma_table_path),
        output_root_dir=str(output_root),
        n_replicates=6,
        burn_in="auto",
        random_seed=17,
        candidate_pool_size=96,
    )

    assert result.profile_name == "sersic"
    assert result.burn_in_applied == 2
    assert result.result_dir.exists()
    assert (result.result_dir / "ppc_summary.json").exists()
    assert (result.result_dir / "ppc_overview.png").exists()
    assert (result.result_dir / "replicated_statistics.npz").exists()
    assert (result.result_dir / "run_manifest.json").exists()

    payload = json.loads((result.result_dir / "ppc_summary.json").read_text(encoding="utf-8"))
    assert payload["profile_name"] == "sersic"
    assert payload["burn_in_applied"] == 2
    assert payload["requested_n_replicates"] == 6
    assert payload["n_posterior_draws_used"] == 6
    assert payload["posterior_draw_mode"] == "sampled_subset"
    assert payload["sample_sizes"]["theta_ein"] == 23
    assert payload["sample_sizes"]["sigma"] == 7
    assert set(payload["statistics"].keys()) == {"theta_ein", "sigma"}
    assert set(payload["statistics"]["theta_ein"].keys()) == {"median", "std", "p10", "p90"}
    assert set(payload["statistics"]["sigma"].keys()) == {"median", "std", "p10", "p90"}

    arrays = np.load(result.result_dir / "replicated_statistics.npz")
    assert arrays["theta_sample_theta_ein"].shape == (6, 23)
    assert arrays["theta_sample_gamma"].shape == (6, 23)
    assert arrays["theta_sample_zd"].shape == (6, 23)
    assert arrays["theta_sample_zs"].shape == (6, 23)
    assert arrays["theta_sample_m5"].shape == (6, 23)
    assert arrays["theta_sample_re_kpc"].shape == (6, 23)
    assert arrays["theta_sample_n"].shape == (6, 23)
    assert arrays["sigma_sample_sigma"].shape == (6, 7)
    assert arrays["sigma_sample_gamma"].shape == (6, 7)
    assert "theta_stat_mean" not in arrays.files
    assert "sigma_stat_mean" not in arrays.files
    assert np.isfinite(arrays["theta_stat_median"]).all()
    assert np.isfinite(arrays["sigma_stat_median"]).all()


def test_run_posterior_predictive_defaults_to_tail_capped_full_chain_mode(tmp_path: Path) -> None:
    """Omitting `n_replicates` should use the tail-capped full chain by default."""

    from cmass_posterior_predictive.predictive import run_posterior_predictive

    run_dir, sigma_table_path = _build_completed_run(tmp_path, profile_name="devauc")
    result = run_posterior_predictive(
        run_dir=str(run_dir),
        sigma_table_path=str(sigma_table_path),
        output_root_dir=str(tmp_path / "ppc_output"),
        burn_in="auto",
        random_seed=29,
        candidate_pool_size=64,
        worker_processes=1,
    )

    payload = json.loads((result.result_dir / "ppc_summary.json").read_text(encoding="utf-8"))
    arrays = np.load(result.result_dir / "replicated_statistics.npz")

    # The synthetic backend stores 5 steps, and burn-in removes 2 of them.
    assert payload["requested_n_replicates"] is None
    assert payload["n_posterior_draws_used"] == 3 * 24
    assert payload["posterior_draw_mode"] == "tail_capped_full_chain"
    assert arrays["theta_sample_theta_ein"].shape == (3 * 24, 23)
    assert arrays["sigma_sample_sigma"].shape == (3 * 24, 7)
    assert "theta_stat_mean" not in arrays.files
    assert np.isfinite(arrays["theta_stat_median"]).all()


def test_run_posterior_predictive_supports_devauc_sigma_tables(tmp_path: Path) -> None:
    """The PPC API should also accept the 3D devauc sigma table format."""

    from cmass_posterior_predictive.predictive import run_posterior_predictive

    run_dir, sigma_table_path = _build_completed_run(tmp_path, profile_name="devauc")

    result = run_posterior_predictive(
        run_dir=str(run_dir),
        sigma_table_path=str(sigma_table_path),
        output_root_dir=str(tmp_path / "ppc_output"),
        n_replicates=4,
        burn_in=1,
        random_seed=23,
        candidate_pool_size=80,
    )

    arrays = np.load(result.result_dir / "replicated_statistics.npz")
    assert arrays["theta_sample_n"].shape == (4, 23)
    assert np.allclose(arrays["theta_sample_n"], 4.0)
    assert np.isfinite(arrays["sigma_sample_sigma"]).all()


def test_sigma_unit_table_from_hdf5_supports_clipped_sersic_queries(tmp_path: Path) -> None:
    """The sigma-table loader should understand the new HDF5 schema."""

    from cmass_posterior_predictive.predictive import SigmaUnitTable

    table_path = _write_sigma_table_hdf5(tmp_path / "sersic_sigma_table.h5", profile_name="sersic")
    table = SigmaUnitTable.from_path(table_path)

    actual = table.evaluate(
        gamma=np.array([1.1, 1.8, 2.9]),
        zd=np.array([0.40, 0.60, 0.90]),
        log_re_kpc=np.array([0.30, 0.90, 1.50]),
        n_values=np.array([2.0, 6.0, 12.0]),
    )

    assert table.profile_name == "sersic"
    assert table.n_axis is not None
    assert np.isfinite(actual).all()


def test_histogram_panel_writes_left_and_right_tail_labels() -> None:
    """The PPC panel should annotate both posterior-predictive tail percentages."""

    from cmass_posterior_predictive.predictive import _write_histogram_panel

    figure, axis = plt.subplots()
    try:
        _write_histogram_panel(
            axis,
            values=np.linspace(0.8, 1.2, 50),
            observed=1.0,
            title=r"$\theta_{\mathrm{ein}}$ median",
            left_percentile=24.2,
            right_percentile=75.8,
            quantity_name="theta_ein",
            stat_name="median",
        )
        text_labels = {text.get_text() for text in axis.texts}
        assert "L 24.2%" in text_labels
        assert "R 75.8%" in text_labels
        assert axis.get_title() == r"$\theta_{\mathrm{ein}}$ median"
    finally:
        plt.close(figure)


def test_histogram_window_keeps_observed_marker_near_center_without_full_tail_range() -> None:
    """The plotting window should clip extreme tails while keeping the observed line near the center."""

    from cmass_posterior_predictive.predictive import _compute_histogram_x_limits

    values = np.concatenate((np.linspace(0.9, 1.1, 120), np.array([10.0])))
    x_min, x_max = _compute_histogram_x_limits(values=values, observed=1.0)

    assert x_min < 1.0 < x_max
    assert x_max < 5.0
    observed_relative_position = (1.0 - x_min) / (x_max - x_min)
    assert 0.4 <= observed_relative_position <= 0.6


def test_histogram_panel_recomputes_hist_within_display_window() -> None:
    """The histogram bars should be recomputed from the display window, not just clipped by xlim."""

    from cmass_posterior_predictive.predictive import _compute_histogram_x_limits, _write_histogram_panel

    values = np.concatenate((np.linspace(0.9, 1.1, 40), np.array([10.0])))
    expected_x_min, expected_x_max = _compute_histogram_x_limits(values=values, observed=1.0)

    figure, axis = plt.subplots()
    try:
        _write_histogram_panel(
            axis,
            values=values,
            observed=1.0,
            title=r"$\theta_{\mathrm{ein}}$ median",
            left_percentile=7.9,
            right_percentile=92.1,
            quantity_name="theta_ein",
            stat_name="median",
        )

        assert axis.get_xlim() == (expected_x_min, expected_x_max)
        patch_x_positions = [patch.get_x() for patch in axis.patches]
        patch_right_edges = [patch.get_x() + patch.get_width() for patch in axis.patches]
        assert patch_x_positions
        assert min(patch_x_positions) >= expected_x_min - 1.0e-9
        assert max(patch_right_edges) <= expected_x_max + 1.0e-9
    finally:
        plt.close(figure)


def test_theta_ein_std_panel_uses_fixed_histogram_window_and_small_negative_padding() -> None:
    """
    The theta_ein std panel should use the exact fixed window the user asked for.

    The histogram itself must live on the physically meaningful interval
    `[0, 3]`, while the visible axis keeps only a tiny negative padding so the
    first bin does not visually touch the y-axis spine.
    """

    from cmass_posterior_predictive.predictive import _resolve_histogram_ranges, _write_histogram_panel

    values = np.linspace(0.1, 0.8, 80)
    hist_range, display_xlim = _resolve_histogram_ranges(
        values=values,
        observed=0.3,
        quantity_name="theta_ein",
        stat_name="std",
    )

    assert hist_range == (0.0, 3.0)
    assert display_xlim == pytest.approx((-0.075, 3.0))

    figure, axis = plt.subplots()
    try:
        _write_histogram_panel(
            axis,
            values=values,
            observed=0.3,
            title=r"$\theta_{\mathrm{ein}}$ std",
            left_percentile=7.9,
            right_percentile=92.1,
            quantity_name="theta_ein",
            stat_name="std",
        )

        patch_x_positions = [patch.get_x() for patch in axis.patches]
        patch_right_edges = [patch.get_x() + patch.get_width() for patch in axis.patches]
        assert patch_x_positions
        assert min(patch_x_positions) >= 0.0
        assert max(patch_right_edges) <= 3.0 + 1.0e-9
        assert axis.get_xlim() == pytest.approx((-0.075, 3.0))
        assert len(axis.patches) == 24
    finally:
        plt.close(figure)


def test_sigma_std_panel_uses_one_sided_upper_envelope_and_small_negative_padding() -> None:
    """
    The sigma std panel should stop centering on the observed value.

    Its histogram should start at zero and end at a one-sided robust upper
    envelope derived from the replicated sample, with only a small negative
    padding on the visible axis.
    """

    from cmass_posterior_predictive.predictive import _resolve_histogram_ranges

    values = np.concatenate((np.linspace(15.0, 75.0, 60), np.array([110.0, 140.0])))
    expected_upper_bound = 1.03 * max(25.0, float(np.percentile(values, 99.5)))

    hist_range, display_xlim = _resolve_histogram_ranges(
        values=values,
        observed=25.0,
        quantity_name="sigma",
        stat_name="std",
    )

    assert hist_range[0] == 0.0
    assert hist_range[1] == expected_upper_bound
    assert display_xlim == (-0.025 * expected_upper_bound, expected_upper_bound)


def test_only_theta_ein_std_uses_fixed_upper_bound_of_three() -> None:
    """
    The theta_ein upper cap of 3 applies only to the std panel.

    Median, p10, and p90 should keep the existing data-driven window logic so
    the change stays as narrow as the user requested.
    """

    from cmass_posterior_predictive.predictive import (
        _compute_histogram_x_limits,
        _resolve_histogram_ranges,
    )

    values = np.concatenate((np.linspace(0.9, 1.3, 50), np.array([4.2])))
    expected_limits = _compute_histogram_x_limits(values=values, observed=1.0)

    theta_std_hist_range, theta_std_display_xlim = _resolve_histogram_ranges(
        values=values,
        observed=1.0,
        quantity_name="theta_ein",
        stat_name="std",
    )
    theta_median_hist_range, theta_median_display_xlim = _resolve_histogram_ranges(
        values=values,
        observed=1.0,
        quantity_name="theta_ein",
        stat_name="median",
    )

    assert theta_std_hist_range[1] == 3.0
    assert theta_std_display_xlim == pytest.approx((-0.075, 3.0))
    assert theta_median_hist_range == expected_limits
    assert theta_median_display_xlim == expected_limits


def test_overview_figure_uses_profile_suptitle_and_latex_panel_titles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The overview figure should carry a profile suptitle and LaTeX subplot titles."""

    from matplotlib.figure import Figure

    from cmass_posterior_predictive.predictive import _write_overview_figure

    saved: dict[str, object] = {}

    def _capture_savefig(self: Figure, path: Path, *args, **kwargs) -> None:
        saved["figure"] = self
        saved["path"] = Path(path)

    monkeypatch.setattr(Figure, "savefig", _capture_savefig)
    monkeypatch.setattr(plt, "close", lambda figure: None)

    replicated_stats = {
        "median": np.linspace(0.9, 1.1, 32),
        "std": np.linspace(0.2, 0.6, 32),
        "p10": np.linspace(0.7, 0.9, 32),
        "p90": np.linspace(1.1, 1.4, 32),
    }
    summary = {
        "median": {"observed": 1.0, "left_percentile": 50.0, "right_percentile": 50.0},
        "std": {"observed": 0.3, "left_percentile": 7.9, "right_percentile": 92.1},
        "p10": {"observed": 0.8, "left_percentile": 30.0, "right_percentile": 70.0},
        "p90": {"observed": 1.2, "left_percentile": 40.0, "right_percentile": 60.0},
    }

    _write_overview_figure(
        tmp_path / "overview.png",
        profile_name="devauc",
        theta_replicated_stats=replicated_stats,
        theta_summary=summary,
        sigma_replicated_stats=replicated_stats,
        sigma_summary=summary,
    )

    figure = saved["figure"]
    assert saved["path"] == tmp_path / "overview.png"
    assert figure._suptitle is not None
    assert figure._suptitle.get_text() == "Posterior Predictive Check: devauc"
    titles = [axis.get_title() for axis in figure.axes]
    assert titles[:4] == [
        r"$\theta_{\mathrm{ein}}$ median",
        r"$\theta_{\mathrm{ein}}$ std",
        r"$\theta_{\mathrm{ein}}$ p10",
        r"$\theta_{\mathrm{ein}}$ p90",
    ]
    assert titles[4:] == [
        r"$\sigma$ median",
        r"$\sigma$ std",
        r"$\sigma$ p10",
        r"$\sigma$ p90",
    ]
    plt.close(figure)


def test_run_posterior_predictive_supports_hdf5_sigma_tables(tmp_path: Path) -> None:
    """The PPC API should accept the new HDF5 sigma-table schema end-to-end."""

    from cmass_posterior_predictive.predictive import run_posterior_predictive

    run_dir, _ = _build_completed_run(tmp_path, profile_name="devauc")
    sigma_table_path = _write_sigma_table_hdf5(tmp_path / "devauc_sigma_table.h5", profile_name="devauc")

    result = run_posterior_predictive(
        run_dir=str(run_dir),
        sigma_table_path=str(sigma_table_path),
        output_root_dir=str(tmp_path / "ppc_output"),
        n_replicates=3,
        burn_in=1,
        random_seed=19,
        candidate_pool_size=64,
    )

    arrays = np.load(result.result_dir / "replicated_statistics.npz")
    assert np.isfinite(arrays["sigma_sample_sigma"]).all()
    assert arrays["sigma_sample_sigma"].shape == (3, 7)


def test_wait_for_external_sigma_tables_runs_both_profiles_when_overwritten_tables_are_ready(tmp_path: Path) -> None:
    """The monitor entrypoint should wait on fixed paths, then launch both PPC runs."""

    from cmass_posterior_predictive.predictive import wait_for_external_sigma_tables_and_run

    devauc_run_dir, _ = _build_completed_run(tmp_path / "devauc_case", profile_name="devauc")
    sersic_run_dir, _ = _build_completed_run(tmp_path / "sersic_case", profile_name="sersic")
    external_dir = tmp_path / "external"
    external_dir.mkdir(parents=True, exist_ok=True)
    devauc_table_path = _write_external_sigma_table_hdf5(external_dir / "jeans_deV_grid.h5", profile_name="devauc")
    sersic_table_path = _write_external_sigma_table_hdf5(external_dir / "jeans_sers_grid.h5", profile_name="sersic")

    not_before = datetime(2026, 3, 9, 15, 27, 7, tzinfo=timezone(timedelta(hours=8)))
    _touch_with_mtime(devauc_table_path, not_before + timedelta(seconds=5))
    _touch_with_mtime(sersic_table_path, not_before + timedelta(seconds=8))

    result = wait_for_external_sigma_tables_and_run(
        external_dir=str(external_dir),
        output_root_dir=str(tmp_path / "ppc_output"),
        devauc_run_dir=str(devauc_run_dir),
        sersic_run_dir=str(sersic_run_dir),
        not_before=not_before,
        poll_interval_seconds=0.01,
        timeout_seconds=0.1,
        n_replicates=2,
        burn_in=1,
        random_seed=41,
        candidate_pool_size=64,
    )

    assert result.status == "completed"
    assert result.devauc_table_path == devauc_table_path.resolve()
    assert result.sersic_table_path == sersic_table_path.resolve()
    assert result.devauc_result is not None
    assert result.sersic_result is not None
    assert result.devauc_result.result_dir.exists()
    assert result.sersic_result.result_dir.exists()


def test_wait_for_external_sigma_tables_rejects_stale_tables(tmp_path: Path) -> None:
    """The monitor entrypoint should not accept files whose mtime predates the trigger baseline."""

    from cmass_posterior_predictive.predictive import wait_for_external_sigma_tables_and_run

    devauc_run_dir, _ = _build_completed_run(tmp_path / "devauc_case", profile_name="devauc")
    sersic_run_dir, _ = _build_completed_run(tmp_path / "sersic_case", profile_name="sersic")
    external_dir = tmp_path / "external"
    external_dir.mkdir(parents=True, exist_ok=True)
    devauc_table_path = _write_external_sigma_table_hdf5(external_dir / "jeans_deV_grid.h5", profile_name="devauc")
    sersic_table_path = _write_external_sigma_table_hdf5(external_dir / "jeans_sers_grid.h5", profile_name="sersic")

    not_before = datetime(2026, 3, 9, 15, 27, 7, tzinfo=timezone(timedelta(hours=8)))
    _touch_with_mtime(devauc_table_path, not_before - timedelta(seconds=5))
    _touch_with_mtime(sersic_table_path, not_before - timedelta(seconds=5))

    try:
        wait_for_external_sigma_tables_and_run(
            external_dir=str(external_dir),
            output_root_dir=str(tmp_path / "ppc_output"),
            devauc_run_dir=str(devauc_run_dir),
            sersic_run_dir=str(sersic_run_dir),
            not_before=not_before,
            poll_interval_seconds=0.01,
            timeout_seconds=0.05,
            n_replicates=2,
            burn_in=1,
            random_seed=41,
            candidate_pool_size=64,
        )
    except TimeoutError as exc:
        assert "not updated after" in str(exc)
    else:
        raise AssertionError("Expected stale tables to keep the monitor in the waiting state.")


def test_cli_posterior_predictive_command_executes_pipeline(tmp_path: Path) -> None:
    """The CLI should expose a machine-readable `posterior-predictive` command."""

    run_dir, sigma_table_path = _build_completed_run(tmp_path, profile_name="sersic")
    output_root = tmp_path / "cli_output"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cmass_posterior_predictive.cli",
            "posterior-predictive",
            "--run-dir",
            str(run_dir),
            "--sigma-table",
            str(sigma_table_path),
            "--output-dir",
            str(output_root),
            "--n-replicates",
            "3",
            "--burn-in",
            "auto",
            "--seed",
            "31",
            "--candidate-pool-size",
            "72",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["profile_name"] == "sersic"
    assert payload["sample_sizes"]["theta_ein"] == 23
    assert payload["sample_sizes"]["sigma"] == 7
    assert Path(payload["result_dir"]).exists()


def test_posterior_predictive_public_defaults_use_1000_replicates() -> None:
    """The public PPC defaults should use canonical tail-capped full-chain mode."""

    from cmass_posterior_predictive.cli import build_argument_parser
    from cmass_posterior_predictive.predictive import run_posterior_predictive, wait_for_external_sigma_tables_and_run

    parser = build_argument_parser()
    posterior_predictive_args = parser.parse_args(
        [
            "posterior-predictive",
            "--run-dir",
            "/tmp/run",
            "--sigma-table",
            "/tmp/table.npz",
            "--output-dir",
            "/tmp/output",
        ]
    )
    monitor_args = parser.parse_args(
        [
            "posterior-predictive-monitor",
            "--output-dir",
            "/tmp/output",
        ]
    )

    assert posterior_predictive_args.n_replicates is None
    assert monitor_args.n_replicates is None
    assert posterior_predictive_args.worker_processes is None
    assert monitor_args.worker_processes is None
    assert inspect.signature(run_posterior_predictive).parameters["n_replicates"].default is None
    assert inspect.signature(wait_for_external_sigma_tables_and_run).parameters["n_replicates"].default is None
    assert inspect.signature(run_posterior_predictive).parameters["worker_processes"].default is None
    assert inspect.signature(wait_for_external_sigma_tables_and_run).parameters["worker_processes"].default is None


def test_select_posterior_draws_defaults_to_tail_capped_chain() -> None:
    """The default posterior selection should keep the tail of the flattened chain."""

    from cmass_posterior_predictive.predictive import _select_posterior_draws

    flattened_chain = np.arange(40, dtype=float).reshape(10, 4)
    selected, mode = _select_posterior_draws(
        flattened_chain=flattened_chain,
        n_replicates=None,
        rng=np.random.default_rng(11),
        tail_cap=6,
    )

    assert mode == "tail_capped_full_chain"
    assert np.array_equal(selected, flattened_chain[-6:])


def test_resolve_candidate_pool_size_uses_new_100000_cap() -> None:
    """The PPC default candidate-pool cap should now be 100000 instead of 4096."""

    from cmass_posterior_predictive.predictive import _resolve_candidate_pool_size

    assert _resolve_candidate_pool_size(candidate_pool_size=None, base_normals_count=200000) == 100000
    assert _resolve_candidate_pool_size(candidate_pool_size=None, base_normals_count=128) == 128
    assert _resolve_candidate_pool_size(candidate_pool_size=64, base_normals_count=200000) == 64


def test_run_posterior_predictive_parallel_execution_matches_single_process_results(tmp_path: Path) -> None:
    """Changing worker count should not change PPC outputs for the same seed and posterior draws."""

    from cmass_posterior_predictive.predictive import run_posterior_predictive

    run_dir, sigma_table_path = _build_completed_run(tmp_path, profile_name="sersic")
    output_root = tmp_path / "ppc_output"

    serial_result = run_posterior_predictive(
        run_dir=str(run_dir),
        sigma_table_path=str(sigma_table_path),
        output_root_dir=str(output_root / "serial"),
        n_replicates=4,
        burn_in=1,
        random_seed=53,
        candidate_pool_size=64,
        worker_processes=1,
    )
    parallel_result = run_posterior_predictive(
        run_dir=str(run_dir),
        sigma_table_path=str(sigma_table_path),
        output_root_dir=str(output_root / "parallel"),
        n_replicates=4,
        burn_in=1,
        random_seed=53,
        candidate_pool_size=64,
        worker_processes=2,
    )

    serial_arrays = np.load(serial_result.result_dir / "replicated_statistics.npz")
    parallel_arrays = np.load(parallel_result.result_dir / "replicated_statistics.npz")

    assert np.array_equal(serial_arrays["theta_sample_theta_ein"], parallel_arrays["theta_sample_theta_ein"])
    assert np.array_equal(serial_arrays["sigma_sample_sigma"], parallel_arrays["sigma_sample_sigma"])
    assert np.array_equal(serial_arrays["theta_stat_median"], parallel_arrays["theta_stat_median"])
    assert np.array_equal(serial_arrays["sigma_stat_median"], parallel_arrays["sigma_stat_median"])


def test_monitor_defaults_to_tail_capped_common_draw_count(tmp_path: Path) -> None:
    """The monitor should align both profiles to the common tail-capped posterior length by default."""

    from cmass_posterior_predictive.predictive import wait_for_external_sigma_tables_and_run

    devauc_run_dir, _ = _build_completed_run(tmp_path / "devauc_case", profile_name="devauc", n_steps=8)
    sersic_run_dir, _ = _build_completed_run(tmp_path / "sersic_case", profile_name="sersic", n_steps=5)
    external_dir = tmp_path / "external"
    external_dir.mkdir(parents=True, exist_ok=True)
    devauc_table_path = _write_external_sigma_table_hdf5(external_dir / "jeans_deV_grid.h5", profile_name="devauc")
    sersic_table_path = _write_external_sigma_table_hdf5(external_dir / "jeans_sers_grid.h5", profile_name="sersic")

    not_before = datetime(2026, 3, 9, 15, 27, 7, tzinfo=timezone(timedelta(hours=8)))
    _touch_with_mtime(devauc_table_path, not_before + timedelta(seconds=5))
    _touch_with_mtime(sersic_table_path, not_before + timedelta(seconds=8))

    result = wait_for_external_sigma_tables_and_run(
        external_dir=str(external_dir),
        output_root_dir=str(tmp_path / "ppc_output"),
        devauc_run_dir=str(devauc_run_dir),
        sersic_run_dir=str(sersic_run_dir),
        not_before=not_before,
        poll_interval_seconds=0.01,
        timeout_seconds=0.1,
        burn_in=1,
        random_seed=67,
        candidate_pool_size=64,
        worker_processes=1,
    )

    devauc_summary = json.loads((result.devauc_result.result_dir / "ppc_summary.json").read_text(encoding="utf-8"))
    sersic_summary = json.loads((result.sersic_result.result_dir / "ppc_summary.json").read_text(encoding="utf-8"))

    assert devauc_summary["posterior_draw_mode"] == "tail_capped_full_chain"
    assert sersic_summary["posterior_draw_mode"] == "tail_capped_full_chain"
    assert devauc_summary["n_posterior_draws_used"] == 4 * 24
    assert sersic_summary["n_posterior_draws_used"] == 4 * 24


def test_cli_posterior_predictive_monitor_command_waits_and_runs_both_profiles(tmp_path: Path) -> None:
    """The monitor CLI should validate the fixed external tables and run both profiles."""

    devauc_run_dir, _ = _build_completed_run(tmp_path / "devauc_case", profile_name="devauc")
    sersic_run_dir, _ = _build_completed_run(tmp_path / "sersic_case", profile_name="sersic")
    external_dir = tmp_path / "external"
    external_dir.mkdir(parents=True, exist_ok=True)
    devauc_table_path = _write_external_sigma_table_hdf5(external_dir / "jeans_deV_grid.h5", profile_name="devauc")
    sersic_table_path = _write_external_sigma_table_hdf5(external_dir / "jeans_sers_grid.h5", profile_name="sersic")

    not_before = datetime(2026, 3, 9, 15, 27, 7, tzinfo=timezone(timedelta(hours=8)))
    _touch_with_mtime(devauc_table_path, not_before + timedelta(seconds=5))
    _touch_with_mtime(sersic_table_path, not_before + timedelta(seconds=8))

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cmass_posterior_predictive.cli",
            "posterior-predictive-monitor",
            "--external-dir",
            str(external_dir),
            "--output-dir",
            str(tmp_path / "cli_monitor_output"),
            "--devauc-run-dir",
            str(devauc_run_dir),
            "--sersic-run-dir",
            str(sersic_run_dir),
            "--not-before",
            not_before.isoformat(),
            "--poll-interval-seconds",
            "0.01",
            "--timeout-seconds",
            "0.1",
            "--n-replicates",
            "2",
            "--burn-in",
            "1",
            "--seed",
            "73",
            "--candidate-pool-size",
            "64",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    assert payload["devauc_table_path"] == str(devauc_table_path.resolve())
    assert payload["sersic_table_path"] == str(sersic_table_path.resolve())
    assert Path(payload["devauc_result"]["result_dir"]).exists()
    assert Path(payload["sersic_result"]["result_dir"]).exists()


def test_draw_trend_parent_population_samples_full_latent_distribution(tmp_path: Path) -> None:
    """
    Trend generation must now materialize a full parent population.

    This test replaces the old average-size helper contract with the new one:
    the trend workflow should draw `logM*`, `Re`, and, for the sersic branch,
    `n` from the fitted parent-population model rather than freezing them to a
    representative template before averaging.
    """

    from cmass_lens_inference.compiled_context import build_compiled_context
    from cmass_lens_inference.config import load_runtime_config
    from cmass_posterior_predictive.predictive import (
        SigmaUnitTable,
        _draw_trend_parent_population,
        _load_posterior_draws,
    )

    devauc_run_dir, devauc_sigma_table_path = _build_completed_run(tmp_path / "devauc_case", profile_name="devauc")
    sersic_run_dir, sersic_sigma_table_path = _build_completed_run(tmp_path / "sersic_case", profile_name="sersic")

    devauc_runtime_config = load_runtime_config(devauc_run_dir / "config_snapshot.yaml")
    sersic_runtime_config = load_runtime_config(sersic_run_dir / "config_snapshot.yaml")
    devauc_context, devauc_profile, _, _, _, _ = build_compiled_context(devauc_runtime_config)
    sersic_context, sersic_profile, _, _, _, _ = build_compiled_context(sersic_runtime_config)
    devauc_theta_draws, _ = _load_posterior_draws(
        devauc_run_dir / "chain.h5",
        burn_in=2,
        rng=np.random.default_rng(41),
        n_replicates=1,
    )
    sersic_theta_draws, _ = _load_posterior_draws(
        sersic_run_dir / "chain.h5",
        burn_in=2,
        rng=np.random.default_rng(43),
        n_replicates=1,
    )
    devauc_theta = devauc_theta_draws[0]
    sersic_theta = sersic_theta_draws[0]

    devauc_population = _draw_trend_parent_population(
        theta=devauc_theta,
        profile=devauc_profile,
        context=devauc_context,
        sigma_table=SigmaUnitTable.from_path(devauc_sigma_table_path),
        rng=np.random.default_rng(101),
        n_parent_sample=128,
    )
    sersic_population = _draw_trend_parent_population(
        theta=sersic_theta,
        profile=sersic_profile,
        context=sersic_context,
        sigma_table=SigmaUnitTable.from_path(sersic_sigma_table_path),
        rng=np.random.default_rng(103),
        n_parent_sample=128,
    )

    assert np.std(devauc_population["log_mstar"]) > 0.0
    assert np.std(devauc_population["log_re_kpc"]) > 0.0
    assert np.allclose(devauc_population["n"], 4.0)
    assert np.isfinite(devauc_population["sigma_ap"]).any()

    assert np.std(sersic_population["log_mstar"]) > 0.0
    assert np.std(sersic_population["log_re_kpc"]) > 0.0
    assert np.std(sersic_population["n"]) > 0.0
    assert np.isfinite(sersic_population["sigma_ap"]).any()


def test_reduce_population_to_mass_bins_uses_expected_bin_statistics() -> None:
    """
    External-style Fig. 8 trends are defined by bin averages, not pointwise curves.

    This test locks in the exact reducer contract we want:
    - parent: ordinary arithmetic mean within each stellar-mass bin
    - detectable: weighted mean within each bin using geometric weights only
    - selected: weighted mean within each bin using full selection weights
    - empty bins or zero-weight bins: `NaN` for the weighted categories
    """

    from cmass_posterior_predictive.predictive import _reduce_population_to_mass_bins

    log_mstar = np.array([10.20, 10.40, 10.70, 11.20], dtype=float)
    values = np.array([1.0, 3.0, 9.0, 20.0], dtype=float)
    detectable_weights = np.array([1.0, 3.0, 0.0, 2.0], dtype=float)
    selected_weights = np.array([0.5, 1.5, 0.0, 1.0], dtype=float)
    mass_bin_edges = np.array([10.0, 10.5, 11.0, 11.5, 12.0], dtype=float)

    reduced = _reduce_population_to_mass_bins(
        log_mstar=log_mstar,
        values=values,
        mass_bin_edges=mass_bin_edges,
        detectable_weights=detectable_weights,
        selected_weights=selected_weights,
    )

    assert np.allclose(reduced["parent"][:3], np.array([2.0, 9.0, 20.0]))
    assert np.isnan(reduced["parent"][3])
    assert np.isclose(reduced["detectable"][0], np.average([1.0, 3.0], weights=[1.0, 3.0]))
    assert np.isnan(reduced["detectable"][1])
    assert np.isclose(reduced["detectable"][2], 20.0)
    assert np.isnan(reduced["detectable"][3])
    assert np.isclose(reduced["selected"][0], np.average([1.0, 3.0], weights=[0.5, 1.5]))
    assert np.isnan(reduced["selected"][1])
    assert np.isclose(reduced["selected"][2], 20.0)
    assert np.isnan(reduced["selected"][3])
    assert np.array_equal(reduced["parent_bin_counts"], np.array([2, 1, 1, 0]))
    assert np.allclose(reduced["detectable_weight_sums"], np.array([4.0, 0.0, 2.0, 0.0]))
    assert np.allclose(reduced["selected_weight_sums"], np.array([2.0, 0.0, 1.0, 0.0]))


def test_write_trend_panel_uses_distinct_band_solid_and_dashed_encodings() -> None:
    """
    The trend figure should separate the three populations by visual channel.

    The user explicitly asked to stop stacking three translucent bands because
    they are difficult to distinguish. The intended mapping is therefore:
    - parent population: magenta `p16-p84` uncertainty band
    - detectable lenses: black solid `p16` and `p84` boundary lines
    - SLACS-like selected: blue dashed `p16` and `p84` boundary lines
    """

    from cmass_posterior_predictive.predictive import _write_trend_panel

    mass_grid = np.array([11.0, 11.4, 11.8], dtype=float)
    summary = {
        "p16": np.array([1.0, 2.0, 3.0], dtype=float),
        "p50": np.array([1.5, 2.5, 3.5], dtype=float),
        "p84": np.array([2.0, 3.0, 4.0], dtype=float),
    }

    figure, axis = plt.subplots()
    _write_trend_panel(
        axis,
        mass_grid=mass_grid,
        parent_summary=summary,
        detectable_summary=summary,
        selected_summary=summary,
        y_label="demo",
    )

    assert len(axis.collections) == 1
    assert len(axis.lines) == 4
    assert [line.get_linestyle() for line in axis.lines].count("-") == 2
    assert [line.get_linestyle() for line in axis.lines].count("--") == 2
    assert "#d81b60" not in {str(line.get_color()) for line in axis.lines}
    plt.close(figure)


def test_run_posterior_trends_generates_expected_artifacts_for_sersic(tmp_path: Path) -> None:
    """
    The Fig. 8-like API should write figure, summary JSON, and raw curve arrays.

    This guards the main end-to-end contract without trying to lock down
    synthetic scientific numbers that may change with implementation detail.
    """

    from cmass_posterior_predictive.predictive import run_posterior_trends

    run_dir, sigma_table_path = _build_completed_run(tmp_path, profile_name="sersic")
    output_root = tmp_path / "trend_output"

    result = run_posterior_trends(
        run_dir=str(run_dir),
        sigma_table_path=str(sigma_table_path),
        output_root_dir=str(output_root),
        n_posterior_draws=4,
        burn_in="auto",
        random_seed=101,
        n_parent_sample=96,
        n_mass_bins=6,
        mass_bin_min=10.8,
        mass_bin_max=12.0,
        worker_processes=1,
    )

    assert result.profile_name == "sersic"
    assert result.burn_in_applied == 2
    assert result.n_mass_bins == 6
    assert result.result_dir.exists()
    assert (result.result_dir / "fig8_like.png").exists()
    assert (result.result_dir / "fig8_like_summary.json").exists()
    assert (result.result_dir / "fig8_like_curves.npz").exists()

    payload = json.loads((result.result_dir / "fig8_like_summary.json").read_text(encoding="utf-8"))
    assert payload["profile_name"] == "sersic"
    assert payload["requested_n_posterior_draws"] == 4
    assert payload["n_posterior_draws"] == 4
    assert payload["n_posterior_draws_used"] == 4
    assert payload["posterior_draw_mode"] == "sampled_subset"
    assert payload["posterior_draw_tail_cap"] == 192000
    assert payload["parallel_strategy"] == "off"
    assert payload["worker_processes"] == 0
    assert payload["n_parent_sample"] == 96
    assert payload["n_mass_bins"] == 6
    assert len(payload["mass_bin_edges"]) == 7
    assert len(payload["mass_bin_centers"]) == 6
    assert set(payload["quantities"].keys()) == {"m5", "gamma", "sigma_ap"}
    assert set(payload["categories"].keys()) == {"parent", "detectable", "selected"}

    arrays = np.load(result.result_dir / "fig8_like_curves.npz")
    assert arrays["mass_bin_edges"].shape == (7,)
    assert arrays["mass_bin_centers"].shape == (6,)
    assert arrays["parent_m5_draws"].shape == (4, 6)
    assert arrays["detectable_gamma_draws"].shape == (4, 6)
    assert arrays["selected_sigma_ap_draws"].shape == (4, 6)
    assert arrays["parent_bin_counts_draws"].shape == (4, 6)
    assert arrays["detectable_weight_sums_draws"].shape == (4, 6)
    assert arrays["selected_weight_sums_draws"].shape == (4, 6)
    assert np.isfinite(arrays["selected_sigma_ap_draws"]).any()
    assert result.metadata["generator_mode"] == "sampled_population_binned"
    assert result.metadata["posterior_draw_mode"] == "sampled_subset"
    assert result.metadata["posterior_draw_tail_cap"] == 192000
    assert result.metadata["parallel_strategy"] == "off"
    assert result.metadata["worker_processes"] == 0


def test_run_posterior_trends_defaults_to_tail_capped_full_chain_mode(tmp_path: Path) -> None:
    """Omitting `n_posterior_draws` should use the tail-capped full chain by default."""

    from cmass_posterior_predictive.predictive import run_posterior_trends

    run_dir, sigma_table_path = _build_completed_run(tmp_path, profile_name="devauc")
    result = run_posterior_trends(
        run_dir=str(run_dir),
        sigma_table_path=str(sigma_table_path),
        output_root_dir=str(tmp_path / "trend_output"),
        burn_in="auto",
        random_seed=37,
        n_parent_sample=64,
        n_mass_bins=5,
        mass_bin_min=10.9,
        mass_bin_max=11.9,
        worker_processes=1,
    )

    payload = json.loads((result.result_dir / "fig8_like_summary.json").read_text(encoding="utf-8"))
    arrays = np.load(result.result_dir / "fig8_like_curves.npz")

    assert payload["requested_n_posterior_draws"] is None
    assert payload["n_posterior_draws"] == 3 * 24
    assert payload["n_posterior_draws_used"] == 3 * 24
    assert payload["posterior_draw_mode"] == "tail_capped_full_chain"
    assert payload["posterior_draw_tail_cap"] == 192000
    assert payload["parallel_strategy"] == "off"
    assert payload["worker_processes"] == 0
    assert arrays["parent_m5_draws"].shape == (3 * 24, 5)
    assert result.metadata["posterior_draw_mode"] == "tail_capped_full_chain"
    assert result.metadata["requested_n_posterior_draws"] is None


def test_run_posterior_trends_parallel_execution_matches_single_process_results(tmp_path: Path) -> None:
    """Changing trend worker count should not change outputs for the same seed and posterior draws."""

    from cmass_posterior_predictive.predictive import run_posterior_trends

    run_dir, sigma_table_path = _build_completed_run(tmp_path, profile_name="sersic")
    output_root = tmp_path / "trend_parallel_output"

    serial_result = run_posterior_trends(
        run_dir=str(run_dir),
        sigma_table_path=str(sigma_table_path),
        output_root_dir=str(output_root / "serial"),
        n_posterior_draws=4,
        burn_in=1,
        random_seed=73,
        n_parent_sample=64,
        n_mass_bins=5,
        mass_bin_min=10.9,
        mass_bin_max=11.9,
        worker_processes=1,
    )
    parallel_result = run_posterior_trends(
        run_dir=str(run_dir),
        sigma_table_path=str(sigma_table_path),
        output_root_dir=str(output_root / "parallel"),
        n_posterior_draws=4,
        burn_in=1,
        random_seed=73,
        n_parent_sample=64,
        n_mass_bins=5,
        mass_bin_min=10.9,
        mass_bin_max=11.9,
        worker_processes=2,
    )

    serial_arrays = np.load(serial_result.result_dir / "fig8_like_curves.npz")
    parallel_arrays = np.load(parallel_result.result_dir / "fig8_like_curves.npz")

    assert np.array_equal(serial_arrays["parent_m5_draws"], parallel_arrays["parent_m5_draws"], equal_nan=True)
    assert np.array_equal(
        serial_arrays["detectable_gamma_draws"],
        parallel_arrays["detectable_gamma_draws"],
        equal_nan=True,
    )
    assert np.array_equal(
        serial_arrays["selected_sigma_ap_draws"],
        parallel_arrays["selected_sigma_ap_draws"],
        equal_nan=True,
    )
    assert np.array_equal(serial_arrays["parent_bin_counts_draws"], parallel_arrays["parent_bin_counts_draws"])
    assert np.array_equal(
        serial_arrays["detectable_weight_sums_draws"],
        parallel_arrays["detectable_weight_sums_draws"],
        equal_nan=True,
    )
    assert np.array_equal(
        serial_arrays["selected_weight_sums_draws"],
        parallel_arrays["selected_weight_sums_draws"],
        equal_nan=True,
    )


def test_cli_posterior_trends_command_executes_pipeline(tmp_path: Path) -> None:
    """The CLI should expose the new Fig. 8-like trend command."""

    run_dir, sigma_table_path = _build_completed_run(tmp_path, profile_name="devauc")
    output_root = tmp_path / "cli_trend_output"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cmass_posterior_predictive.cli",
            "posterior-trends",
            "--run-dir",
            str(run_dir),
            "--sigma-table",
            str(sigma_table_path),
            "--output-dir",
            str(output_root),
            "--n-posterior-draws",
            "3",
            "--burn-in",
            "1",
            "--seed",
            "113",
            "--n-parent-sample",
            "72",
            "--worker-processes",
            "1",
            "--n-mass-bins",
            "5",
            "--mass-bin-min",
            "10.9",
            "--mass-bin-max",
            "11.9",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["profile_name"] == "devauc"
    assert payload["n_posterior_draws"] == 3
    assert payload["n_mass_bins"] == 5
    assert payload["metadata"]["n_parent_sample"] == 72
    assert payload["metadata"]["worker_processes"] == 0
    assert payload["metadata"]["posterior_draw_mode"] == "sampled_subset"
    assert Path(payload["result_dir"]).exists()


def test_cli_surface_exposes_canonical_trend_defaults() -> None:
    """The trend CLI should default to canonical full-posterior mode and expose worker control."""

    from cmass_posterior_predictive.cli import build_argument_parser
    from cmass_posterior_predictive.trends import run_posterior_trends

    parser = build_argument_parser()
    args = parser.parse_args(
        [
            "posterior-trends",
            "--run-dir",
            "/tmp/run",
            "--sigma-table",
            "/tmp/table.h5",
            "--output-dir",
            "/tmp/out",
        ]
    )

    assert args.n_posterior_draws is None
    assert args.worker_processes is None
    assert inspect.signature(run_posterior_trends).parameters["n_posterior_draws"].default is None
    assert inspect.signature(run_posterior_trends).parameters["worker_processes"].default is None


def test_cli_posterior_trends_rejects_removed_conditional_curve_arguments(tmp_path: Path) -> None:
    """The CLI should fail loudly when callers use the retired mass-grid arguments."""

    run_dir, sigma_table_path = _build_completed_run(tmp_path, profile_name="devauc")
    output_root = tmp_path / "cli_trend_output"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cmass_posterior_predictive.cli",
            "posterior-trends",
            "--run-dir",
            str(run_dir),
            "--sigma-table",
            str(sigma_table_path),
            "--output-dir",
            str(output_root),
            "--n-mass-grid",
            "5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "bin-based" in completed.stderr
