"""
Regression tests for the production Numba/emcee inference backend.

These tests lock the backend contract that matters for Phase A and Phase B:
emcee-native config fields, bounded walker initialization, finite CMASS Numba
log-probability evaluation, canonical dataset context construction, and native
`chain.h5` artifacts.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.emcee_sampler import initialize_walkers
from cmass_lens_inference.numba_backend.likelihood_engine import (
    build_compiled_model as build_numba_model,
    log_prob as numba_log_prob,
)
from cmass_lens_inference.runner import run_inference


def _write_minimal_canonical_dataset(path: Path) -> Path:
    """
    Write a tiny canonical CMASS dataset for backend-level tests.

    The fixture bypasses legacy raw observation and cross-section products so
    the tests assert the production runtime starts from the canonical schema.
    """

    gamma_grid = np.linspace(1.3, 2.7, 17)
    theta_e_axis = np.linspace(0.0, 5.0, 64)
    cs_over_theta = np.linspace(0.6, 1.4, gamma_grid.size)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.attrs["schema_version"] = "canonical_inference_dataset_v1"
        metadata.attrs["unit_convention"] = "h_units_v1"
        metadata.attrs["h_ref"] = 0.7
        metadata.attrs["profile_name"] = "sersic"
        metadata.attrs["mass_definition_label"] = "m5_hinvkpc"
        metadata.attrs["mass_radius_kpc"] = 5.0
        metadata.attrs["cosmology_h0"] = 70.0
        metadata.attrs["cosmology_omega_m"] = 0.3
        metadata.create_dataset(
            "capabilities",
            data=np.asarray(
                [
                    "lens_observations.v1",
                    "lensing_mass_grids.v1",
                    "lensing_cross_section.theta_gamma_grid.v1",
                    "velocity_dispersion.per_lens_s2.v1",
                ],
                dtype=object,
            ),
            dtype=string_dtype,
        )

        lenses = handle.create_group("lenses")
        lenses.create_dataset("lens_id", data=np.asarray(["lens-canonical"], dtype=object), dtype=string_dtype)
        lenses.create_dataset("z_d", data=np.asarray([0.55], dtype=float))
        lenses.create_dataset("z_s", data=np.asarray([1.75], dtype=float))
        lenses.create_dataset("log_mstar_obs", data=np.asarray([10.99], dtype=float))
        lenses.create_dataset("log_mstar_err", data=np.asarray([0.05], dtype=float))
        lenses.create_dataset("log_re_obs", data=np.asarray([0.64], dtype=float))
        lenses.create_dataset("n_obs", data=np.asarray([4.2], dtype=float))
        lenses.create_dataset("theta_e_obs", data=np.asarray([1.3], dtype=float))
        lenses.create_dataset("num_sigma", data=np.asarray([1], dtype=np.int64))
        lenses.create_dataset("sigma_obs", data=np.asarray([[320000.0, 0.0]], dtype=float))
        lenses.create_dataset("sigma_err", data=np.asarray([[20000.0, 1.0]], dtype=float))

        mass_grids = handle.create_group("lensing_mass_grids")
        mass_grids.create_dataset("gamma_grid", data=gamma_grid[None, :])
        mass_grids.create_dataset("log_enclosed_mass_grid", data=np.linspace(11.7, 10.9, 17)[None, :])
        mass_grids.create_dataset("dmass_dthetaein_grid", data=np.linspace(-2.0, -1.0, 17)[None, :])
        mass_grids.create_dataset("s2_grid", data=np.linspace(0.8, 1.2, 17)[None, :])
        mass_grids.create_dataset("has_s2", data=np.asarray([1], dtype=np.int64))

        cross_section = handle.create_group("lensing_cross_section")
        cross_section.create_dataset("theta_e_axis", data=theta_e_axis)
        cross_section.create_dataset("gamma_axis", data=gamma_grid)
        cross_section.create_dataset(
            "cross_section_grid",
            data=np.pi * (theta_e_axis[:, None] * cs_over_theta[None, :]) ** 2,
        )
        cross_section.attrs["boundary_policy"] = "zero_outside_theta_clip_gamma"

        velocity = handle.create_group("velocity_dispersion_grids")
        per_lens_s2 = velocity.create_group("per_lens_s2")
        per_lens_s2.create_dataset("s2_grid", data=np.linspace(0.8, 1.2, 17)[None, :])
        per_lens_s2.create_dataset("has_s2", data=np.asarray([1], dtype=np.int64))

    return path


def test_config_loads_cmass_model_and_emcee_sampling_fields(
    synthetic_config_path: Path,
) -> None:
    """New configs should expose one concrete model and emcee sampling fields."""

    runtime_config = load_runtime_config(synthetic_config_path)

    assert runtime_config.model.name == "cmass"
    assert not hasattr(runtime_config.model, "components")
    assert runtime_config.parameter_schema.model_metadata["gamma_distribution"] == "sigma_star_dependent"
    assert runtime_config.mass_definition.label == "m5_hinvkpc"
    assert runtime_config.sampling.n_walkers == 24
    assert runtime_config.sampling.n_steps == 3
    assert runtime_config.sampling.burn_in == runtime_config.sampling.warmup == 1


def test_emcee_walker_initialization_uses_independent_bounded_values(
    synthetic_config_path: Path,
) -> None:
    """emcee walkers should start from independently jittered in-bounds values."""

    runtime_config = load_runtime_config(synthetic_config_path)
    walkers = initialize_walkers(
        runtime_config.sampling.initial_center,
        runtime_config.sampling.n_walkers,
        runtime_config.sampling.initial_jitter_scale,
        runtime_config.sampling.random_seed,
    )
    lower_bounds = np.asarray([lower for lower, _ in runtime_config.parameter_schema.prior_bounds])
    upper_bounds = np.asarray([upper for _, upper in runtime_config.parameter_schema.prior_bounds])

    assert walkers.shape == (24, runtime_config.parameter_schema.n_dim)
    assert np.all(walkers >= lower_bounds[None, :])
    assert np.all(walkers <= upper_bounds[None, :])
    assert np.linalg.matrix_rank(walkers - walkers.mean(axis=0)) > 1


def test_numba_log_prob_is_finite_for_cmass_model(
    synthetic_config_path: Path,
) -> None:
    """A valid initial point should produce a finite Numba posterior."""

    runtime_config = load_runtime_config(synthetic_config_path)
    theta = runtime_config.sampling.initial_center.to_array()
    numba_model = build_numba_model(runtime_config)

    value, blob = numba_log_prob(theta, numba_model)

    assert np.isfinite(value)
    assert blob["backend"].decode("utf-8").rstrip("\x00") == "numba"
    assert blob["kernel"].decode("utf-8").rstrip("\x00") == "cmass"
    assert np.isfinite(float(blob["normalization_value"]))


def test_numba_log_prob_rejects_out_of_bounds_theta(
    synthetic_config_path: Path,
) -> None:
    """The host-side box prior should reject proposals before kernel work."""

    runtime_config = load_runtime_config(synthetic_config_path)
    theta = runtime_config.sampling.initial_center.to_array()
    theta[0] = runtime_config.parameter_schema.prior_bounds[0][1] + 1.0
    numba_model = build_numba_model(runtime_config)

    value, blob = numba_log_prob(theta, numba_model)

    assert value == -np.inf
    assert float(blob["normalization_value"]) == 0.0


def test_numba_log_prob_is_finite_for_canonical_dataset(
    tmp_path: Path,
    synthetic_config_path: Path,
) -> None:
    """The CMASS runtime should be able to start from canonical dataset input."""

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    payload["data"] = {
        "inference_dataset_path": str(_write_minimal_canonical_dataset(tmp_path / "canonical.hdf5")),
    }
    canonical_config_path = tmp_path / "canonical_config.yaml"
    canonical_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    runtime_config = load_runtime_config(canonical_config_path)
    theta = runtime_config.sampling.initial_center.to_array()
    numba_model = build_numba_model(runtime_config)

    value, blob = numba_log_prob(theta, numba_model)

    assert runtime_config.data.inference_dataset_path is not None
    assert numba_model.context.cs_cross_section_grid.shape == (64, 17)
    assert np.isfinite(value)
    assert np.isfinite(float(blob["normalization_value"]))


def test_numba_fp_prior_log_prob_is_finite(
    synthetic_fp_prior_config_path: Path,
) -> None:
    """FP-enabled Numba evaluation should expose finite FP diagnostics."""

    runtime_config = load_runtime_config(synthetic_fp_prior_config_path)
    theta = runtime_config.sampling.initial_center.to_array()
    numba_model = build_numba_model(runtime_config)

    value, blob = numba_log_prob(theta, numba_model)

    assert np.isfinite(value)
    assert np.isfinite(float(blob["fp_prior_log_term"]))
    assert np.isfinite(float(blob["fpfit_mu"]))
    assert np.isfinite(float(blob["fpfit_beta"]))


def test_numba_log_prob_uses_h_unit_current_model_context(
    synthetic_config_path: Path,
) -> None:
    """The Numba backend must honor the hunit mass and size convention."""

    runtime_config = load_runtime_config(synthetic_config_path)
    theta = runtime_config.sampling.initial_center.to_array()
    numba_model = build_numba_model(runtime_config)

    value, blob = numba_log_prob(theta, numba_model)

    assert runtime_config.mass_definition.label == "m5_hinvkpc"
    assert numba_model.context.stellar_mass_pivot == pytest.approx(11.4 + 2.0 * np.log10(0.7))
    assert numba_model.context.mass_radius_kpc == pytest.approx(5.0 / 0.7)
    assert numba_model.context.mass_log_physical_offset == pytest.approx(np.log10(0.7**-1))
    assert np.isfinite(value)
    assert blob["backend"].decode("utf-8").rstrip("\x00") == "numba"


def test_run_inference_writes_emcee_chain_artifact(
    synthetic_config_path: Path,
) -> None:
    """Production runs should use emcee HDFBackend output artifacts."""

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    payload["sampling"].update(
        {
            "n_walkers": 24,
            "n_steps": 2,
            "burn_in": 1,
            "initial_jitter_scale": 1.0e-3,
        }
    )
    config_path = synthetic_config_path.parent / "synthetic_emcee.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    runtime_config = load_runtime_config(config_path)
    run_result = run_inference(str(config_path))

    assert run_result.status == "completed"
    assert run_result.completed_steps == 2
    assert run_result.acceptance_fraction_mean >= 0.0
    assert run_result.metadata["chain_storage"] == "emcee_hdf_backend"
    assert (run_result.run_dir / "chain.h5").exists()
    assert not (run_result.run_dir / "samples.npz").exists()
    assert not (run_result.run_dir / "posterior.nc").exists()

    import emcee

    backend = emcee.backends.HDFBackend(str(run_result.run_dir / "chain.h5"), read_only=True)
    assert backend.get_chain().shape == (2, 24, runtime_config.parameter_schema.n_dim)
