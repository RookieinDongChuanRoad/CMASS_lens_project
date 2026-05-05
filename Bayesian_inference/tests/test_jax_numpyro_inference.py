"""
Regression tests for the JAX/NumPyro inference backend.

The legacy numba/emcee backend has been removed from the production package, so
these tests now lock the active JAX contract directly: model-registry dispatch,
finite posterior evaluation, hunit-aware context values, and NumPyro artifacts.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.jax_backend.likelihood_engine import (
    build_compiled_model as build_jax_model,
    log_prob as jax_log_prob,
)
from cmass_lens_inference.numpyro_sampler import _build_jittered_initial_strategy
from cmass_lens_inference.runner import run_inference


def _write_minimal_canonical_dataset(path: Path) -> Path:
    """
    Write a tiny canonical dataset that can drive the CMASS JAX runtime.

    This fixture intentionally bypasses legacy observation/cross-section files:
    the test asserts that inference can start from the canonical schema itself.
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


def test_config_loads_cmass_model_and_numpyro_sampling_fields(
    synthetic_config_path: Path,
) -> None:
    """
    New configs should expose one concrete model and NumPyro sampling fields.
    """

    runtime_config = load_runtime_config(synthetic_config_path)

    assert runtime_config.model.name == "cmass"
    assert not hasattr(runtime_config.model, "components")
    assert runtime_config.parameter_schema.model_metadata["gamma_distribution"] == "sigma_star_dependent"
    assert runtime_config.mass_definition.label == "m5_hinvkpc"
    assert runtime_config.sampling.num_chains == 24
    assert runtime_config.sampling.num_samples == 3
    assert runtime_config.sampling.num_warmup == runtime_config.sampling.warmup
    assert runtime_config.sampling.thinning == 1
    assert runtime_config.sampling.chain_method == "sequential"


def test_numpyro_jittered_initial_strategy_uses_independent_bounded_chain_values(
    synthetic_config_path: Path,
) -> None:
    """
    NumPyro chains should start from independently jittered, in-bounds values.

    This replaces the old emcee walker cloud condition-number concern with the
    NUTS-specific contract: each chain gets its own valid constrained initial
    point, while box-prior bounds remain hard guards.
    """

    import jax

    runtime_config = load_runtime_config(synthetic_config_path)
    strategy = _build_jittered_initial_strategy(runtime_config)
    parameter_name = runtime_config.parameter_schema.internal_parameter_names[0]
    lower, upper = runtime_config.parameter_schema.prior_bounds[0]

    site_template = {
        "type": "sample",
        "is_observed": False,
        "name": parameter_name,
        "kwargs": {},
    }
    first_site = {**site_template, "kwargs": {"rng_key": jax.random.PRNGKey(1)}}
    second_site = {**site_template, "kwargs": {"rng_key": jax.random.PRNGKey(2)}}

    first_value = float(strategy(first_site))
    second_value = float(strategy(second_site))

    assert lower < first_value < upper
    assert lower < second_value < upper
    assert first_value != pytest.approx(second_value)


def test_jax_log_prob_is_finite_for_cmass_model(
    synthetic_config_path: Path,
) -> None:
    """
    A valid initial point should produce a finite posterior through backend dispatch.
    """

    runtime_config = load_runtime_config(synthetic_config_path)
    theta = runtime_config.sampling.initial_center.to_array()

    jax_model = build_jax_model(runtime_config)

    jax_value, blob = jax_log_prob(theta, jax_model)

    assert np.isfinite(jax_value)
    assert blob["backend"].decode("utf-8").rstrip("\x00") == "jax"
    assert np.isfinite(float(blob["normalization_value"]))


def test_jax_log_prob_is_finite_for_canonical_dataset(
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
    jax_model = build_jax_model(runtime_config)

    jax_value, blob = jax_log_prob(theta, jax_model)

    assert runtime_config.data.inference_dataset_path is not None
    assert jax_model.context.cs_cross_section_grid.shape == (64, 17)
    assert np.isfinite(jax_value)
    assert np.isfinite(float(blob["normalization_value"]))


def test_jax_fp_prior_log_prob_is_finite(
    synthetic_fp_prior_config_path: Path,
) -> None:
    """
    FP-enabled JAX evaluation should expose finite FP diagnostics.

    This specifically guards the population-summary path because it estimates
    both selection normalization and sufficient statistics for the Fundamental
    Plane prior.
    """

    runtime_config = load_runtime_config(synthetic_fp_prior_config_path)
    theta = runtime_config.sampling.initial_center.to_array()

    jax_model = build_jax_model(runtime_config)

    jax_value, blob = jax_log_prob(theta, jax_model)

    assert np.isfinite(jax_value)
    assert np.isfinite(float(blob["fp_prior_log_term"]))
    assert np.isfinite(float(blob["fpfit_mu"]))
    assert np.isfinite(float(blob["fpfit_beta"]))


def test_jax_log_prob_uses_h_unit_current_model_context(
    synthetic_config_path: Path,
) -> None:
    """
    The JAX backend must honor the merged hunit mass and size convention.

    This is the regression that protects the non-textual merge risk: Git can
    merge `compiled_context.py` automatically, but JAX still has to consume the
    hunit-aware `stellar_mass_pivot`, physical aperture radius, and mass-log
    offset exported by that context.
    """

    runtime_config = load_runtime_config(synthetic_config_path)
    theta = runtime_config.sampling.initial_center.to_array()

    jax_model = build_jax_model(runtime_config)

    jax_value, blob = jax_log_prob(theta, jax_model)

    assert runtime_config.mass_definition.label == "m5_hinvkpc"
    assert jax_model.context.stellar_mass_pivot == pytest.approx(11.4 + 2.0 * np.log10(0.7))
    assert jax_model.context.mass_radius_kpc == pytest.approx(5.0 / 0.7)
    assert jax_model.context.mass_log_physical_offset == pytest.approx(np.log10(0.7**-1))
    assert np.isfinite(jax_value)
    assert blob["backend"].decode("utf-8").rstrip("\x00") == "jax"


def test_run_inference_writes_numpyro_posterior_artifacts(
    synthetic_config_path: Path,
) -> None:
    """
    Production runs should use NumPyro output artifacts instead of emcee HDF5.

    The synthetic run is deliberately tiny: it verifies orchestration,
    serialization, and metadata contracts without trying to assess MCMC quality.
    """

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    payload["sampling"].update(
        {
            "num_chains": 2,
            "num_samples": 2,
            "num_warmup": 1,
            "chain_method": "sequential",
            "thinning": 1,
            "initial_jitter_scale": 1.0e-3,
        }
    )
    numpyro_config_path = synthetic_config_path.parent / "synthetic_numpyro.yaml"
    numpyro_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    runtime_config = load_runtime_config(numpyro_config_path)
    run_result = run_inference(str(numpyro_config_path))

    assert run_result.status == "completed"
    assert run_result.completed_steps == 2
    assert run_result.acceptance_fraction_mean >= 0.0
    assert run_result.metadata["chain_storage"] == "numpyro_arviz_netcdf"
    assert (run_result.run_dir / "samples.npz").exists()
    assert (run_result.run_dir / "posterior.nc").exists()
    assert not (run_result.run_dir / "chain.h5").exists()
    with np.load(run_result.run_dir / "samples.npz") as payload:
        assert payload["samples_by_chain"].shape == (2, 2, runtime_config.parameter_schema.n_dim)
