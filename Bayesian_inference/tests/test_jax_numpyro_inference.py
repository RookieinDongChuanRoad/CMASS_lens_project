"""
Regression tests for the JAX/NumPyro inference migration.

These tests intentionally describe the new production contract before the
implementation exists.  The old numba/emcee path remains useful as a numerical
oracle during migration, but the public run path must now produce NumPyro-style
posterior artifacts.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.jax_model import build_jax_model, log_prob as jax_log_prob
from cmass_lens_inference.model import build_compiled_model, log_prob as legacy_log_prob
from cmass_lens_inference.numpyro_sampler import _build_jittered_initial_strategy
from cmass_lens_inference.runner import run_inference


def _write_h_unit_observation_file(path: Path) -> Path:
    """
    Write the minimal h-units observation schema needed by the JAX regression.

    The fixture is intentionally local to this test module because it is not a
    generic data-builder.  It exists to exercise the exact boundary between the
    merged hunit I/O contract and the JAX posterior: h-unit stellar mass,
    h-unit size, and the `m5_hinvkpc` mass-definition subgroup.
    """

    gamma_grid = np.linspace(1.3, 2.7, 17)
    with h5py.File(path, "w") as handle:
        handle.attrs["unit_convention"] = "h_units_v1"
        handle.attrs["h_ref"] = 0.7

        group = handle.create_group("lens-h")
        group.attrs["unit_convention"] = "h_units_v1"
        group.attrs["h_ref"] = 0.7
        group.attrs["zd"] = 0.55
        group.attrs["zs"] = 1.75
        group.attrs["logmchab_h2"] = 10.99
        group.attrs["logmchab_err"] = 0.05
        group.attrs["log10_re_hinv_kpc"] = 0.64
        group.attrs["nser"] = 4.2
        group.attrs["rein_arcsec"] = 1.3
        group.attrs["num_sigma"] = 0
        group.create_dataset("gamma_grid", data=gamma_grid)

        mass_group = group.create_group("mass_definitions").create_group("m5_hinvkpc")
        mass_group.attrs["unit_convention"] = "h_units_v1"
        mass_group.attrs["h_ref"] = 0.7
        mass_group.create_dataset("mass_grid", data=np.linspace(11.7, 10.9, 17))
        mass_group.create_dataset("dmass_dthetaein_grid", data=np.linspace(-2.0, -1.0, 17))

    return path


def test_config_loads_model_components_and_numpyro_sampling_aliases(
    synthetic_config_path: Path,
) -> None:
    """
    Existing configs should migrate to the default component model implicitly.

    Why this matters:
    - real user configs should not break only because the backend moved from
      emcee to NumPyro
    - the new component model surface must be visible on RuntimeConfig so
      switching to the Sonnenfeld 2024 model can be done through YAML alone
    - NumPyro uses chain/sample/warmup terminology, but legacy fixture configs
      still carry walker/step/warmup fields
    """

    runtime_config = load_runtime_config(synthetic_config_path)

    assert runtime_config.model.name == "cmass_current"
    assert runtime_config.model.components == {
        "foreground_population": "cmass_fixed_z_skew_mstar",
        "size_relation": "profile_default",
        "mass_gamma_distribution": "current_power_law",
        "source_redshift": "truncated_normal",
        "selection": "theta_sigmoid_cross_section",
        "velocity_dispersion": "grid_sigma_unit",
        "fp_prior": "optional_ols_summary",
    }
    assert runtime_config.sampling.num_chains == 2 * runtime_config.parameter_schema.n_dim
    assert runtime_config.sampling.num_samples == runtime_config.sampling.n_steps
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


@pytest.mark.parametrize(
    "fixture_name",
    [
        "synthetic_config_path",
        "synthetic_independent_config_path",
        "synthetic_sigma_star_dependent_config_path",
    ],
)
def test_jax_log_prob_matches_legacy_kernel_for_current_model(
    request: pytest.FixtureRequest,
    fixture_name: str,
) -> None:
    """
    The first migration milestone is numerical equivalence for current models.

    The legacy numba implementation is used here only as a reference oracle.
    The new JAX path must return the same posterior value for all supported
    gamma parameterizations before NumPyro sampling is trusted.
    """

    runtime_config = load_runtime_config(request.getfixturevalue(fixture_name))
    theta = runtime_config.sampling.initial_center.to_array()

    legacy_model = build_compiled_model(runtime_config)
    jax_model = build_jax_model(runtime_config)

    legacy_value, _ = legacy_log_prob(theta, legacy_model)
    jax_value, blob = jax_log_prob(theta, jax_model)

    assert np.isfinite(jax_value)
    assert jax_value == pytest.approx(legacy_value, rel=1.0e-8, abs=1.0e-8)
    assert blob["backend"].decode("utf-8").rstrip("\x00") == "jax"


def test_jax_fp_prior_log_prob_matches_legacy_kernel(
    synthetic_fp_prior_config_path: Path,
) -> None:
    """
    The JAX migration must preserve the optional FP-prior posterior term.

    This specifically guards the population-summary path, which is the most
    complex part of the old normalization kernel because it simultaneously
    estimates the selection normalization and sufficient statistics for the
    Fundamental Plane prior.
    """

    runtime_config = load_runtime_config(synthetic_fp_prior_config_path)
    theta = runtime_config.sampling.initial_center.to_array()

    legacy_model = build_compiled_model(runtime_config)
    jax_model = build_jax_model(runtime_config)

    legacy_value, _ = legacy_log_prob(theta, legacy_model)
    jax_value, _ = jax_log_prob(theta, jax_model)

    assert np.isfinite(jax_value)
    assert jax_value == pytest.approx(legacy_value, rel=1.0e-8, abs=1.0e-8)


def test_jax_log_prob_matches_legacy_kernel_for_h_unit_current_model(
    tmp_path: Path,
    synthetic_config_path: Path,
) -> None:
    """
    The JAX backend must honor the merged hunit mass and size convention.

    This is the regression that protects the non-textual merge risk: Git can
    merge `compiled_context.py` automatically, but JAX still has to consume the
    hunit-aware `stellar_mass_pivot`, physical aperture radius, and mass-log
    offset exported by that context.
    """

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    payload["unit_convention"] = "h_units_v1"
    payload["data"]["observation_path"] = str(_write_h_unit_observation_file(tmp_path / "h_units_observations.hdf5"))
    payload["mass_definition"] = {"aperture_hinv_kpc": 5}
    payload["box_prior"] = {
        "mu5h_0": [9.0, 12.0],
        "beta5h": [-3.0, 3.0],
        "xi5h": [-3.0, 3.0],
        "sigma5h": [1.0e-2, 0.2],
        "mu_gamma_0": [1.5, 2.5],
        "beta_gamma": [-3.0, 3.0],
        "xi_gamma": [-3.0, 3.0],
        "sigma_gamma": [0.0, 0.5],
        "mu_zs": [1.0, 3.0],
        "sigma_zs": [0.0, 2.0],
        "theta0": [0.0, 3.0],
        "loga": [-1.0, 3.0],
    }
    payload["sampling"]["initial_center"] = {
        "mu5h_0": 11.17,
        "beta5h": 0.59,
        "xi5h": -0.11,
        "sigma5h": 0.06,
        "mu_gamma_0": 1.99,
        "beta_gamma": 0.10,
        "xi_gamma": -0.67,
        "sigma_gamma": 0.149,
        "mu_zs": 1.8,
        "sigma_zs": 0.215,
        "theta0": 0.93,
        "loga": 1.0,
    }

    h_unit_config_path = tmp_path / "h_unit_jax.yaml"
    h_unit_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    runtime_config = load_runtime_config(h_unit_config_path)
    theta = runtime_config.sampling.initial_center.to_array()

    legacy_model = build_compiled_model(runtime_config)
    jax_model = build_jax_model(runtime_config)

    legacy_value, _ = legacy_log_prob(theta, legacy_model)
    jax_value, _ = jax_log_prob(theta, jax_model)

    assert runtime_config.mass_definition.label == "m5_hinvkpc"
    assert jax_model.context.stellar_mass_pivot == pytest.approx(11.4 + 2.0 * np.log10(0.7))
    assert jax_model.context.mass_radius_kpc == pytest.approx(5.0 / 0.7)
    assert jax_model.context.mass_log_physical_offset == pytest.approx(np.log10(0.7**-1))
    assert np.isfinite(jax_value)
    assert jax_value == pytest.approx(legacy_value, rel=1.0e-8, abs=1.0e-8)


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
