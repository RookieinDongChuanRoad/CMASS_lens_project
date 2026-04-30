"""
Regression tests for the JAX/NumPyro inference migration.

These tests intentionally describe the new production contract before the
implementation exists.  The old numba/emcee path remains useful as a numerical
oracle during migration, but the public run path must now produce NumPyro-style
posterior artifacts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.jax_model import build_jax_model, log_prob as jax_log_prob
from cmass_lens_inference.model import build_compiled_model, log_prob as legacy_log_prob
from cmass_lens_inference.runner import run_inference


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
    assert runtime_config.sampling.num_chains == 1
    assert runtime_config.sampling.num_samples == runtime_config.sampling.n_steps
    assert runtime_config.sampling.num_warmup == runtime_config.sampling.warmup
    assert runtime_config.sampling.thinning == 1
    assert runtime_config.sampling.chain_method == "sequential"


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
            "num_chains": 1,
            "num_samples": 2,
            "num_warmup": 1,
            "chain_method": "sequential",
            "thinning": 1,
        }
    )
    numpyro_config_path = synthetic_config_path.parent / "synthetic_numpyro.yaml"
    numpyro_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    run_result = run_inference(str(numpyro_config_path))

    assert run_result.status == "completed"
    assert run_result.completed_steps == 2
    assert run_result.acceptance_fraction_mean >= 0.0
    assert run_result.metadata["chain_storage"] == "numpyro_arviz_netcdf"
    assert (run_result.run_dir / "samples.npz").exists()
    assert (run_result.run_dir / "posterior.nc").exists()
    assert not (run_result.run_dir / "chain.h5").exists()
