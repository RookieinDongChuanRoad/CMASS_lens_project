"""Architecture acceptance tests for adding a production model.

The toy model is intentionally synthetic.  These tests verify the extension
boundary promised by the component refactor: adding a model requires model
files, registration, and tests, but not changes to runner, sampler, output
writer, or posterior-reader modules.
"""

from __future__ import annotations

from pathlib import Path

import emcee
import numpy as np
import yaml

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.model_registry import get_model_definition
from cmass_lens_inference.numba_backend.likelihood_engine import (
    build_compiled_model,
    log_prob,
)
from cmass_lens_inference.runner import run_inference


def _toy_config_payload(tmp_path: Path) -> dict:
    """Return a minimal config payload for the toy production model."""

    return {
        "profile": {"name": "toy"},
        "unit_convention": "h_units_v1",
        "model": {"name": "toy_hierarchical"},
        "data": {"inference_dataset_path": str(tmp_path / "unused_toy_context.hdf5")},
        "box_prior": {
            "population_mean": [-5.0, 5.0],
            "log_population_scatter": [-5.0, 1.0],
        },
        "sampling": {
            "random_seed": 11,
            "n_walkers": 8,
            "n_steps": 2,
            "burn_in": 0,
            "initial_center": {
                "population_mean": 0.1,
                "log_population_scatter": -1.2,
            },
            "initial_jitter_scale": 1.0e-3,
        },
        "integration": {
            "gamma_points": 1,
            "mstar_points": 1,
            "normalization_samples": 4,
        },
        "cosmology": {"h0": 70.0, "omega_m": 0.3},
        "runtime": {
            "checkpoint_every": 1,
            "parallel_strategy": "off",
            "progress": False,
            "progress_summary_every": 1,
            "show_stage_timing": False,
            "disable_hdf5_file_locking": False,
            "num_threads": 1,
            "reserve_cores": 0,
        },
        "output": {
            "root_dir": str(tmp_path / "outputs"),
            "run_label": "toy-extension",
            "overwrite_latest": True,
        },
    }


def _write_toy_config(tmp_path: Path) -> Path:
    """Write the toy config and return its path."""

    config_path = tmp_path / "toy_hierarchical.yaml"
    config_path.write_text(
        yaml.safe_dump(_toy_config_payload(tmp_path), sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def test_toy_model_registry_entry_owns_posterior_callable() -> None:
    """Registry should bind toy assembly, runtime, and posterior adapter."""

    model_definition = get_model_definition("toy_hierarchical")

    assert model_definition.name == "toy_hierarchical"
    assert model_definition.backend_kernel == "toy_hierarchical"
    assert callable(model_definition.evaluate_log_prob)
    assert model_definition.required_capabilities == ()


def test_toy_model_evaluates_finite_log_prob_and_host_rejection(tmp_path: Path) -> None:
    """The generic likelihood engine should call the toy model-owned adapter."""

    runtime_config = load_runtime_config(_write_toy_config(tmp_path))
    compiled_model = build_compiled_model(runtime_config)
    theta = runtime_config.sampling.initial_center.to_array()

    value, blob = log_prob(theta, compiled_model)

    assert np.isfinite(value)
    assert blob["backend"].decode("utf-8").rstrip("\x00") == "numba"
    assert blob["kernel"].decode("utf-8").rstrip("\x00") == "toy_hierarchical"
    assert float(blob["normalization_value"]) == 1.0

    rejected_theta = theta.copy()
    rejected_theta[0] = runtime_config.parameter_schema.prior_bounds[0][1] + 1.0
    rejected_value, rejected_blob = log_prob(rejected_theta, compiled_model)

    assert rejected_value == -np.inf
    assert rejected_blob["kernel"].decode("utf-8").rstrip("\x00") == "toy_hierarchical"


def test_toy_model_short_emcee_run_writes_chain_without_framework_changes(tmp_path: Path) -> None:
    """A new model should run through the existing emcee/HDFBackend pipeline."""

    config_path = _write_toy_config(tmp_path)
    runtime_config = load_runtime_config(config_path)

    run_result = run_inference(str(config_path))

    assert run_result.status == "completed"
    assert run_result.completed_steps == 2
    assert run_result.metadata["model"]["name"] == "toy_hierarchical"
    assert run_result.metadata["chain_storage"] == "emcee_hdf_backend"
    assert (run_result.run_dir / "chain.h5").exists()

    backend = emcee.backends.HDFBackend(str(run_result.run_dir / "chain.h5"), read_only=True)
    assert backend.get_chain().shape == (2, 8, runtime_config.parameter_schema.n_dim)
