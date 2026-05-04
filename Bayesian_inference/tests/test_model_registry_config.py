"""
Tests for the model-registry configuration contract.

These tests intentionally describe the breaking-change surface requested for
the JAX/NumPyro refactor: model-specific choices live under
``model.components`` and are resolved through the model registry.  The old
top-level ``gamma_model`` / ``mass_definition`` config surface is no longer a
supported input contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.mass_definition import LEGACY_FIXED_KPC, get_mass_definition
from cmass_lens_inference.model_registry import get_model_definition


def _minimal_new_model_config(tmp_path: Path) -> dict:
    """
    Build a small config payload that exercises parsing without touching HDF5.

    The observation and cross-section files do not need to exist because
    ``load_runtime_config`` only validates and normalizes config syntax.  This
    keeps the model-registry tests focused on the public YAML contract instead
    of coupling them to I/O fixtures.
    """

    return {
        "profile": {"name": "sersic"},
        "unit_convention": LEGACY_FIXED_KPC,
        "model": {
            "name": "cmass_current",
            "components": {
                "mass_definition": "m5",
                "gamma_distribution": "dependent",
            },
        },
        "data": {
            "observation_path": str(tmp_path / "observations.hdf5"),
            "cross_section_path": str(tmp_path / "cross_section.h5"),
        },
        "box_prior": {
            "mu5_0": [9.0, 12.0],
            "beta5": [-3.0, 3.0],
            "xi5": [-3.0, 3.0],
            "sigma5": [1.0e-2, 0.2],
            "mu_gamma_0": [1.5, 2.5],
            "beta_gamma": [-3.0, 3.0],
            "xi_gamma": [-3.0, 3.0],
            "sigma_gamma": [0.0, 0.5],
            "mu_zs": [1.0, 3.0],
            "sigma_zs": [0.0, 2.0],
            "theta0": [0.0, 3.0],
            "loga": [-1.0, 3.0],
        },
        "sampling": {
            "random_seed": 7,
            "num_chains": 24,
            "num_samples": 3,
            "num_warmup": 1,
            "initial_center": {
                "mu5_0": 11.32,
                "beta5": 0.59,
                "xi5": -0.11,
                "sigma5": 0.06,
                "mu_gamma_0": 1.99,
                "beta_gamma": 0.1,
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
        "cosmology": {"h0": 70.0, "omega_m": 0.3},
        "runtime": {
            "checkpoint_every": 1,
            "parallel_strategy": "auto",
            "progress": False,
            "progress_summary_every": 1,
            "show_stage_timing": True,
            "disable_hdf5_file_locking": False,
            "num_threads": 0,
            "reserve_cores": 2,
        },
        "output": {
            "root_dir": str(tmp_path / "outputs"),
            "run_label": "registry-contract",
            "overwrite_latest": True,
        },
    }


def test_runtime_config_resolves_model_components_without_legacy_sections(tmp_path: Path) -> None:
    """New configs should resolve CMASS model components through the registry."""

    config_path = tmp_path / "new_model_contract.yaml"
    config_path.write_text(
        yaml.safe_dump(_minimal_new_model_config(tmp_path), sort_keys=False),
        encoding="utf-8",
    )

    runtime_config = load_runtime_config(config_path)

    assert runtime_config.model.name == "cmass_current"
    assert runtime_config.model.components["mass_definition"] == "m5"
    assert runtime_config.model.components["gamma_distribution"] == "dependent"
    assert runtime_config.mass_definition == get_mass_definition(5, unit_convention=LEGACY_FIXED_KPC)
    assert runtime_config.parameter_schema.model_name == "cmass_current"
    assert not hasattr(runtime_config, "gamma_model")


def test_legacy_top_level_model_sections_are_rejected(tmp_path: Path) -> None:
    """The breaking-change parser should fail fast on the removed YAML surface."""

    payload = _minimal_new_model_config(tmp_path)
    payload["mass_definition"] = {"enclosed_radius_kpc": 5}
    payload["gamma_model"] = {"mode": "dependent"}
    config_path = tmp_path / "legacy_sections.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="mass_definition.*gamma_model"):
        load_runtime_config(config_path)


def test_legacy_sampling_fields_are_rejected(tmp_path: Path) -> None:
    """The NumPyro-only config parser should reject emcee-era sampling names."""

    payload = _minimal_new_model_config(tmp_path)
    payload["sampling"].update(
        {
            "n_walkers": 24,
            "n_steps": 3,
            "warmup": 1,
        }
    )
    config_path = tmp_path / "legacy_sampling.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="n_walkers.*n_steps.*warmup"):
        load_runtime_config(config_path)


def test_model_registry_exposes_cmass_and_blocks_unimplemented_sonnenfeld() -> None:
    """
    Registry dispatch should be explicit about what is implemented today.

    The Sonnenfeld file exists as the next model module boundary, but the
    numerical implementation is intentionally not enabled by this refactor.
    """

    cmass_model = get_model_definition("cmass_current")
    assert cmass_model.name == "cmass_current"

    with pytest.raises(NotImplementedError, match="sonnenfeld2024_slacs"):
        get_model_definition("sonnenfeld2024_slacs")
