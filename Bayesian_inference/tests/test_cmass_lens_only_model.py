"""Tests for the CMASS lens-only model."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from cmass_lens_inference.canonical_dataset import (
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
)
from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.model_registry import get_model_definition
from cmass_lens_inference.numba_backend.likelihood_engine import (
    build_compiled_model as build_numba_model,
    log_prob as numba_log_prob,
)


def test_cmass_lens_only_is_registered_as_concrete_model() -> None:
    """The registry should expose lens-only as its own scientific model."""

    model_definition = get_model_definition("cmass_lens_only")

    assert model_definition.name == "cmass_lens_only"
    assert model_definition.backend_kernel == "cmass_lens_only"
    assert model_definition.required_capabilities == (
        CAPABILITY_LENS_OBSERVATIONS_V1,
        CAPABILITY_LENSING_MASS_GRIDS_V1,
        CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    )
    assert (
        CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1
        not in model_definition.required_capabilities
    )
    assert model_definition.optional_capabilities == ()


def test_cmass_lens_only_config_has_lens_only_parameter_schema(
    synthetic_lens_only_config_path: Path,
) -> None:
    """Lens-only should drop source-redshift and discovery parameters."""

    runtime_config = load_runtime_config(synthetic_lens_only_config_path)

    assert runtime_config.model.name == "cmass_lens_only"
    assert runtime_config.parameter_schema.public_parameter_names == (
        "mu_mstar_lens",
        "sigma_mstar_lens",
        "mu5h_0",
        "beta5h",
        "xi5h",
        "sigma5h",
        "mu_gamma_0",
        "beta_sigma_star_gamma",
        "sigma_gamma",
    )
    assert "mu_zs" not in runtime_config.parameter_schema.public_parameter_names
    assert "sigma_zs" not in runtime_config.parameter_schema.public_parameter_names
    assert "theta0" not in runtime_config.parameter_schema.public_parameter_names
    assert "loga" not in runtime_config.parameter_schema.public_parameter_names
    assert runtime_config.parameter_schema.model_metadata["selection_correction"] is False


def test_cmass_lens_only_rejects_fp_prior(
    synthetic_lens_only_config_path: Path,
) -> None:
    """The first lens-only implementation should not silently mix FP prior semantics."""

    payload = yaml.safe_load(synthetic_lens_only_config_path.read_text(encoding="utf-8"))
    payload["fp_prior"] = {"enabled": True}
    config_path = synthetic_lens_only_config_path.parent / "lens_only_fp_prior.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    runtime_config = load_runtime_config(config_path)

    with pytest.raises(ValueError, match="cmass_lens_only.*fp_prior"):
        build_numba_model(runtime_config)
