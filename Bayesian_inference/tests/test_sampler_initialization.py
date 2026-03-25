"""
Focused tests for walker initialization under explicit box-prior contracts.

These tests lock the strict-failure policy introduced with configurable
parameter bounds: initialization must honor the configured box prior instead of
quietly seeding invalid walkers and relying on later `-inf` rejections.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.sampler import initialize_walkers


def test_initialize_walkers_respects_configured_box_prior(
    synthetic_config_path: Path,
) -> None:
    """Every fresh walker coordinate should stay within the configured bounds."""

    runtime_config = load_runtime_config(synthetic_config_path)
    walkers = initialize_walkers(
        runtime_config.sampling.initial_center,
        runtime_config.sampling.n_walkers,
        runtime_config.sampling.initial_jitter_scale,
        runtime_config.sampling.random_seed,
    )
    lower_bounds = np.asarray(
        [lower for lower, _ in runtime_config.parameter_schema.prior_bounds],
        dtype=float,
    )
    upper_bounds = np.asarray(
        [upper for _, upper in runtime_config.parameter_schema.prior_bounds],
        dtype=float,
    )

    assert walkers.shape == (runtime_config.sampling.n_walkers, runtime_config.parameter_schema.n_dim)
    assert np.all(walkers >= lower_bounds[None, :])
    assert np.all(walkers <= upper_bounds[None, :])


def test_initialize_walkers_fails_when_bounds_leave_no_room_for_jitter(
    synthetic_config_path: Path,
) -> None:
    """Extremely tight box priors should fail fast instead of clipping walkers."""

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    initial_center = payload["sampling"]["initial_center"]
    payload["box_prior"] = {
        name: [float(value) - 1.0e-8, float(value) + 1.0e-8]
        for name, value in initial_center.items()
    }
    narrow_bounds_path = synthetic_config_path.parent / "narrow_bounds.yaml"
    narrow_bounds_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    runtime_config = load_runtime_config(narrow_bounds_path)

    with pytest.raises(ValueError, match="Unable to initialize"):
        initialize_walkers(
            runtime_config.sampling.initial_center,
            runtime_config.sampling.n_walkers,
            runtime_config.sampling.initial_jitter_scale,
            runtime_config.sampling.random_seed,
        )
