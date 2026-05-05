"""
Real-data regression tests for the canonical inference-dataset path.

These tests are intentionally marked as both ``real_data`` and ``slow`` because
they require the locally synced ``data/`` products and compile the real CMASS
JAX log-probability.  They protect the migration boundary that matters most in
this cleanup step: production inference should consume canonical HDF5, while
legacy raw files remain available only as a numerical oracle.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from cmass_lens_inference.compiled_context import build_compiled_context as build_legacy_context
from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.jax_backend.likelihood_engine import (
    build_compiled_model as build_jax_model,
    log_prob as jax_log_prob,
)
from cmass_lens_inference.parallel import resolve_parallelism
from cmass_lens_inference.types import CompiledModel, DataConfig


pytestmark = [pytest.mark.real_data, pytest.mark.slow]

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPOSITORY_ROOT / "data"
DEVauc_CANONICAL_SLIT = DATA_ROOT / "external" / "inference_dataset_devauc_slit_m5_hunits_v1.hdf5"
DEVauc_CANONICAL_BOSS = DATA_ROOT / "external" / "inference_dataset_devauc_boss_m5_hunits_v1.hdf5"
DEVauc_RAW_SLIT = DATA_ROOT / "raw" / "observations_deV_with_mass_grids_hunits_v1.hdf5"
LEGACY_CROSS_SECTION = DATA_ROOT / "external" / "cs_grid_power.h5"


def _skip_if_missing(paths: tuple[Path, ...]) -> None:
    """Skip real-data tests with one complete missing-file report."""

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        pytest.skip("Missing locally synced real-data products: " + ", ".join(missing))


def _devauc_runtime_config_payload(*, dataset_path: Path, output_root: Path, fp_prior_enabled: bool = False) -> dict:
    """
    Build the small real-data config used by these migration tests.

    The integration sizes are deliberately reduced from production values.  The
    test goal is canonical-vs-legacy equivalence and finite FP diagnostics, not
    production-quality posterior sampling.
    """

    return {
        "profile": {"name": "devauc"},
        "unit_convention": "h_units_v1",
        "model": {"name": "cmass"},
        "fp_prior": {"enabled": bool(fp_prior_enabled)},
        "data": {"inference_dataset_path": str(dataset_path)},
        "box_prior": {
            "mu5h_0": [9.0, 12.0],
            "beta5h": [-3.0, 3.0],
            "xi5h": [-3.0, 3.0],
            "sigma5h": [1.0e-2, 0.2],
            "mu_gamma_0": [1.5, 2.5],
            "beta_sigma_star_gamma": [-3.0, 3.0],
            "sigma_gamma": [0.0, 0.5],
            "mu_zs": [1.0, 3.0],
            "sigma_zs": [0.0, 2.0],
            "theta0": [0.0, 3.0],
            "loga": [-1.0, 3.0],
        },
        "sampling": {
            "random_seed": 7,
            "num_chains": 2,
            "num_samples": 2,
            "num_warmup": 1,
            "thinning": 1,
            "chain_method": "sequential",
            "initial_jitter_scale": 1.0e-3,
            "initial_center": {
                "mu5h_0": 11.32,
                "beta5h": 0.59,
                "xi5h": -0.11,
                "sigma5h": 0.06,
                "mu_gamma_0": 1.99,
                "beta_sigma_star_gamma": 0.10,
                "sigma_gamma": 0.149,
                "mu_zs": 1.8,
                "sigma_zs": 0.215,
                "theta0": 0.93,
                "loga": 1.0,
            },
        },
        "integration": {
            "gamma_points": 80,
            "mstar_points": 80,
            "normalization_samples": 4096,
        },
        "cosmology": {"h0": 70.0, "omega_m": 0.3},
        "runtime": {
            "checkpoint_every": 1,
            "parallel_strategy": "off",
            "progress": False,
            "progress_summary_every": 1,
            "show_stage_timing": False,
            "disable_hdf5_file_locking": False,
            "num_threads": 0,
            "reserve_cores": 2,
        },
        "output": {
            "root_dir": str(output_root),
            "run_label": "real-data-canonical",
            "overwrite_latest": True,
        },
    }


def _write_runtime_config(tmp_path: Path, payload: dict, file_name: str) -> Path:
    """Write one temporary YAML config and return its path."""

    config_path = tmp_path / file_name
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _legacy_oracle_compiled_model(runtime_config, *, observation_path: Path, cross_section_path: Path) -> CompiledModel:
    """
    Build a compiled model from the legacy raw readers for oracle comparisons.

    ``load_runtime_config()`` no longer accepts raw paths after this migration,
    so the test constructs this object explicitly.  That keeps the legacy path
    out of production configuration while still preserving an independent
    numerical reference during the canonical transition.
    """

    legacy_runtime_config = replace(
        runtime_config,
        data=DataConfig(
            observation_path=observation_path.resolve(),
            cross_section_path=cross_section_path.resolve(),
        ),
    )
    context, profile, cross_section_grid, cosmology, _random_basis, _observations = build_legacy_context(
        legacy_runtime_config
    )
    parallelism = resolve_parallelism(
        legacy_runtime_config.runtime,
        legacy_runtime_config.sampling.num_chains,
    )
    return CompiledModel(
        config=legacy_runtime_config,
        profile=profile,
        cross_section_grid=cross_section_grid,
        cosmology=cosmology,
        parallelism=parallelism,
        context=context,
    )


def test_real_devauc_canonical_log_prob_matches_legacy_raw_oracle(tmp_path: Path) -> None:
    """The real devauc canonical dataset should preserve legacy log-probability."""

    _skip_if_missing((DEVauc_CANONICAL_SLIT, DEVauc_RAW_SLIT, LEGACY_CROSS_SECTION))
    config_path = _write_runtime_config(
        tmp_path,
        _devauc_runtime_config_payload(
            dataset_path=DEVauc_CANONICAL_SLIT,
            output_root=tmp_path / "outputs",
            fp_prior_enabled=False,
        ),
        "real_devauc_canonical.yaml",
    )
    runtime_config = load_runtime_config(config_path)
    theta = runtime_config.sampling.initial_center.to_array()

    canonical_value, canonical_blob = jax_log_prob(theta, build_jax_model(runtime_config))
    legacy_value, legacy_blob = jax_log_prob(
        theta,
        _legacy_oracle_compiled_model(
            runtime_config,
            observation_path=DEVauc_RAW_SLIT,
            cross_section_path=LEGACY_CROSS_SECTION,
        ),
    )

    assert canonical_value == pytest.approx(legacy_value, rel=1.0e-10, abs=1.0e-10)
    assert float(canonical_blob["normalization_value"]) == pytest.approx(
        float(legacy_blob["normalization_value"]),
        rel=1.0e-10,
        abs=1.0e-10,
    )


@pytest.mark.parametrize("dataset_path", [DEVauc_CANONICAL_SLIT, DEVauc_CANONICAL_BOSS])
def test_real_devauc_canonical_fp_prior_diagnostics_are_finite(tmp_path: Path, dataset_path: Path) -> None:
    """Canonical devauc slit and BOSS datasets should carry usable FP grids."""

    _skip_if_missing((dataset_path,))
    config_path = _write_runtime_config(
        tmp_path,
        _devauc_runtime_config_payload(
            dataset_path=dataset_path,
            output_root=tmp_path / "outputs",
            fp_prior_enabled=True,
        ),
        f"{dataset_path.stem}_fp.yaml",
    )
    runtime_config = load_runtime_config(config_path)
    theta = runtime_config.sampling.initial_center.to_array()

    log_prob_value, blob = jax_log_prob(theta, build_jax_model(runtime_config))

    assert np.isfinite(log_prob_value)
    assert np.isfinite(float(blob["fp_prior_log_term"]))
    assert np.isfinite(float(blob["fpfit_mu"]))
    assert np.isfinite(float(blob["fpfit_beta"]))
    assert np.isnan(float(blob["fpfit_xi"]))
    assert np.isfinite(float(blob["fpfit_scatter"]))
