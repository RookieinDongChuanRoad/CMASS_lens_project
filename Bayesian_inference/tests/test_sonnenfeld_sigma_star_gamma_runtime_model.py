"""Runtime tests for the Sonnenfeld sigma-star-gamma peer model.

The tests use the same tiny canonical fixture style as the existing
Sonnenfeld runtime tests.  They are not a paper-results validation; they prove
that the new peer model is wired into the production Numba/emcee path and that
its sampled dimension is separate from the original 12D Sonnenfeld model.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest
import yaml

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.mass_definition import H_UNITS_V1, LEGACY_FIXED_KPC, get_mass_definition
from cmass_lens_inference.model_registry import get_model_definition
from cmass_lens_inference.models.sonnenfeld2024_slacs import paper_constants as parameters
from cmass_lens_inference.numba_backend.likelihood_engine import build_compiled_model, log_prob
from cmass_lens_inference.runner import run_inference
from test_canonical_dataset import _write_canonical_dataset


def _minimal_sigma_star_gamma_config(
    tmp_path: Path,
    dataset_path: Path,
    *,
    model_name: str,
    unit_convention: str,
) -> Path:
    """
    Write a compact config for runtime smoke tests.

    The helper mirrors the production YAML surface but keeps sampler and
    integration sizes tiny so the test exercises the real backend without
    turning into a production run.
    """

    paper_mu5_center = 11.2
    active_mu5_center = paper_mu5_center
    if unit_convention == H_UNITS_V1:
        active_mu5_center = paper_mu5_center + np.log10(0.7)

    payload = {
        "profile": {"name": "sersic"},
        "unit_convention": unit_convention,
        "model": {"name": model_name},
        "data": {"inference_dataset_path": str(dataset_path)},
        "box_prior": {
            "mu5_0": [10.0, 12.2],
            "beta5": [-3.0, 3.0],
            "xi5": [-3.0, 3.0],
            "sigma5": [1.0e-2, 0.3],
            "mu_gamma_0": [1.2, 2.8],
            "beta_sigma_star_gamma": [-3.0, 3.0],
            "sigma_gamma": [1.0e-2, 0.8],
            "mu_zs": [0.0, 2.0],
            "sigma_zs": [1.0e-3, 1.0],
            "theta0": [0.0, 3.0],
            "loga": [-1.0, 3.0],
        },
        "sampling": {
            "random_seed": 7,
            "n_walkers": 24,
            "n_steps": 2,
            "burn_in": 1,
            "initial_center": {
                "mu5_0": float(active_mu5_center),
                "beta5": 0.5,
                "xi5": -0.1,
                "sigma5": 0.08,
                "mu_gamma_0": 2.0,
                "beta_sigma_star_gamma": 0.05,
                "sigma_gamma": 0.15,
                "mu_zs": 1.5,
                "sigma_zs": 0.25,
                "theta0": 0.2,
                "loga": 0.0,
            },
            "initial_jitter_scale": 1.0e-3,
        },
        "integration": {
            "gamma_points": 16,
            "mstar_points": 12,
            "normalization_samples": 32,
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
            "run_label": "sonnenfeld-sigma-star-gamma-smoke",
            "overwrite_latest": True,
        },
    }
    config_path = tmp_path / f"{model_name}_{unit_convention}.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.fixture
def sigma_star_hunit_ready_dataset_path(tmp_path: Path) -> Path:
    """Create the smallest h-unit canonical dataset with Sonnenfeld capabilities."""

    return _write_canonical_dataset(
        tmp_path / "sonnenfeld_sigma_star_hunit_ready.hdf5",
        declare_population_sigma_unit=True,
        write_population_sigma_unit=True,
    )


@pytest.fixture
def sigma_star_fixed_m5_ready_dataset_path(
    tmp_path: Path,
    sigma_star_hunit_ready_dataset_path: Path,
) -> Path:
    """Rewrite fixture metadata so it represents paper-native fixed m5 input."""

    fixed_path = tmp_path / "sonnenfeld_sigma_star_fixed_m5_ready.hdf5"
    shutil.copy2(sigma_star_hunit_ready_dataset_path, fixed_path)
    with h5py.File(fixed_path, "r+") as handle:
        metadata = handle["metadata"]
        metadata.attrs["unit_convention"] = LEGACY_FIXED_KPC
        metadata.attrs["mass_definition_label"] = "m5"
    return fixed_path


def test_sigma_star_gamma_paper_native_log_prob_is_finite(
    tmp_path: Path,
    sigma_star_fixed_m5_ready_dataset_path: Path,
) -> None:
    """The paper-native model should evaluate through the Numba backend."""

    config = load_runtime_config(
        _minimal_sigma_star_gamma_config(
            tmp_path,
            sigma_star_fixed_m5_ready_dataset_path,
            model_name="sonnenfeld2024_slacs_sigma_star_gamma",
            unit_convention=LEGACY_FIXED_KPC,
        )
    )
    theta = config.sampling.initial_center.to_array()

    compiled = build_compiled_model(config)
    value, blob = log_prob(theta, compiled)

    assert config.parameter_schema.n_dim == 11
    assert np.isfinite(value)
    assert np.isfinite(float(blob["normalization_value"]))
    assert compiled.context.mass_radius_kpc == pytest.approx(5.0)
    assert compiled.context.mstar_pivot == pytest.approx(parameters.MSTAR_PIVOT_PHYSICAL)


def test_sigma_star_gamma_hunit_log_prob_is_finite(
    tmp_path: Path,
    sigma_star_hunit_ready_dataset_path: Path,
) -> None:
    """The h-unit model should share Sonnenfeld h-shift preprocessing."""

    config = load_runtime_config(
        _minimal_sigma_star_gamma_config(
            tmp_path,
            sigma_star_hunit_ready_dataset_path,
            model_name="sonnenfeld2024_slacs_sigma_star_gamma_hunit",
            unit_convention=H_UNITS_V1,
        )
    )
    theta = config.sampling.initial_center.to_array()

    compiled = build_compiled_model(config)
    value, blob = log_prob(theta, compiled)

    assert config.parameter_schema.n_dim == 11
    assert np.isfinite(value)
    assert np.isfinite(float(blob["normalization_value"]))
    assert compiled.context.mass_radius_kpc == pytest.approx(5.0 / 0.7)
    assert compiled.context.mstar_pivot == pytest.approx(
        parameters.shift_physical_mass_location_to_hunits(parameters.MSTAR_PIVOT_PHYSICAL, 0.7)
    )


def test_sigma_star_gamma_short_emcee_run_writes_11d_chain(
    tmp_path: Path,
    sigma_star_fixed_m5_ready_dataset_path: Path,
) -> None:
    """The production sampler should store an 11D chain for the new model."""

    config_path = _minimal_sigma_star_gamma_config(
        tmp_path,
        sigma_star_fixed_m5_ready_dataset_path,
        model_name="sonnenfeld2024_slacs_sigma_star_gamma",
        unit_convention=LEGACY_FIXED_KPC,
    )
    run_result = run_inference(str(config_path))

    assert run_result.status == "completed"
    assert run_result.metadata["backend"] == "numba_emcee"
    assert run_result.metadata["chain_storage"] == "emcee_hdf_backend"
    assert (run_result.run_dir / "chain.h5").exists()

    import emcee

    backend = emcee.backends.HDFBackend(str(run_result.run_dir / "chain.h5"), read_only=True)
    assert backend.get_chain().shape == (2, 24, 11)


def test_original_sonnenfeld_models_remain_12d() -> None:
    """Adding the peer model must not change the original Sonnenfeld schema."""

    paper_model = get_model_definition("sonnenfeld2024_slacs")
    hunit_model = get_model_definition("sonnenfeld2024_slacs_hunit")
    paper_schema = paper_model.build_parameter_schema(
        mass_definition=get_mass_definition(5, unit_convention=LEGACY_FIXED_KPC),
        public_box_prior=None,
    )
    hunit_schema = hunit_model.build_parameter_schema(
        mass_definition=get_mass_definition(5, unit_convention=H_UNITS_V1),
        public_box_prior=None,
    )

    assert paper_schema.n_dim == 12
    assert hunit_schema.n_dim == 12
    assert "beta_gamma" in paper_schema.public_parameter_names
    assert "xi_gamma" in paper_schema.public_parameter_names
    assert "beta_sigma_star_gamma" not in paper_schema.public_parameter_names
