"""
Tests for configuration loading, profile selection, and HDF5 compatibility.

These tests lock the public API for the configuration and input layers before
the implementation exists. The goal is to make the structure explicit:
configuration must produce typed objects, profile-specific aliases must be
resolved centrally, and input files must be converted into normalized runtime
records that the numerical code can consume.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.io import load_cross_section_grid, load_observations, load_sigma_unit_table
from cmass_lens_inference.mass_definition import H_UNITS_V1, LEGACY_FIXED_KPC, get_mass_definition
from cmass_lens_inference.profiles import build_profile_spec


def _default_box_prior_payload(
    *,
    mass_radius_kpc: int,
    gamma_mode: str = "dependent",
) -> dict[str, list[float]]:
    """Build one full explicit box-prior payload for inline config tests."""

    if mass_radius_kpc == 10:
        mass_bounds = {
            "mu10_0": [9.0, 12.0],
            "beta10": [-3.0, 3.0],
            "xi10": [-3.0, 3.0],
            "sigma10": [1.0e-2, 0.2],
        }
    else:
        mass_bounds = {
            "mu5_0": [9.0, 12.0],
            "beta5": [-3.0, 3.0],
            "xi5": [-3.0, 3.0],
            "sigma5": [1.0e-2, 0.2],
        }

    if gamma_mode == "dependent":
        gamma_bounds = {
            "mu_gamma_0": [1.5, 2.5],
            "beta_gamma": [-3.0, 3.0],
            "xi_gamma": [-3.0, 3.0],
            "sigma_gamma": [0.0, 0.5],
        }
    elif gamma_mode == "independent":
        gamma_bounds = {
            "mu_gamma_0": [1.5, 2.5],
            "sigma_gamma": [0.0, 0.5],
        }
    elif gamma_mode == "sigma_star_dependent":
        gamma_bounds = {
            "mu_gamma_0": [1.5, 2.5],
            "beta_sigma_star_gamma": [-3.0, 3.0],
            "sigma_gamma": [0.0, 0.5],
        }
    else:
        raise ValueError(f"Unsupported gamma mode for test box-prior payload: {gamma_mode}")

    return {
        **mass_bounds,
        **gamma_bounds,
        "mu_zs": [1.0, 3.0],
        "sigma_zs": [0.0, 2.0],
        "theta0": [0.0, 3.0],
        "loga": [-1.0, 3.0],
    }


def test_load_runtime_config_builds_typed_sections(synthetic_config_path: Path) -> None:
    """
    The configuration loader should convert YAML into a typed runtime config.

    This test locks the project's public contract:
    - profile, sampling, integration, runtime, and output sections exist
    - the configured output root is preserved
    - default run labels remain user-visible because they are part of `run_id`
    """

    runtime_config = load_runtime_config(synthetic_config_path)

    assert runtime_config.profile.name == "sersic"
    assert runtime_config.mass_definition == get_mass_definition(5)
    assert runtime_config.model.components == {
        "mass_definition": "m5",
        "gamma_distribution": "dependent",
    }
    assert runtime_config.cosmology.h0 == 70.0
    assert runtime_config.cosmology.omega_m == 0.3
    assert runtime_config.sampling.n_walkers == 24
    assert runtime_config.integration.normalization_samples == 128
    assert runtime_config.output.run_label == "synthetic"
    assert runtime_config.output.root_dir == synthetic_config_path.parent / "outputs"
    assert runtime_config.runtime.parallel_strategy == "auto"
    assert runtime_config.runtime.num_threads == 0
    assert runtime_config.runtime.reserve_cores == 2
    assert runtime_config.runtime.progress_summary_every == 1
    assert runtime_config.runtime.show_stage_timing is True
    assert runtime_config.fp_prior.enabled is False
    assert runtime_config.data.sigma_table_path is None
    assert runtime_config.parameter_schema.prior_bounds[0] == pytest.approx((9.0, 12.0))


def test_load_runtime_config_builds_h_unit_mass_definition(synthetic_config_path: Path) -> None:
    """The h-units config surface should expose h-dependent labels and pivots."""

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    payload["unit_convention"] = "h_units_v1"
    payload["model"]["components"]["mass_definition"] = "m5_hinvkpc"
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
    h_unit_config_path = synthetic_config_path.parent / "h_unit_config.yaml"
    h_unit_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    runtime_config = load_runtime_config(h_unit_config_path)

    assert runtime_config.unit_convention == "h_units_v1"
    assert runtime_config.h_ref == pytest.approx(0.7)
    assert runtime_config.mass_definition.label == "m5_hinvkpc"
    assert runtime_config.mass_definition.subgroup_name == "m5_hinvkpc"
    assert runtime_config.parameter_schema.public_parameter_names[:4] == (
        "mu5h_0",
        "beta5h",
        "xi5h",
        "sigma5h",
    )


def test_load_runtime_config_requires_sigma_table_path_when_fp_prior_enabled(
    synthetic_config_path: Path,
) -> None:
    """FP prior cannot be enabled without an explicit sigma-table input path."""

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    payload["fp_prior"] = {"enabled": True}
    missing_sigma_table_path = synthetic_config_path.parent / "missing_sigma_table_path.yaml"
    missing_sigma_table_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sigma_table_path"):
        load_runtime_config(missing_sigma_table_path)


def test_load_runtime_config_builds_fp_prior_config_when_enabled(
    synthetic_config_path: Path,
    synthetic_sersic_sigma_table_file: Path,
) -> None:
    """The loader should preserve the optional FP-prior configuration surface."""

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    payload["data"]["sigma_table_path"] = str(synthetic_sersic_sigma_table_file)
    payload["fp_prior"] = {"enabled": True}
    fp_enabled_path = synthetic_config_path.parent / "fp_enabled.yaml"
    fp_enabled_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    runtime_config = load_runtime_config(fp_enabled_path)

    assert runtime_config.fp_prior.enabled is True
    assert runtime_config.data.sigma_table_path == synthetic_sersic_sigma_table_file
    assert runtime_config.fp_prior.fit_mstar_min == pytest.approx(11.0)
    assert runtime_config.fp_prior.pivot_mstar == pytest.approx(11.3)
    assert runtime_config.fp_prior.fiducial_scatter == pytest.approx(0.075)
    assert runtime_config.fp_prior.scatter_error == pytest.approx(0.003)
    assert runtime_config.fp_prior.mu_v_prior == pytest.approx(2.34548, abs=1.0e-5)
    assert runtime_config.fp_prior.mu_v_error == pytest.approx(0.00611, abs=1.0e-5)
    assert runtime_config.fp_prior.beta_v_prior == pytest.approx(0.176, abs=1.0e-6)
    assert runtime_config.fp_prior.beta_v_error == pytest.approx(0.011)


def test_load_runtime_config_requires_model_components_for_gamma_distribution(tmp_path: Path) -> None:
    """
    New source configurations must declare the gamma parameterization through
    ``model.components`` rather than the removed top-level ``gamma_model``.
    """

    missing_gamma_model_path = tmp_path / "missing_gamma_model.yaml"
    missing_gamma_model_path.write_text(
        yaml.safe_dump(
            {
                "profile": {"name": "sersic"},
                "unit_convention": "legacy_fixed_kpc",
                "model": {
                    "name": "cmass_current",
                    "components": {"mass_definition": "m5"},
                },
                "data": {
                    "observation_path": str(tmp_path / "observations.hdf5"),
                    "cross_section_path": str(tmp_path / "cross_section.h5"),
                },
                "box_prior": _default_box_prior_payload(mass_radius_kpc=5),
                "sampling": {
                    "n_walkers": 24,
                    "n_steps": 3,
                    "warmup": 1,
                    "random_seed": 7,
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
                },
                "integration": {
                    "gamma_points": 200,
                    "mstar_points": 200,
                    "normalization_samples": 128,
                },
                "cosmology": {
                    "h0": 70.0,
                    "omega_m": 0.3,
                },
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
                    "run_label": "synthetic",
                    "overwrite_latest": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="gamma_distribution"):
        load_runtime_config(missing_gamma_model_path)


def test_load_runtime_config_requires_explicit_cosmology_section(tmp_path: Path) -> None:
    """
    The astropy migration introduces a dedicated top-level `cosmology` section.

    Legacy configs that only provide the removed distance-table runtime knobs
    must fail fast so users do not unknowingly keep depending on deleted schema.
    """

    legacy_style_path = tmp_path / "legacy_runtime_only.yaml"
    legacy_style_path.write_text(
        yaml.safe_dump(
            {
                "profile": {"name": "sersic"},
                "unit_convention": "legacy_fixed_kpc",
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
                "box_prior": _default_box_prior_payload(mass_radius_kpc=5),
                "sampling": {
                    "n_walkers": 24,
                    "n_steps": 3,
                    "warmup": 1,
                    "random_seed": 7,
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
                },
                "integration": {
                    "gamma_points": 200,
                    "mstar_points": 200,
                    "normalization_samples": 128,
                },
                "runtime": {
                    "distance_table_max_z": 5.0,
                    "distance_table_size": 8001,
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
                    "run_label": "synthetic",
                    "overwrite_latest": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="Missing required config section: cosmology"):
        load_runtime_config(legacy_style_path)


def test_load_runtime_config_requires_explicit_box_prior_section(tmp_path: Path) -> None:
    """Fresh source configs must declare the full public-name box prior."""

    missing_box_prior_path = tmp_path / "missing_box_prior.yaml"
    missing_box_prior_path.write_text(
        yaml.safe_dump(
            {
                "profile": {"name": "sersic"},
                "unit_convention": "legacy_fixed_kpc",
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
                "sampling": {
                    "n_walkers": 24,
                    "n_steps": 3,
                    "warmup": 1,
                    "random_seed": 7,
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
                },
                "integration": {
                    "gamma_points": 200,
                    "mstar_points": 200,
                    "normalization_samples": 128,
                },
                "cosmology": {
                    "h0": 70.0,
                    "omega_m": 0.3,
                },
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
                    "run_label": "synthetic",
                    "overwrite_latest": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="Missing required config section: box_prior"):
        load_runtime_config(missing_box_prior_path)


def test_load_runtime_config_rejects_run_snapshot_missing_box_prior(tmp_path: Path) -> None:
    """Run snapshots use the same explicit box-prior contract as source configs."""

    snapshot_path = tmp_path / "config_snapshot.yaml"
    snapshot_path.write_text(
        yaml.safe_dump(
            {
                "profile": {"name": "sersic"},
                "unit_convention": "legacy_fixed_kpc",
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
                "sampling": {
                    "n_walkers": 24,
                    "n_steps": 3,
                    "warmup": 1,
                    "random_seed": 7,
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
                },
                "integration": {
                    "gamma_points": 200,
                    "mstar_points": 200,
                    "normalization_samples": 128,
                },
                "cosmology": {
                    "h0": 70.0,
                    "omega_m": 0.3,
                },
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
                    "run_label": "synthetic",
                    "overwrite_latest": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="Missing required config section: box_prior"):
        load_runtime_config(snapshot_path)


def test_load_runtime_config_maps_m10_public_parameter_names_to_internal_vector(
    synthetic_m10_config_path: Path,
) -> None:
    """The config loader should accept the `mu10_*` public naming surface."""

    runtime_config = load_runtime_config(synthetic_m10_config_path)

    assert runtime_config.mass_definition == get_mass_definition(10)
    np_values = runtime_config.sampling.initial_center.to_array()
    assert np_values[0] == pytest.approx(11.42)
    assert np_values[1] == pytest.approx(0.49)
    assert np_values[2] == pytest.approx(-0.21)
    assert np_values[3] == pytest.approx(0.08)


def test_load_runtime_config_builds_independent_gamma_parameter_vector(
    synthetic_independent_config_path: Path,
) -> None:
    """
    Independent gamma mode should expose only the gamma mean and scatter slots.

    This locks the public 10-dimensional parameter contract so later refactors
    cannot accidentally reintroduce the removed gamma slope parameters into the
    sampled vector or serialized public metadata.
    """

    runtime_config = load_runtime_config(synthetic_independent_config_path)

    assert runtime_config.model.components["gamma_distribution"] == "independent"
    parameter_vector = runtime_config.sampling.initial_center.to_array()
    assert parameter_vector.shape == (10,)
    public_center = runtime_config.sampling.initial_center.to_public_dict()
    assert "beta_gamma" not in public_center
    assert "xi_gamma" not in public_center
    assert set(public_center) == {
        "mu5_0",
        "beta5",
        "xi5",
        "sigma5",
        "mu_gamma_0",
        "sigma_gamma",
        "mu_zs",
        "sigma_zs",
        "theta0",
        "loga",
    }


def test_load_runtime_config_rejects_gamma_slopes_in_independent_mode(tmp_path: Path) -> None:
    """
    Independent gamma mode must reject the removed slope parameters explicitly.

    Silently ignoring `beta_gamma` or `xi_gamma` would make a malformed config
    look valid while sampling a different model than the user requested.
    """

    path = tmp_path / "invalid_independent_gamma.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "profile": {"name": "sersic"},
                "unit_convention": "legacy_fixed_kpc",
                "model": {
                    "name": "cmass_current",
                    "components": {
                        "mass_definition": "m5",
                        "gamma_distribution": "independent",
                    },
                },
                "data": {
                    "observation_path": str(tmp_path / "observations.hdf5"),
                    "cross_section_path": str(tmp_path / "cross_section.h5"),
                },
                "box_prior": _default_box_prior_payload(mass_radius_kpc=5, gamma_mode="independent"),
                "sampling": {
                    "n_walkers": 24,
                    "n_steps": 3,
                    "warmup": 1,
                    "random_seed": 7,
                    "initial_center": {
                        "mu5_0": 11.32,
                        "beta5": 0.59,
                        "xi5": -0.11,
                        "sigma5": 0.06,
                        "mu_gamma_0": 1.99,
                        "beta_gamma": 0.1,
                        "sigma_gamma": 0.149,
                        "mu_zs": 1.8,
                        "sigma_zs": 0.215,
                        "theta0": 0.93,
                        "loga": 1.0,
                    },
                },
                "integration": {
                    "gamma_points": 200,
                    "mstar_points": 200,
                    "normalization_samples": 128,
                },
                "cosmology": {
                    "h0": 70.0,
                    "omega_m": 0.3,
                },
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
                    "run_label": "synthetic",
                    "overwrite_latest": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="beta_gamma"):
        load_runtime_config(path)


def test_load_runtime_config_rejects_incomplete_box_prior_mapping(tmp_path: Path) -> None:
    """Omitting one sampled parameter from `box_prior` should fail clearly."""

    path = tmp_path / "incomplete_box_prior.yaml"
    payload = {
        "profile": {"name": "sersic"},
        "unit_convention": "legacy_fixed_kpc",
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
        "box_prior": _default_box_prior_payload(mass_radius_kpc=5),
        "sampling": {
            "n_walkers": 24,
            "n_steps": 3,
            "warmup": 1,
            "random_seed": 7,
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
        },
        "integration": {
            "gamma_points": 200,
            "mstar_points": 200,
            "normalization_samples": 128,
        },
        "cosmology": {
            "h0": 70.0,
            "omega_m": 0.3,
        },
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
            "run_label": "synthetic",
            "overwrite_latest": True,
        },
    }
    payload["box_prior"].pop("theta0")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="theta0"):
        load_runtime_config(path)


def test_load_runtime_config_builds_sigma_star_gamma_parameter_vector(
    synthetic_sigma_star_dependent_config_path: Path,
) -> None:
    """
    Sigma-star gamma mode should expose the 11D public parameter contract.

    The third mode removes the historical `beta_gamma` / `xi_gamma` pair and
    replaces them with one explicit `Sigma_*` slope. This test locks both the
    sampled vector length and the serialized public names.
    """

    runtime_config = load_runtime_config(synthetic_sigma_star_dependent_config_path)

    assert runtime_config.model.components["gamma_distribution"] == "sigma_star_dependent"
    parameter_vector = runtime_config.sampling.initial_center.to_array()
    assert parameter_vector.shape == (11,)
    public_center = runtime_config.sampling.initial_center.to_public_dict()
    assert "beta_gamma" not in public_center
    assert "xi_gamma" not in public_center
    assert public_center["beta_sigma_star_gamma"] == pytest.approx(0.24)
    assert set(public_center) == {
        "mu5_0",
        "beta5",
        "xi5",
        "sigma5",
        "mu_gamma_0",
        "beta_sigma_star_gamma",
        "sigma_gamma",
        "mu_zs",
        "sigma_zs",
        "theta0",
        "loga",
    }


def test_load_runtime_config_rejects_legacy_gamma_slopes_in_sigma_star_mode(tmp_path: Path) -> None:
    """
    Sigma-star gamma mode must reject the removed `logM* + logRe` slope names.

    Accepting `beta_gamma` or `xi_gamma` here would silently change the model
    family while preserving a superficially valid config surface.
    """

    path = tmp_path / "invalid_sigma_star_gamma.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "profile": {"name": "sersic"},
                "unit_convention": "legacy_fixed_kpc",
                "model": {
                    "name": "cmass_current",
                    "components": {
                        "mass_definition": "m5",
                        "gamma_distribution": "sigma_star_dependent",
                    },
                },
                "data": {
                    "observation_path": str(tmp_path / "observations.hdf5"),
                    "cross_section_path": str(tmp_path / "cross_section.h5"),
                },
                "box_prior": _default_box_prior_payload(mass_radius_kpc=5, gamma_mode="sigma_star_dependent"),
                "sampling": {
                    "n_walkers": 24,
                    "n_steps": 3,
                    "warmup": 1,
                    "random_seed": 7,
                    "initial_center": {
                        "mu5_0": 11.32,
                        "beta5": 0.59,
                        "xi5": -0.11,
                        "sigma5": 0.06,
                        "mu_gamma_0": 1.99,
                        "beta_gamma": 0.1,
                        "sigma_gamma": 0.149,
                        "mu_zs": 1.8,
                        "sigma_zs": 0.215,
                        "theta0": 0.93,
                        "loga": 1.0,
                    },
                },
                "integration": {
                    "gamma_points": 200,
                    "mstar_points": 200,
                    "normalization_samples": 128,
                },
                "cosmology": {
                    "h0": 70.0,
                    "omega_m": 0.3,
                },
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
                    "run_label": "synthetic",
                    "overwrite_latest": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="beta_gamma"):
        load_runtime_config(path)


def test_load_runtime_config_rejects_initial_center_outside_box_prior(
    synthetic_config_path: Path,
) -> None:
    """The configured initial center must already satisfy the explicit bounds."""

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    payload["box_prior"]["mu5_0"] = [9.0, 11.0]
    invalid_center_path = synthetic_config_path.parent / "invalid_initial_center_bounds.yaml"
    invalid_center_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mu5_0"):
        load_runtime_config(invalid_center_path)


def test_build_profile_spec_exposes_profile_specific_rules() -> None:
    """
    The profile builder should isolate all devauc/sersic differences.

    This protects the main inference pipeline from accumulating profile
    conditionals in numerically sensitive code paths.
    """

    devauc = build_profile_spec("devauc")
    sersic = build_profile_spec("sersic")

    assert devauc.fixed_n == 4.0
    assert devauc.uses_observed_n_in_likelihood is False
    assert devauc.observation_field_aliases["stellar_mass"] == ("logmchab_deV", "logmchab")
    assert devauc.observation_field_aliases["effective_radius_arcsec"] == ("reff_deV", "re_arcsec")

    assert sersic.fixed_n is None
    assert sersic.uses_observed_n_in_likelihood is True
    assert sersic.observation_field_aliases["stellar_mass"] == ("logmchab",)
    assert sersic.observation_field_aliases["effective_radius_arcsec"] == ("re_arcsec",)


def test_load_observations_uses_devauc_aliases(
    synthetic_devauc_observation_file: Path,
) -> None:
    """
    The devauc reader must prefer de Vaucouleurs-specific aliases when present.

    This is a critical compatibility rule from the requirements document and
    must be enforced in the I/O layer rather than inside the statistical model.
    """

    profile_spec = build_profile_spec("devauc")
    observations = load_observations(
        synthetic_devauc_observation_file,
        profile_spec,
        get_mass_definition(5),
    )

    assert len(observations) == 1
    observation = observations[0]

    assert observation.lens_id == "lens-devauc"
    assert observation.log_stellar_mass_obs == 11.1
    assert observation.log_stellar_mass_err == 0.04
    assert observation.effective_radius_arcsec == 1.4
    assert observation.n_observed == 4.0
    assert observation.num_sigma == 0


def test_load_observations_validates_and_reads_h_unit_contract(
    tmp_path: Path,
    synthetic_observation_file: Path,
) -> None:
    """
    h-units observation files must expose h-specific fields and metadata.

    This test deliberately exercises both directions of the guard: an h-units
    config cannot consume legacy files, and a legacy config cannot consume an
    explicitly h-units file.
    """

    path = tmp_path / "synthetic_h_units_observations.hdf5"
    gamma_grid = np.linspace(1.3, 2.7, 17)
    with h5py.File(path, "w") as handle:
        handle.attrs["unit_convention"] = H_UNITS_V1
        handle.attrs["h_ref"] = 0.7
        group = handle.create_group("lens-h")
        group.attrs["unit_convention"] = H_UNITS_V1
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
        mass_group.attrs["unit_convention"] = H_UNITS_V1
        mass_group.attrs["h_ref"] = 0.7
        mass_group.create_dataset("mass_grid", data=np.linspace(11.7, 10.9, 17))
        mass_group.create_dataset("dmass_dthetaein_grid", data=np.linspace(-2.0, -1.0, 17))

    h_mass_definition = get_mass_definition(5, unit_convention=H_UNITS_V1)
    observations = load_observations(
        path,
        build_profile_spec("sersic"),
        h_mass_definition,
        h_ref=0.7,
    )

    assert observations[0].log_stellar_mass_obs == pytest.approx(10.99)
    assert observations[0].log_effective_radius_obs == pytest.approx(0.64)
    assert observations[0].mass_grid_17[0] == pytest.approx(11.7)

    with pytest.raises(ValueError, match="missing unit_convention"):
        load_observations(
            synthetic_observation_file,
            build_profile_spec("sersic"),
            h_mass_definition,
            h_ref=0.7,
        )

    with pytest.raises(ValueError, match="does not match active convention"):
        load_observations(
            path,
            build_profile_spec("sersic"),
            get_mass_definition(5, unit_convention=LEGACY_FIXED_KPC),
            h_ref=0.7,
        )


def test_load_sigma_unit_table_reads_supported_hdf5_schema(
    synthetic_sersic_sigma_table_file: Path,
) -> None:
    """The sigma-table loader should return the normalized HDF5 schema verbatim."""

    sigma_table = load_sigma_unit_table(
        synthetic_sersic_sigma_table_file,
        build_profile_spec("sersic"),
        get_mass_definition(5),
    )

    assert sigma_table.profile_name == "sersic"
    assert sigma_table.mass_definition_label == "m5"
    assert sigma_table.mass_radius_kpc == pytest.approx(5.0)
    assert sigma_table.gamma_axis.shape == (5,)
    assert sigma_table.zd_axis.shape == (4,)
    assert sigma_table.log_re_kpc_axis.shape == (3,)
    assert sigma_table.n_axis is not None
    assert sigma_table.n_axis.shape == (4,)
    assert sigma_table.sigma_unit_grid.shape == (5, 4, 3, 4)


def test_load_sigma_unit_table_validates_h_unit_contract(
    tmp_path: Path,
    synthetic_sersic_sigma_table_file: Path,
) -> None:
    """Sigma tables must fail fast when their convention differs from config."""

    path = tmp_path / "synthetic_h_units_sigma_table.h5"
    gamma_axis = np.linspace(1.2, 2.8, 5)
    zd_axis = np.linspace(0.43, 0.82, 4)
    log_re_axis = np.linspace(0.35, 1.1, 3)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "sigma_unit_hdf5_v1"
        handle.attrs["unit_convention"] = H_UNITS_V1
        handle.attrs["h_ref"] = 0.7
        handle.attrs["mass_definition_label"] = "m5_hinvkpc"
        handle.attrs["mass_radius_kpc"] = 5.0
        handle.attrs["units"] = "km2 s-2 per 10**m5_hinvkpc"
        handle.create_dataset("profile_name", data=np.bytes_("sersic"))
        handle.create_dataset("gamma_axis", data=gamma_axis)
        handle.create_dataset("zd_axis", data=zd_axis)
        handle.create_dataset("log_re_kpc_axis", data=log_re_axis)
        handle.create_dataset("n_axis", data=np.linspace(2.5, 10.5, 4))
        handle.create_dataset("s_unit_grid", data=np.ones((5, 4, 3, 4)))

    h_mass_definition = get_mass_definition(5, unit_convention=H_UNITS_V1)
    sigma_table = load_sigma_unit_table(
        path,
        build_profile_spec("sersic"),
        h_mass_definition,
        h_ref=0.7,
    )

    assert sigma_table.unit_convention == H_UNITS_V1
    assert sigma_table.h_ref == pytest.approx(0.7)
    assert sigma_table.mass_definition_label == "m5_hinvkpc"

    with pytest.raises(ValueError, match="missing unit_convention"):
        load_sigma_unit_table(
            synthetic_sersic_sigma_table_file,
            build_profile_spec("sersic"),
            h_mass_definition,
            h_ref=0.7,
        )

    with pytest.raises(ValueError, match="does not match active convention"):
        load_sigma_unit_table(
            path,
            build_profile_spec("sersic"),
            get_mass_definition(5, unit_convention=LEGACY_FIXED_KPC),
            h_ref=0.7,
        )


def test_load_sigma_unit_table_reads_requested_boss_bundle_leaf(
    synthetic_boss_sigma_bundle_file: Path,
) -> None:
    """FP-prior loading must support bundle files and select the active BOSS leaf."""

    sigma_table = load_sigma_unit_table(
        synthetic_boss_sigma_bundle_file,
        build_profile_spec("devauc"),
        get_mass_definition(5),
        observation_flavor="boss",
    )

    assert sigma_table.profile_name == "devauc"
    assert sigma_table.mass_definition_label == "m5"
    assert sigma_table.mass_radius_kpc == pytest.approx(5.0)
    assert sigma_table.gamma_axis.shape == (5,)
    assert sigma_table.zd_axis.shape == (4,)
    assert sigma_table.log_re_kpc_axis.shape == (3,)
    assert sigma_table.n_axis is None
    assert sigma_table.sigma_unit_grid.shape == (5, 4, 3)


def test_load_sigma_unit_table_reads_requested_within_re_bundle_leaf(
    synthetic_within_re_sigma_bundle_file: Path,
) -> None:
    """The loader must support explicit bundle-group reads for within-Re leaves."""

    sigma_table = load_sigma_unit_table(
        synthetic_within_re_sigma_bundle_file,
        build_profile_spec("sersic"),
        get_mass_definition(5),
        bundle_group="within_re",
    )

    assert sigma_table.profile_name == "sersic"
    assert sigma_table.mass_definition_label == "m5"
    assert sigma_table.mass_radius_kpc == pytest.approx(5.0)
    assert sigma_table.gamma_axis.shape == (5,)
    assert sigma_table.zd_axis is None
    assert sigma_table.log_re_kpc_axis.shape == (3,)
    assert sigma_table.n_axis is not None
    assert sigma_table.n_axis.shape == (4,)
    assert sigma_table.sigma_unit_grid.shape == (5, 3, 4)
    assert sigma_table.sigma_definition == "within_re"
    assert sigma_table.bundle_group_name == "within_re"
    assert sigma_table.observation_flavor is None
    assert sigma_table.bundle_leaf_path == "/within_re/m5"


def test_load_sigma_unit_table_rejects_missing_within_re_bundle_group(
    synthetic_boss_sigma_bundle_file: Path,
) -> None:
    """Explicit within-Re reads must fail on legacy-compatible bundles that only carry slit/boss."""

    with pytest.raises(ValueError, match="does not contain the bundle group 'within_re'"):
        load_sigma_unit_table(
            synthetic_boss_sigma_bundle_file,
            build_profile_spec("sersic"),
            get_mass_definition(5),
            bundle_group="within_re",
        )


def test_build_compiled_context_rejects_fp_prior_bundle_without_within_re_leaf(
    synthetic_bad_boss_sigma_bundle_file: Path,
    synthetic_fp_prior_config_path: Path,
) -> None:
    """
    FP prior should fail fast when the configured sigma bundle lacks within-Re data.

    The scientific contract now requires effective-radius sigma for every
    FP-enabled inference run. A bundle that only carries observation-flavor
    leaves must therefore be rejected immediately instead of silently falling
    back to slit or BOSS apertures.
    """

    from cmass_lens_inference.compiled_context import build_compiled_context
    from cmass_lens_inference.config import load_runtime_config

    payload = yaml.safe_load(synthetic_fp_prior_config_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["data"]["sigma_table_path"] = str(synthetic_bad_boss_sigma_bundle_file)

    broken_config_path = synthetic_fp_prior_config_path.parent / "missing_within_re_fp.yaml"
    broken_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    runtime_config = load_runtime_config(broken_config_path)
    with pytest.raises(ValueError, match="within_re"):
        build_compiled_context(runtime_config)


def test_load_sigma_unit_table_rejects_boss_bundle_with_wrong_seeing(
    synthetic_bad_boss_sigma_bundle_file: Path,
) -> None:
    """BOSS bundle leaves must fail fast when their seeing contract is not 1.5 arcsec."""

    with pytest.raises(ValueError, match="seeing"):
        load_sigma_unit_table(
            synthetic_bad_boss_sigma_bundle_file,
            build_profile_spec("devauc"),
            get_mass_definition(5),
            observation_flavor="boss",
        )


def test_load_sigma_unit_table_rejects_profile_mismatch(
    synthetic_devauc_sigma_table_file: Path,
) -> None:
    """A `devauc` sigma table must not be silently accepted for a `sersic` run."""

    with pytest.raises(ValueError, match="profile"):
        load_sigma_unit_table(
            synthetic_devauc_sigma_table_file,
            build_profile_spec("sersic"),
            get_mass_definition(5),
        )


def test_load_sigma_unit_table_rejects_mass_definition_mismatch(
    synthetic_sersic_m10_sigma_table_file: Path,
) -> None:
    """The sigma-table loader must fail fast on `m5` versus `m10` mismatches."""

    with pytest.raises(ValueError, match="mass definition"):
        load_sigma_unit_table(
            synthetic_sersic_m10_sigma_table_file,
            build_profile_spec("sersic"),
            get_mass_definition(5),
        )


def test_load_observations_rejects_root_level_legacy_grids_without_namespaced_subgroups(
    synthetic_legacy_only_observation_file: Path,
) -> None:
    """
    Observation loading no longer supports root-level mass/sigma grids.

    Once the migration is complete, every supported file must expose
    `mass_definitions/<label>/...` and the loader should fail fast instead of
    silently falling back to deprecated root-level datasets.
    """

    profile_spec = build_profile_spec("sersic")
    with pytest.raises(KeyError, match="mass-definition subgroup"):
        load_observations(
            synthetic_legacy_only_observation_file,
            profile_spec,
            get_mass_definition(10),
        )


def test_load_observations_reads_namespaced_mass_definition_subgroup_when_available(
    synthetic_namespaced_observation_file: Path,
) -> None:
    """
    The new HDF5 schema stores one subgroup per mass definition under each lens.

    The reader must select the subgroup matching the active run rather than
    assuming the historical root-level `m5` datasets exist.
    """

    profile_spec = build_profile_spec("sersic")
    observations = load_observations(
        synthetic_namespaced_observation_file,
        profile_spec,
        get_mass_definition(10),
    )

    assert len(observations) == 1
    observation = observations[0]

    assert observation.lens_id == "lens-namespaced"
    np.testing.assert_allclose(observation.mass_grid_17, np.linspace(11.75, 10.95, 17))
    np.testing.assert_allclose(observation.dmass_dthetaein_grid_17, np.linspace(-1.9, -1.1, 17))
    assert observation.s2_grid_17 is not None
    np.testing.assert_allclose(observation.s2_grid_17, np.linspace(0.45, 0.75, 17))


def test_load_observations_accepts_boss_contract_with_seeing_one_point_five(
    synthetic_boss_observation_file: Path,
) -> None:
    """BOSS raw files should be accepted when every lens group records the 1.5 arcsec seeing contract."""

    observations = load_observations(
        synthetic_boss_observation_file,
        build_profile_spec("devauc"),
        get_mass_definition(5),
    )

    assert len(observations) == 1
    assert observations[0].lens_id == "lens-boss"
    assert observations[0].num_sigma == 1


def test_load_observations_rejects_boss_contract_with_wrong_seeing(
    synthetic_boss_observation_file: Path,
) -> None:
    """BOSS raw files should fail fast when they carry the retired 0.9 arcsec seeing."""

    with h5py.File(synthetic_boss_observation_file, "a") as handle:
        handle["lens-boss"].attrs["seeing_fwhm_arcsec"] = 0.9

    with pytest.raises(ValueError, match="BOSS.*1.5|seeing"):
        load_observations(
            synthetic_boss_observation_file,
            build_profile_spec("devauc"),
            get_mass_definition(5),
        )


def test_load_cross_section_grid_supports_real_world_alias_names(
    synthetic_cross_section_file: Path,
) -> None:
    """
    The cross-section loader must accept the alias field names observed in the
    real `cs_grid_power.h5` file.
    """

    cross_section_grid = load_cross_section_grid(synthetic_cross_section_file)

    assert cross_section_grid.gamma_grid.shape == (25,)
    assert cross_section_grid.cs_over_theta_ein.shape == (25,)
    assert cross_section_grid.gamma_grid[0] == 1.2
