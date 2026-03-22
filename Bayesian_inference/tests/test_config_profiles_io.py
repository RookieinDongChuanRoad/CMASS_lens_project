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

import numpy as np
import pytest
import yaml

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.io import load_cross_section_grid, load_observations, load_sigma_unit_table
from cmass_lens_inference.mass_definition import convert_log_enclosed_mass, get_mass_definition
from cmass_lens_inference.profiles import build_profile_spec


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
    assert runtime_config.gamma_model.mode == "dependent"
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
    assert runtime_config.fp_prior.fiducial_scatter == pytest.approx(0.047)
    assert runtime_config.fp_prior.scatter_error == pytest.approx(0.008)
    assert runtime_config.fp_prior.mu_v_prior == pytest.approx(2.341871, abs=1.0e-6)
    assert runtime_config.fp_prior.mu_v_error == pytest.approx(0.03)
    assert runtime_config.fp_prior.beta_v_prior == pytest.approx(0.25774, abs=1.0e-5)
    assert runtime_config.fp_prior.beta_v_error == pytest.approx(0.03)


def test_load_runtime_config_requires_explicit_gamma_model_section(tmp_path: Path) -> None:
    """
    New source configurations must declare the gamma parameterization mode.

    The implementation is allowed to auto-migrate legacy run snapshots during
    resume/PPC entrypoints, but fresh source YAML should fail fast if the
    model mode is omitted.
    """

    missing_gamma_model_path = tmp_path / "missing_gamma_model.yaml"
    missing_gamma_model_path.write_text(
        yaml.safe_dump(
            {
                "profile": {"name": "sersic"},
                "mass_definition": {"enclosed_radius_kpc": 5},
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

    with pytest.raises(KeyError, match="Missing required config section: gamma_model"):
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
                "mass_definition": {"enclosed_radius_kpc": 5},
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

    assert runtime_config.gamma_model.mode == "independent"
    parameter_vector = runtime_config.sampling.initial_center.to_array()
    assert parameter_vector.shape == (10,)
    public_center = runtime_config.sampling.initial_center.to_public_dict(
        runtime_config.mass_definition,
    )
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
                "mass_definition": {"enclosed_radius_kpc": 5},
                "gamma_model": {"mode": "independent"},
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

    assert runtime_config.gamma_model.mode == "sigma_star_dependent"
    parameter_vector = runtime_config.sampling.initial_center.to_array()
    assert parameter_vector.shape == (11,)
    public_center = runtime_config.sampling.initial_center.to_public_dict(
        runtime_config.mass_definition,
    )
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
                "mass_definition": {"enclosed_radius_kpc": 5},
                "gamma_model": {"mode": "sigma_star_dependent"},
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


def test_load_observations_converts_legacy_m5_grid_to_selected_mass_definition(
    synthetic_observation_file: Path,
) -> None:
    """
    Legacy observation files still store only root-level `m5` datasets.

    The loader must preserve backward compatibility by converting those
    datasets analytically when a run requests `m10`.
    """

    profile_spec = build_profile_spec("sersic")
    observations = load_observations(
        synthetic_observation_file,
        profile_spec,
        get_mass_definition(10),
    )

    assert len(observations) == 1
    observation = observations[0]
    expected_mass_grid = convert_log_enclosed_mass(
        log_mass=np.linspace(11.6, 10.8, 17),
        gamma=np.linspace(1.3, 2.7, 17),
        from_radius_kpc=5,
        to_radius_kpc=10,
    )

    np.testing.assert_allclose(observation.mass_grid_17, expected_mass_grid)
    np.testing.assert_allclose(observation.dmass_dthetaein_grid_17, np.linspace(-2.0, -1.0, 17))
    assert observation.s2_grid_17 is not None


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
