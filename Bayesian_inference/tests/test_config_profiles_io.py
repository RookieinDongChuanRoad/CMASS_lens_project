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
from cmass_lens_inference.models.cmass.constants import (
    CMASS_FP_PRIOR_DEFAULTS_20260429,
    DEVAUC_PROFILE_CONSTANTS,
    SERSIC_PROFILE_CONSTANTS,
)
from cmass_lens_inference.profiles import build_profile_spec


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SONNENFELD_PUBLIC_PARAMETERS = (
    "mu5_0",
    "beta5",
    "xi5",
    "sigma5",
    "mu_gamma_0",
    "beta_gamma",
    "xi_gamma",
    "sigma_gamma",
    "mu_zs",
    "sigma_zs",
    "theta0",
    "loga",
)
SONNENFELD_SIGMA_STAR_PUBLIC_PARAMETERS = (
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
)
SONNENFELD_PAPER_MU5_0 = 11.332


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
    assert runtime_config.mass_definition == get_mass_definition(5, unit_convention=H_UNITS_V1)
    assert runtime_config.model.name == "cmass"
    assert not hasattr(runtime_config.model, "components")
    assert runtime_config.cosmology.h0 == 70.0
    assert runtime_config.cosmology.omega_m == 0.3
    assert runtime_config.sampling.n_walkers == 24
    assert runtime_config.sampling.n_steps == 3
    assert runtime_config.sampling.burn_in == 1
    assert runtime_config.integration.normalization_samples == 128
    assert runtime_config.output.run_label == "synthetic"
    assert runtime_config.output.root_dir == synthetic_config_path.parent / "outputs"
    assert runtime_config.runtime.parallel_strategy == "auto"
    assert runtime_config.runtime.num_threads == 0
    assert runtime_config.runtime.reserve_cores == 2
    assert runtime_config.runtime.progress_summary_every == 1
    assert runtime_config.runtime.show_stage_timing is True
    assert runtime_config.fp_prior.enabled is False
    assert runtime_config.data.inference_dataset_path is not None
    assert runtime_config.data.sigma_table_path is None
    assert runtime_config.parameter_schema.prior_bounds[0] == pytest.approx((9.0, 12.0))


def test_load_runtime_config_builds_h_unit_mass_definition(synthetic_config_path: Path) -> None:
    """The h-units config surface should expose h-dependent labels and pivots."""

    runtime_config = load_runtime_config(synthetic_config_path)

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


def test_sonnenfeld_repository_configs_load_with_distinct_unit_contracts() -> None:
    """
    The checked-in Sonnenfeld configs must keep paper-native and h-unit runs separate.

    This is a regression test for a subtle scientific-contract failure: a single
    YAML file previously mixed the paper-native model name with h-unit data and
    old h-suffixed parameter names.  Loading the real config files here forces
    each file to declare one coherent model, unit convention, mass definition,
    parameter surface, FP-prior choice, and dataset naming contract.
    """

    paper_config = load_runtime_config(CONFIG_DIR / "sonnenfeld2024_slacs.yaml")
    hunit_config = load_runtime_config(CONFIG_DIR / "sonnenfeld2024_slacs_hunit.yaml")

    assert paper_config.profile.name == "devauc"
    assert paper_config.model.name == "sonnenfeld2024_slacs"
    assert paper_config.unit_convention == LEGACY_FIXED_KPC
    assert paper_config.mass_definition == get_mass_definition(5, unit_convention=LEGACY_FIXED_KPC)
    assert paper_config.mass_definition.label == "m5"
    assert paper_config.parameter_schema.public_parameter_names == SONNENFELD_PUBLIC_PARAMETERS
    assert paper_config.fp_prior.enabled is True
    assert paper_config.data.inference_dataset_path is not None
    assert paper_config.data.inference_dataset_path.name == (
        "inference_dataset_sonnenfeld2024_slacs_m5_fixed_v1.hdf5"
    )

    assert hunit_config.profile.name == "devauc"
    assert hunit_config.model.name == "sonnenfeld2024_slacs_hunit"
    assert hunit_config.unit_convention == H_UNITS_V1
    assert hunit_config.mass_definition == get_mass_definition(5, unit_convention=H_UNITS_V1)
    assert hunit_config.mass_definition.label == "m5_hinvkpc"
    assert hunit_config.parameter_schema.public_parameter_names == SONNENFELD_PUBLIC_PARAMETERS
    assert hunit_config.fp_prior.enabled is True
    assert hunit_config.data.inference_dataset_path is not None
    assert hunit_config.data.inference_dataset_path.name == (
        "inference_dataset_sonnenfeld2024_slacs_m5_hunits_v1.hdf5"
    )

    paper_initial_center = paper_config.sampling.initial_center.to_public_dict()
    hunit_initial_center = hunit_config.sampling.initial_center.to_public_dict()
    assert paper_initial_center["mu5_0"] == pytest.approx(SONNENFELD_PAPER_MU5_0)
    assert hunit_initial_center["mu5_0"] == pytest.approx(
        SONNENFELD_PAPER_MU5_0 + np.log10(hunit_config.h_ref)
    )


def test_sonnenfeld_sigma_star_gamma_configs_load_with_distinct_unit_contracts() -> None:
    """
    The sigma-star-gamma peer model should mirror Sonnenfeld's unit split.

    These checked-in YAML files are part of the model contract: the paper-native
    and h-unit variants use distinct registry names and datasets while sharing
    the same 11D sigma-star gamma parameter surface.
    """

    paper_config = load_runtime_config(CONFIG_DIR / "sonnenfeld2024_slacs_sigma_star_gamma.yaml")
    hunit_config = load_runtime_config(CONFIG_DIR / "sonnenfeld2024_slacs_sigma_star_gamma_hunit.yaml")

    assert paper_config.profile.name == "devauc"
    assert paper_config.model.name == "sonnenfeld2024_slacs_sigma_star_gamma"
    assert paper_config.unit_convention == LEGACY_FIXED_KPC
    assert paper_config.mass_definition == get_mass_definition(5, unit_convention=LEGACY_FIXED_KPC)
    assert paper_config.mass_definition.label == "m5"
    assert paper_config.parameter_schema.public_parameter_names == SONNENFELD_SIGMA_STAR_PUBLIC_PARAMETERS
    assert paper_config.parameter_schema.n_dim == 11
    assert paper_config.fp_prior.enabled is True
    assert paper_config.data.inference_dataset_path is not None
    assert paper_config.data.inference_dataset_path.name == (
        "inference_dataset_sonnenfeld2024_slacs_m5_fixed_v1.hdf5"
    )

    assert hunit_config.profile.name == "devauc"
    assert hunit_config.model.name == "sonnenfeld2024_slacs_sigma_star_gamma_hunit"
    assert hunit_config.unit_convention == H_UNITS_V1
    assert hunit_config.mass_definition == get_mass_definition(5, unit_convention=H_UNITS_V1)
    assert hunit_config.mass_definition.label == "m5_hinvkpc"
    assert hunit_config.parameter_schema.public_parameter_names == SONNENFELD_SIGMA_STAR_PUBLIC_PARAMETERS
    assert hunit_config.parameter_schema.n_dim == 11
    assert hunit_config.fp_prior.enabled is True
    assert hunit_config.data.inference_dataset_path is not None
    assert hunit_config.data.inference_dataset_path.name == (
        "inference_dataset_sonnenfeld2024_slacs_m5_hunits_v1.hdf5"
    )

    assert "beta_gamma" not in paper_config.parameter_schema.public_parameter_names
    assert "xi_gamma" not in paper_config.parameter_schema.public_parameter_names
    assert "beta_sigma_star_gamma" in paper_config.parameter_schema.public_parameter_names

    paper_initial_center = paper_config.sampling.initial_center.to_public_dict()
    hunit_initial_center = hunit_config.sampling.initial_center.to_public_dict()
    assert paper_initial_center["mu5_0"] == pytest.approx(SONNENFELD_PAPER_MU5_0)
    assert hunit_initial_center["mu5_0"] == pytest.approx(
        SONNENFELD_PAPER_MU5_0 + np.log10(hunit_config.h_ref)
    )


def test_load_runtime_config_rejects_raw_sigma_table_path(
    synthetic_config_path: Path,
    synthetic_sersic_sigma_table_file: Path,
) -> None:
    """Sigma tables now belong inside the prepared canonical dataset."""

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    payload["data"]["sigma_table_path"] = str(synthetic_sersic_sigma_table_file)
    payload["fp_prior"] = {"enabled": True}
    raw_sigma_table_path = synthetic_config_path.parent / "raw_sigma_table_path.yaml"
    raw_sigma_table_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sigma_table_path"):
        load_runtime_config(raw_sigma_table_path)


def test_load_runtime_config_builds_fp_prior_config_when_enabled(
    synthetic_fp_prior_config_path: Path,
) -> None:
    """CMASS should use model-owned 2026-04-29 FP defaults when enabled."""

    runtime_config = load_runtime_config(synthetic_fp_prior_config_path)
    expected_defaults = CMASS_FP_PRIOR_DEFAULTS_20260429

    assert runtime_config.fp_prior.enabled is True
    assert runtime_config.data.inference_dataset_path is not None
    assert runtime_config.data.sigma_table_path is None
    assert runtime_config.fp_prior.fit_mstar_min == pytest.approx(expected_defaults.fit_mstar_min)
    assert runtime_config.fp_prior.pivot_mstar == pytest.approx(expected_defaults.pivot_mstar)
    assert runtime_config.fp_prior.fiducial_scatter == pytest.approx(expected_defaults.fiducial_scatter)
    assert runtime_config.fp_prior.scatter_error == pytest.approx(expected_defaults.scatter_error)
    assert runtime_config.fp_prior.mu_v_prior == pytest.approx(expected_defaults.mu_v_prior, abs=1.0e-6)
    assert runtime_config.fp_prior.mu_v_error == pytest.approx(expected_defaults.mu_v_error)
    assert runtime_config.fp_prior.beta_v_prior == pytest.approx(expected_defaults.beta_v_prior, abs=1.0e-6)
    assert runtime_config.fp_prior.beta_v_error == pytest.approx(expected_defaults.beta_v_error)


def test_load_runtime_config_fp_prior_overrides_model_defaults(
    synthetic_fp_prior_config_path: Path,
) -> None:
    """Explicit YAML FP-prior values should override model-owned defaults."""

    payload = yaml.safe_load(synthetic_fp_prior_config_path.read_text(encoding="utf-8"))
    payload["fp_prior"] = {
        "enabled": True,
        "fit_mstar_min": 10.9,
        "pivot_mstar": 11.2,
        "fiducial_scatter": 0.051,
        "scatter_error": 0.004,
        "mu_v_prior": 2.22,
        "mu_v_error": 0.02,
        "beta_v_prior": 0.19,
        "beta_v_error": 0.015,
    }
    config_path = synthetic_fp_prior_config_path.parent / "fp_prior_override.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    runtime_config = load_runtime_config(config_path)

    assert runtime_config.fp_prior.enabled is True
    assert runtime_config.fp_prior.fit_mstar_min == pytest.approx(10.9)
    assert runtime_config.fp_prior.pivot_mstar == pytest.approx(11.2)
    assert runtime_config.fp_prior.fiducial_scatter == pytest.approx(0.051)
    assert runtime_config.fp_prior.scatter_error == pytest.approx(0.004)
    assert runtime_config.fp_prior.mu_v_prior == pytest.approx(2.22)
    assert runtime_config.fp_prior.mu_v_error == pytest.approx(0.02)
    assert runtime_config.fp_prior.beta_v_prior == pytest.approx(0.19)
    assert runtime_config.fp_prior.beta_v_error == pytest.approx(0.015)


def test_load_runtime_config_rejects_model_components(synthetic_config_path: Path) -> None:
    """The concrete-model config surface should reject component switches."""

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    payload["model"]["components"] = {
        "mass_definition": "m5_hinvkpc",
        "gamma_distribution": "sigma_star_dependent",
    }
    config_path = synthetic_config_path.parent / "component_switch.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="model.components"):
        load_runtime_config(config_path)


def test_load_runtime_config_requires_explicit_cosmology_section(synthetic_config_path: Path) -> None:
    """The astropy migration keeps requiring a dedicated cosmology section."""

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    payload.pop("cosmology")
    config_path = synthetic_config_path.parent / "missing_cosmology.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(KeyError, match="Missing required config section: cosmology"):
        load_runtime_config(config_path)


def test_load_runtime_config_requires_explicit_box_prior_section(synthetic_config_path: Path) -> None:
    """Fresh source configs must declare the full public-name box prior."""

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    payload.pop("box_prior")
    config_path = synthetic_config_path.parent / "missing_box_prior.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(KeyError, match="Missing required config section: box_prior"):
        load_runtime_config(config_path)


def test_load_runtime_config_rejects_run_snapshot_missing_box_prior(synthetic_config_path: Path) -> None:
    """Run snapshots use the same explicit box-prior contract as source configs."""

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    payload.pop("box_prior")
    snapshot_path = synthetic_config_path.parent / "config_snapshot.yaml"
    snapshot_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(KeyError, match="Missing required config section: box_prior"):
        load_runtime_config(snapshot_path)


def test_load_runtime_config_rejects_incomplete_box_prior_mapping(synthetic_config_path: Path) -> None:
    """Omitting one sampled parameter from `box_prior` should fail clearly."""

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    payload["box_prior"].pop("theta0")
    config_path = synthetic_config_path.parent / "incomplete_box_prior.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="theta0"):
        load_runtime_config(config_path)


def test_load_runtime_config_rejects_legacy_gamma_slope_names(synthetic_config_path: Path) -> None:
    """The fixed CMASS model should reject removed gamma slope names."""

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    payload["sampling"]["initial_center"]["beta_gamma"] = 0.1
    config_path = synthetic_config_path.parent / "legacy_gamma_slope.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="beta_gamma"):
        load_runtime_config(config_path)


def test_load_runtime_config_rejects_initial_center_outside_box_prior(
    synthetic_config_path: Path,
) -> None:
    """The configured initial center must already satisfy the explicit bounds."""

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    payload["box_prior"]["mu5h_0"] = [9.0, 11.0]
    invalid_center_path = synthetic_config_path.parent / "invalid_initial_center_bounds.yaml"
    invalid_center_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mu5h_0"):
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
    assert devauc.mass_function_loc == pytest.approx(DEVAUC_PROFILE_CONSTANTS.mass_function_loc)
    assert devauc.mu_r0 == pytest.approx(DEVAUC_PROFILE_CONSTANTS.mu_r0)
    assert devauc.beta_r == pytest.approx(DEVAUC_PROFILE_CONSTANTS.beta_r)
    assert devauc.sigma_r == pytest.approx(DEVAUC_PROFILE_CONSTANTS.sigma_r)

    assert sersic.fixed_n is None
    assert sersic.uses_observed_n_in_likelihood is True
    assert sersic.observation_field_aliases["stellar_mass"] == ("logmchab",)
    assert sersic.observation_field_aliases["effective_radius_arcsec"] == ("re_arcsec",)
    assert sersic.mass_function_loc == pytest.approx(SERSIC_PROFILE_CONSTANTS.mass_function_loc)
    assert sersic.mu_r0 == pytest.approx(SERSIC_PROFILE_CONSTANTS.mu_r0)
    assert sersic.beta_r == pytest.approx(SERSIC_PROFILE_CONSTANTS.beta_r)
    assert sersic.sigma_r == pytest.approx(SERSIC_PROFILE_CONSTANTS.sigma_r)
    assert sersic.nu_r == pytest.approx(SERSIC_PROFILE_CONSTANTS.nu_r)


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
    synthetic_hunit_sigma_bundle_without_within_re_file: Path,
    synthetic_fp_prior_config_path: Path,
    synthetic_hunit_observation_file: Path,
    synthetic_cross_section_file: Path,
) -> None:
    """
    FP prior should fail fast when the configured sigma bundle lacks within-Re data.

    The scientific contract now requires effective-radius sigma for every
    FP-enabled inference run. A bundle that only carries observation-flavor
    leaves must therefore be rejected immediately instead of silently falling
    back to slit or BOSS apertures.
    """

    from dataclasses import replace

    from cmass_lens_inference.compiled_context import build_compiled_context
    from cmass_lens_inference.config import load_runtime_config
    from cmass_lens_inference.types import DataConfig

    runtime_config = load_runtime_config(synthetic_fp_prior_config_path)
    legacy_oracle_config = replace(
        runtime_config,
        data=DataConfig(
            observation_path=synthetic_hunit_observation_file,
            cross_section_path=synthetic_cross_section_file,
            sigma_table_path=synthetic_hunit_sigma_bundle_without_within_re_file,
        ),
    )
    with pytest.raises(ValueError, match="within_re"):
        build_compiled_context(legacy_oracle_config)


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
