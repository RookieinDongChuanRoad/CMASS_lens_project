"""
Pytest shared fixtures for the CMASS lens inference project.

These fixtures intentionally create tiny synthetic HDF5 files and YAML
configurations so that the test suite can exercise the project skeleton,
data-loading compatibility, and output-management behavior without needing
to run the full scientific inference on the real datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml


# The project uses a `src/` layout. Adding that directory to `sys.path` here
# keeps the test harness simple and avoids requiring an editable install just
# to exercise the package during local development.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _write_synthetic_sigma_table(
    path: Path,
    *,
    profile_name: str,
    mass_definition_label: str,
    mass_radius_kpc: float,
) -> Path:
    """
    Write a tiny sigma-unit table using the shared HDF5 schema.

    The fixtures only need enough structure to exercise configuration and
    schema-validation logic, so the numerical values are deliberately simple
    and monotonic.
    """

    gamma_axis = np.linspace(1.2, 2.8, 5)
    zd_axis = np.linspace(0.43, 0.82, 4)
    log_re_kpc_axis = np.linspace(0.45, 1.20, 3)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("profile_name", data=np.bytes_(profile_name))
        handle.create_dataset("gamma_axis", data=gamma_axis)
        handle.create_dataset("zd_axis", data=zd_axis)
        handle.create_dataset("log_re_kpc_axis", data=log_re_kpc_axis)
        handle.attrs["schema_version"] = "sigma_unit_hdf5_v1"
        handle.attrs["mass_definition_label"] = mass_definition_label
        handle.attrs["mass_radius_kpc"] = float(mass_radius_kpc)
        handle.attrs["units"] = f"km2 s-2 per 10**{mass_definition_label}"

        if profile_name == "sersic":
            n_axis = np.linspace(2.5, 10.5, 4)
            handle.create_dataset("n_axis", data=n_axis)
            values = np.linspace(
                0.4,
                1.0,
                gamma_axis.size * zd_axis.size * log_re_kpc_axis.size * n_axis.size,
            ).reshape(gamma_axis.size, zd_axis.size, log_re_kpc_axis.size, n_axis.size)
        else:
            values = np.linspace(
                0.4,
                1.0,
                gamma_axis.size * zd_axis.size * log_re_kpc_axis.size,
            ).reshape(gamma_axis.size, zd_axis.size, log_re_kpc_axis.size)

        handle.create_dataset("s_unit_grid", data=values)

    return path


def _write_synthetic_sigma_bundle(
    path: Path,
    *,
    profile_name: str,
    mass_definition_label: str = "m5",
    mass_radius_kpc: float = 5.0,
    observation_flavor: str = "boss",
    seeing_fwhm_arcsec: float = 1.5,
) -> Path:
    """
    Write a tiny bundle-style sigma table for flavor-aware loader tests.

    The bundle fixture mirrors the production grouped schema closely enough to
    exercise mass-definition selection and BOSS aperture/seeing validation
    without depending on the heavy interpolation-grid builder.
    """

    gamma_axis = np.linspace(1.2, 2.8, 5)
    zd_axis = np.linspace(0.43, 0.82, 4)
    log_re_kpc_axis = np.linspace(0.45, 1.20, 3)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("profile_name", data=np.bytes_(profile_name))
        handle.attrs["schema_version"] = "sigma_unit_bundle_hdf5_v2"
        handle.attrs["quantity_name"] = "S_unit"

        slit_group = handle.create_group("slit")
        boss_group = handle.create_group("boss")
        target_group = boss_group if observation_flavor == "boss" else slit_group
        leaf = target_group.create_group(mass_definition_label)
        leaf.create_dataset("gamma_axis", data=gamma_axis)
        leaf.create_dataset("zd_axis", data=zd_axis)
        leaf.create_dataset("log_re_kpc_axis", data=log_re_kpc_axis)
        leaf.attrs["mass_definition_label"] = mass_definition_label
        leaf.attrs["mass_radius_kpc"] = float(mass_radius_kpc)
        leaf.attrs["units"] = f"km2 s-2 per 10**{mass_definition_label}"
        leaf.attrs["observation_flavor"] = observation_flavor
        if observation_flavor == "boss":
            values = np.linspace(
                0.5,
                1.1,
                gamma_axis.size * zd_axis.size * log_re_kpc_axis.size,
            ).reshape(gamma_axis.size, zd_axis.size, log_re_kpc_axis.size)
            leaf.attrs["aperture_shape"] = "circular"
            leaf.attrs["aperture_radius_arcsec"] = 1.0
        else:
            values = np.linspace(
                0.4,
                1.0,
                gamma_axis.size * zd_axis.size * log_re_kpc_axis.size,
            ).reshape(gamma_axis.size, zd_axis.size, log_re_kpc_axis.size)
            leaf.attrs["aperture_shape"] = "rectangular"
            leaf.attrs["aperture_width_arcsec"] = 1.6
            leaf.attrs["aperture_height_arcsec"] = 0.9

        if profile_name == "sersic":
            n_axis = np.linspace(2.5, 10.5, 4)
            leaf.create_dataset("n_axis", data=n_axis)
            values = np.repeat(values[..., None], n_axis.size, axis=3)

        leaf.create_dataset("s_unit_grid", data=values)
        leaf.attrs["seeing_fwhm_arcsec"] = float(seeing_fwhm_arcsec)

    return path


@pytest.fixture
def synthetic_observation_file(tmp_path: Path) -> Path:
    """
    Create a one-lens HDF5 observation file that mirrors the required schema.

    The synthetic lens is deliberately simple:
    - 17-point grids exist for `gamma`, `m5`, and the Jacobian term.
    - velocity-dispersion information is present so tests can cover the
      `num_sigma=1` branch and `s2_grid` loading.
    - attribute names follow the generic schema used by the `sersic` profile.
    """

    path = tmp_path / "synthetic_observations.hdf5"
    gamma_grid = np.linspace(1.3, 2.7, 17)
    m5_grid = np.linspace(11.6, 10.8, 17)
    dm5_dthetaein_grid = np.linspace(-2.0, -1.0, 17)
    s2_grid = np.linspace(0.8, 1.2, 17)

    with h5py.File(path, "w") as handle:
        group = handle.create_group("lens-0001")
        group.attrs["zd"] = 0.55
        group.attrs["zs"] = 1.75
        group.attrs["logmchab"] = 11.3
        group.attrs["logmchab_err"] = 0.05
        group.attrs["nser"] = 4.2
        group.attrs["re_arcsec"] = 1.1
        group.attrs["rein_arcsec"] = 1.3
        group.attrs["num_sigma"] = 1
        # Keep the synthetic dispersion observations on the same rough scale as
        # `sqrt(s2 * 10**m5)` so the monolithic likelihood kernel has a
        # numerically meaningful `num_sigma=1` branch to integrate.
        group.attrs["sigma"] = 320000.0
        group.attrs["sigma_err"] = 20000.0
        group.create_dataset("gamma_grid", data=gamma_grid)
        group.create_dataset("m5_grid", data=m5_grid)
        group.create_dataset("dm5_dthetaein_grid", data=dm5_dthetaein_grid)
        group.create_dataset("s2_grid", data=s2_grid)

    return path


@pytest.fixture
def synthetic_devauc_observation_file(tmp_path: Path) -> Path:
    """
    Create a one-lens observation file that exercises the devauc field aliases.

    This file adds `logmchab_deV` and `reff_deV`, which the `devauc` profile
    must prefer over the generic stellar-mass and effective-radius fields.
    """

    path = tmp_path / "synthetic_devauc_observations.hdf5"
    gamma_grid = np.linspace(1.25, 2.65, 17)
    m5_grid = np.linspace(11.5, 10.7, 17)
    dm5_dthetaein_grid = np.linspace(-1.8, -1.1, 17)

    with h5py.File(path, "w") as handle:
        group = handle.create_group("lens-devauc")
        group.attrs["zd"] = 0.58
        group.attrs["zs"] = 1.9
        group.attrs["logmchab"] = 10.9
        group.attrs["logmchab_deV"] = 11.1
        group.attrs["logmchab_err"] = 0.04
        group.attrs["nser"] = 3.1
        group.attrs["re_arcsec"] = 0.9
        group.attrs["reff_deV"] = 1.4
        group.attrs["rein_arcsec"] = 1.0
        group.attrs["num_sigma"] = 0
        group.create_dataset("gamma_grid", data=gamma_grid)
        group.create_dataset("m5_grid", data=m5_grid)
        group.create_dataset("dm5_dthetaein_grid", data=dm5_dthetaein_grid)

    return path


@pytest.fixture
def synthetic_namespaced_observation_file(tmp_path: Path) -> Path:
    """
    Create a one-lens observation file using only the new namespaced schema.

    Why this fixture exists:
    - the migration keeps the top-level lens grouping stable while moving the
      mass-dependent grids under `mass_definitions/<label>/`
    - the reader must support files that no longer carry legacy root-level
      `m5_*` datasets once the transition is complete
    - tests need one fixture that proves the reader prefers the selected
      namespaced subgroup rather than silently falling back to old aliases
    """

    path = tmp_path / "synthetic_namespaced_observations.hdf5"
    gamma_grid = np.linspace(1.3, 2.7, 17)
    m5_grid = np.linspace(11.55, 10.75, 17)
    m10_grid = np.linspace(11.75, 10.95, 17)
    dmass_grid = np.linspace(-1.9, -1.1, 17)
    s2_m5_grid = np.linspace(0.7, 1.0, 17)
    s2_m10_grid = np.linspace(0.45, 0.75, 17)

    with h5py.File(path, "w") as handle:
        group = handle.create_group("lens-namespaced")
        group.attrs["zd"] = 0.57
        group.attrs["zs"] = 1.92
        group.attrs["logmchab"] = 11.18
        group.attrs["logmchab_err"] = 0.06
        group.attrs["nser"] = 3.9
        group.attrs["re_arcsec"] = 1.05
        group.attrs["rein_arcsec"] = 1.22
        group.attrs["num_sigma"] = 1
        group.attrs["sigma"] = 290000.0
        group.attrs["sigma_err"] = 18000.0
        group.create_dataset("gamma_grid", data=gamma_grid)

        mass_root = group.create_group("mass_definitions")
        m5_group = mass_root.create_group("m5")
        m5_group.create_dataset("mass_grid", data=m5_grid)
        m5_group.create_dataset("dmass_dthetaein_grid", data=dmass_grid)
        m5_group.create_dataset("s2_grid", data=s2_m5_grid)

        m10_group = mass_root.create_group("m10")
        m10_group.create_dataset("mass_grid", data=m10_grid)
        m10_group.create_dataset("dmass_dthetaein_grid", data=dmass_grid)
        m10_group.create_dataset("s2_grid", data=s2_m10_grid)

    return path


@pytest.fixture
def synthetic_cross_section_file(tmp_path: Path) -> Path:
    """
    Create a cross-section HDF5 file using the alias field names observed in
    the real project data (`gamma_grids`, `cs_over_theta_ein_grid`).
    """

    path = tmp_path / "synthetic_cs_grid_power.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("compressed_grids")
        group.create_dataset("gamma_grids", data=np.linspace(1.2, 2.8, 25))
        group.create_dataset("cs_over_theta_ein_grid", data=np.linspace(0.6, 1.4, 25))
    return path


@pytest.fixture
def synthetic_sersic_sigma_table_file(tmp_path: Path) -> Path:
    """Create a tiny `sersic + m5` sigma-unit table for FP-prior tests."""

    return _write_synthetic_sigma_table(
        tmp_path / "synthetic_sersic_sigma_table.h5",
        profile_name="sersic",
        mass_definition_label="m5",
        mass_radius_kpc=5.0,
    )


@pytest.fixture
def synthetic_sersic_m10_sigma_table_file(tmp_path: Path) -> Path:
    """Create a tiny `sersic + m10` sigma-unit table for mismatch tests."""

    return _write_synthetic_sigma_table(
        tmp_path / "synthetic_sersic_sigma_table_m10.h5",
        profile_name="sersic",
        mass_definition_label="m10",
        mass_radius_kpc=10.0,
    )


@pytest.fixture
def synthetic_devauc_sigma_table_file(tmp_path: Path) -> Path:
    """Create a tiny `devauc + m5` sigma-unit table for mismatch tests."""

    return _write_synthetic_sigma_table(
        tmp_path / "synthetic_devauc_sigma_table.h5",
        profile_name="devauc",
        mass_definition_label="m5",
        mass_radius_kpc=5.0,
    )


@pytest.fixture
def synthetic_boss_observation_file(tmp_path: Path) -> Path:
    """Create a one-lens BOSS observation file with the canonical 1.5 arcsec seeing."""

    path = tmp_path / "synthetic_boss_observations.hdf5"
    gamma_grid = np.linspace(1.25, 2.65, 17)
    mass_grid = np.linspace(11.5, 10.7, 17)
    dmass_grid = np.linspace(-1.8, -1.1, 17)

    with h5py.File(path, "w") as handle:
        group = handle.create_group("lens-boss")
        group.attrs["zd"] = 0.58
        group.attrs["zs"] = 1.9
        group.attrs["logmchab"] = 10.9
        group.attrs["logmchab_deV"] = 11.1
        group.attrs["logmchab_err"] = 0.04
        group.attrs["nser"] = 3.1
        group.attrs["re_arcsec"] = 0.9
        group.attrs["reff_deV"] = 1.4
        group.attrs["rein_arcsec"] = 1.0
        group.attrs["num_sigma"] = 1
        group.attrs["sigma"] = np.asarray([290000.0], dtype=float)
        group.attrs["sigma_err"] = np.asarray([18000.0], dtype=float)
        group.attrs["aperture_shape"] = "circular"
        group.attrs["aperture_radius_arcsec"] = 1.0
        group.attrs["seeing_fwhm_arcsec"] = 1.5
        group.create_dataset("gamma_grid", data=gamma_grid)
        group.create_dataset("m5_grid", data=mass_grid)
        group.create_dataset("dm5_dthetaein_grid", data=dmass_grid)

    return path


@pytest.fixture
def synthetic_boss_sigma_bundle_file(tmp_path: Path) -> Path:
    """Create a tiny BOSS bundle leaf with the canonical 1.5 arcsec seeing."""

    return _write_synthetic_sigma_bundle(
        tmp_path / "synthetic_devauc_sigma_bundle.h5",
        profile_name="devauc",
        observation_flavor="boss",
        seeing_fwhm_arcsec=1.5,
    )


@pytest.fixture
def synthetic_bad_boss_sigma_bundle_file(tmp_path: Path) -> Path:
    """Create a BOSS bundle leaf that still carries the retired 0.9 arcsec seeing."""

    return _write_synthetic_sigma_bundle(
        tmp_path / "synthetic_bad_boss_sigma_bundle.h5",
        profile_name="devauc",
        observation_flavor="boss",
        seeing_fwhm_arcsec=0.9,
    )


@pytest.fixture
def synthetic_config_path(
    tmp_path: Path,
    synthetic_observation_file: Path,
    synthetic_cross_section_file: Path,
) -> Path:
    """
    Create a YAML configuration file for the `sersic` profile.

    The output root intentionally points at the temporary directory so the
    tests can assert against the generated run tree without touching the
    user's long-lived outputs directory.
    """

    path = tmp_path / "synthetic_sersic.yaml"
    config = {
        "profile": {"name": "sersic"},
        "mass_definition": {"enclosed_radius_kpc": 5},
        "gamma_model": {"mode": "dependent"},
        "data": {
            "observation_path": str(synthetic_observation_file),
            "cross_section_path": str(synthetic_cross_section_file),
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
            "initial_jitter_scale": 1.0e-3,
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
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def synthetic_m10_config_path(
    tmp_path: Path,
    synthetic_observation_file: Path,
    synthetic_cross_section_file: Path,
) -> Path:
    """
    Create a YAML configuration file for the `m10` public naming surface.

    This fixture intentionally uses the `mu10_*` family so tests can prove the
    loader maps definition-specific public names onto the same internal model.
    """

    path = tmp_path / "synthetic_sersic_m10.yaml"
    config = {
        "profile": {"name": "sersic"},
        "mass_definition": {"enclosed_radius_kpc": 10},
        "gamma_model": {"mode": "dependent"},
        "data": {
            "observation_path": str(synthetic_observation_file),
            "cross_section_path": str(synthetic_cross_section_file),
        },
        "sampling": {
            "n_walkers": 24,
            "n_steps": 3,
            "warmup": 1,
            "random_seed": 7,
            "initial_center": {
                "mu10_0": 11.42,
                "beta10": 0.49,
                "xi10": -0.21,
                "sigma10": 0.08,
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
            "run_label": "synthetic-m10",
            "overwrite_latest": True,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def synthetic_independent_config_path(
    tmp_path: Path,
    synthetic_observation_file: Path,
    synthetic_cross_section_file: Path,
) -> Path:
    """
    Create a YAML configuration file for the `independent` gamma mode.

    This fixture intentionally omits `beta_gamma` and `xi_gamma` from the
    public initial center because the independent mode should expose only the
    gamma mean and scatter parameters in the sampled vector.
    """

    path = tmp_path / "synthetic_sersic_gamma_independent.yaml"
    config = {
        "profile": {"name": "sersic"},
        "mass_definition": {"enclosed_radius_kpc": 5},
        "gamma_model": {"mode": "independent"},
        "data": {
            "observation_path": str(synthetic_observation_file),
            "cross_section_path": str(synthetic_cross_section_file),
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
            "run_label": "synthetic-gamma-independent",
            "overwrite_latest": True,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def synthetic_sigma_star_dependent_config_path(
    tmp_path: Path,
    synthetic_observation_file: Path,
    synthetic_cross_section_file: Path,
) -> Path:
    """
    Create a YAML configuration file for the `sigma_star_dependent` gamma mode.

    This fixture locks the third gamma-parameterization contract: the sampled
    vector should keep the four mass hyper-parameters, replace the removed
    `logM* + logRe` gamma slopes with one `Sigma_*` slope, and preserve the
    shared tail parameters unchanged.
    """

    path = tmp_path / "synthetic_sersic_gamma_sigma_star.yaml"
    config = {
        "profile": {"name": "sersic"},
        "mass_definition": {"enclosed_radius_kpc": 5},
        "gamma_model": {"mode": "sigma_star_dependent"},
        "data": {
            "observation_path": str(synthetic_observation_file),
            "cross_section_path": str(synthetic_cross_section_file),
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
                "beta_sigma_star_gamma": 0.24,
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
            "run_label": "synthetic-gamma-sigma-star",
            "overwrite_latest": True,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def synthetic_fp_prior_config_path(
    synthetic_config_path: Path,
    synthetic_sersic_sigma_table_file: Path,
) -> Path:
    """Create a `sersic + m5` config with the optional FP prior enabled."""

    payload = yaml.safe_load(synthetic_config_path.read_text(encoding="utf-8"))
    payload["data"]["sigma_table_path"] = str(synthetic_sersic_sigma_table_file)
    payload["fp_prior"] = {"enabled": True}
    path = synthetic_config_path.parent / "synthetic_sersic_fp_prior.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def synthetic_fp_prior_independent_config_path(
    synthetic_independent_config_path: Path,
    synthetic_sersic_sigma_table_file: Path,
) -> Path:
    """Create an `independent` gamma-mode config with FP prior enabled."""

    payload = yaml.safe_load(synthetic_independent_config_path.read_text(encoding="utf-8"))
    payload["data"]["sigma_table_path"] = str(synthetic_sersic_sigma_table_file)
    payload["fp_prior"] = {"enabled": True}
    path = synthetic_independent_config_path.parent / "synthetic_sersic_fp_prior_independent.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def synthetic_fp_prior_sigma_star_zero_slope_config_path(
    synthetic_sigma_star_dependent_config_path: Path,
    synthetic_sersic_sigma_table_file: Path,
) -> Path:
    """Create a zero-slope sigma-star config with the FP prior enabled."""

    payload = yaml.safe_load(synthetic_sigma_star_dependent_config_path.read_text(encoding="utf-8"))
    payload["sampling"]["initial_center"]["beta_sigma_star_gamma"] = 0.0
    payload["data"]["sigma_table_path"] = str(synthetic_sersic_sigma_table_file)
    payload["fp_prior"] = {"enabled": True}
    path = synthetic_sigma_star_dependent_config_path.parent / "synthetic_sersic_fp_prior_sigma_star_zero.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def synthetic_devauc_fp_prior_config_path(
    tmp_path: Path,
    synthetic_devauc_observation_file: Path,
    synthetic_cross_section_file: Path,
    synthetic_devauc_sigma_table_file: Path,
) -> Path:
    """Create a `devauc + m5` config with the optional FP prior enabled."""

    path = tmp_path / "synthetic_devauc_fp_prior.yaml"
    config = {
        "profile": {"name": "devauc"},
        "mass_definition": {"enclosed_radius_kpc": 5},
        "gamma_model": {"mode": "dependent"},
        "fp_prior": {"enabled": True},
        "data": {
            "observation_path": str(synthetic_devauc_observation_file),
            "cross_section_path": str(synthetic_cross_section_file),
            "sigma_table_path": str(synthetic_devauc_sigma_table_file),
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
            "initial_jitter_scale": 1.0e-3,
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
            "run_label": "synthetic-devauc-fp",
            "overwrite_latest": True,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
