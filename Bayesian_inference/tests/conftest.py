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
