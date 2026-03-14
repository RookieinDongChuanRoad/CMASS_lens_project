"""
Pytest shared fixtures for the CMASS lens inference project.

These fixtures intentionally create tiny synthetic HDF5 files and YAML
configurations so that the test suite can exercise the project skeleton,
data-loading compatibility, and output-management behavior without needing
to run the full scientific inference on the real datasets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml


# The standalone PPT package also uses a `src/` layout, but it is allowed to
# depend on the reusable inference core that still lives in
# `Bayesian_inference/src`. The test harness therefore exposes both roots so we
# can run the migrated package in-place without an installation step.
PPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PPT_SOURCE_ROOT = PPT_PROJECT_ROOT / "src"
INFERENCE_SOURCE_ROOT = Path("/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/src")

for source_root in (PPT_SOURCE_ROOT, INFERENCE_SOURCE_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

existing_pythonpath_entries = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
merged_pythonpath_entries: list[str] = []
for entry in (str(PPT_SOURCE_ROOT), str(INFERENCE_SOURCE_ROOT), *existing_pythonpath_entries):
    if entry not in merged_pythonpath_entries:
        merged_pythonpath_entries.append(entry)
os.environ["PYTHONPATH"] = os.pathsep.join(merged_pythonpath_entries)


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
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
