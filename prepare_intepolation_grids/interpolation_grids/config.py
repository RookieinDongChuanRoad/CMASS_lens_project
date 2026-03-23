"""Central configuration for interpolation-grid generation.

Why this module exists:
- The legacy implementation duplicated physical constants and field names across
  multiple standalone scripts.
- Centralizing them here makes the code easier to audit and reduces the chance
  of changing one branch while silently forgetting another.

The constants below are intentionally explicit because downstream scientific
code depends on their exact values and units.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from interpolation_grids.models import AperturePolicy


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIRECTORY = PROJECT_ROOT.parent / "data" / "raw"
EXTERNAL_DATA_DIRECTORY = PROJECT_ROOT.parent / "data" / "external"

# Canonical default inputs used by the main pipeline.
# The older `with_m5_grids` filenames remain readable as legacy inputs, but the
# public defaults should no longer bake one specific mass definition into the
# filename surface.
DEFAULT_INPUT_FILENAMES = (
    "observations_deV_with_mass_grids.hdf5",
    "observations_with_mass_grids_all.hdf5",
)

# The gamma grid is already present in the real files and in the reference
# scripts. We keep the exact range and sampling to preserve compatibility.
GAMMA_GRID = np.linspace(1.2, 2.8, 17, dtype=float)

# Dataset names are centralized because the spec text and the real files use
# slightly different spellings for the derivative grid.
GAMMA_DATASET_NAME = "gamma_grid"
M5_DATASET_NAME = "m5_grid"
DERIVATIVE_DATASET_NAME = "dm5_dthetaein_grid"
LEGACY_DERIVATIVE_DATASET_NAME = "dm5_dtheta_ein_grid"
S2_DATASET_NAME = "s2_grid"
MASS_DEFINITIONS_GROUP_NAME = "mass_definitions"
MASS_GRID_DATASET_NAME = "mass_grid"
MASS_DERIVATIVE_DATASET_NAME = "dmass_dthetaein_grid"
SUPPORTED_MASS_RADII_KPC = (5.0, 10.0)
MASS_DEFINITION_LABELS = {
    5.0: "m5",
    10.0: "m10",
}

# Aperture and seeing are expressed in arcsec first and converted per galaxy to
# physical kpc using the lens redshift. The slit and BOSS products now use
# distinct seeing contracts, so those values are carried as explicit
# flavor-specific constants rather than one shared global number.
DEFAULT_APERTURE_WIDTH_ARCSEC = 1.6
APERTURE_HEIGHT_ARCSEC = 0.9
DEFAULT_SLIT_SEEING_FWHM_ARCSEC = 0.9
DEFAULT_BOSS_SEEING_FWHM_ARCSEC = 1.5
DEFAULT_PRODUCTION_APERTURE_POLICY = AperturePolicy.rectangular(
    width_arcsec=DEFAULT_APERTURE_WIDTH_ARCSEC,
    height_arcsec=APERTURE_HEIGHT_ARCSEC,
    seeing_fwhm_arcsec=DEFAULT_SLIT_SEEING_FWHM_ARCSEC,
)
BOSS_CIRCULAR_APERTURE_POLICY = AperturePolicy.circular(
    radius_arcsec=1.0,
    seeing_fwhm_arcsec=DEFAULT_BOSS_SEEING_FWHM_ARCSEC,
)
SUPPORTED_OBSERVATION_FLAVORS = ("slit", "boss")
OBSERVATION_FLAVOR_APERTURE_POLICIES = {
    "slit": DEFAULT_PRODUCTION_APERTURE_POLICY,
    "boss": BOSS_CIRCULAR_APERTURE_POLICY,
}

# BOSS observation-file rebuild inputs and outputs.
BOSS_SUMMARY_FILENAME = "summary_table_deV.txt"
BOSS_OUTPUT_FILENAMES = {
    "devauc": "observations_deV_with_BOSS_mass_grids.hdf5",
    "sersic": "observations_with_BOSS_mass_grids_all.hdf5",
}

# The Jeans integration follows the legacy script and evaluates the 3D mass
# profile over six dex around the tracer scale radius.
DEFAULT_RADIAL_GRID_SIZE = 1001

# The derivative grid intentionally mirrors the dense numerical recipe from the
# legacy script because compatibility is currently more important than speed.
DEFAULT_DERIVATIVE_THETA_SAMPLES = 10_000

# Sigma-unit interpolation tables for posterior predictive checks.
# These grids intentionally follow the PPT requirements exactly so upstream and
# downstream code do not need to infer bounds from old files.
SIGMA_UNIT_SCHEMA_VERSION = "sigma_unit_hdf5_v1"
SIGMA_UNIT_BUNDLE_SCHEMA_VERSION = "sigma_unit_bundle_hdf5_v2"
SIGMA_UNIT_QUANTITY_NAME = "S_unit"
SIGMA_UNIT_UNITS = "km2 s-2 per 10**m5"
SIGMA_UNIT_PROFILE_FILENAMES = {
    ("devauc", 5.0): "jeans_deV_m5_grid.h5",
    ("devauc", 10.0): "jeans_deV_m10_grid.h5",
    ("sersic", 5.0): "jeans_sers_m5_grid.h5",
    ("sersic", 10.0): "jeans_sers_m10_grid.h5",
}
SIGMA_UNIT_BUNDLE_FILENAMES = {
    "devauc": "jeans_deV_sigma_bundle.h5",
    "sersic": "jeans_sers_sigma_bundle.h5",
}
SIGMA_UNIT_GAMMA_AXIS = GAMMA_GRID.copy()
SIGMA_UNIT_ZD_AXIS = np.linspace(0.43, 0.82, 21, dtype=float)
SIGMA_UNIT_DEVAUC_LOG_RE_KPC_AXIS = np.linspace(0.45, 1.20, 21, dtype=float)
SIGMA_UNIT_SERSIC_LOG_RE_KPC_AXIS = np.linspace(0.50, 1.40, 21, dtype=float)
SIGMA_UNIT_SERSIC_N_AXIS = np.linspace(2.5, 10.5, 21, dtype=float)


def mass_definition_label(radius_kpc: float) -> str:
    """Return the canonical public label for one supported mass radius."""

    return MASS_DEFINITION_LABELS[float(radius_kpc)]


def sigma_unit_units_for_radius(radius_kpc: float) -> str:
    """Return the sigma-unit table units string for one mass definition."""

    return f"km2 s-2 per 10**{mass_definition_label(radius_kpc)}"
