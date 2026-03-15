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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIRECTORY = PROJECT_ROOT.parent / "data" / "raw"
EXTERNAL_DATA_DIRECTORY = PROJECT_ROOT.parent / "data" / "external"

# Historical default inputs used by this project.
DEFAULT_INPUT_FILENAMES = (
    "observations_deV_with_m5_grids.hdf5",
    "observations_with_m5_grids_all.hdf5",
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
# physical kpc using the lens redshift. Production `s2_grid` generation now
# follows the updated business rule of a fixed 1.6 arcsec slit width, while
# legacy 0.8 arcsec behavior is preserved only in regression tests that
# reproduce the historical reference script.
DEFAULT_APERTURE_WIDTH_ARCSEC = 1.6
APERTURE_HEIGHT_ARCSEC = 0.9
SEEING_FWHM_ARCSEC = 0.9

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
SIGMA_UNIT_QUANTITY_NAME = "S_unit"
SIGMA_UNIT_UNITS = "km2 s-2 per 10**m5"
SIGMA_UNIT_PROFILE_FILENAMES = {
    ("devauc", 5.0): "jeans_deV_m5_grid.h5",
    ("devauc", 10.0): "jeans_deV_m10_grid.h5",
    ("sersic", 5.0): "jeans_sers_m5_grid.h5",
    ("sersic", 10.0): "jeans_sers_m10_grid.h5",
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
