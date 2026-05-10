"""Fixed paper constants for the Sonnenfeld 2024 SLACS model."""

from __future__ import annotations

import numpy as np


MSTAR_PIVOT_PHYSICAL = 11.3
MBAR_PHYSICAL = 11.06
PARENT_ALPHA = -1.207
SIZE_MU0_PHYSICAL = 7.55
SIZE_MU1_PHYSICAL = -1.84
SIZE_MU2_PHYSICAL = 0.11
SIZE_SCATTER = 0.112
TRUNCATION_MASS_POLYNOMIAL_COEFFICIENTS = np.asarray(
    [9.388, 7.855, 48.34, -312.5, 535.7, -274.2],
    dtype=np.float64,
)
TRUNCATION_MASS_SCATTER = 0.0007
GAMMA_TRUNC_LOW = 1.2
GAMMA_TRUNC_HIGH = 2.8
PARENT_ZD_MIN = 0.05
PARENT_ZD_MAX = 0.40
PARENT_MSTAR_MIN_PHYSICAL = 9.0
PARENT_MSTAR_MAX_PHYSICAL = 12.5
PARENT_MSTAR_MIN_OFFSET = -0.8
PARENT_MSTAR_MAX_OFFSET = 1.0
SOURCE_Z_MIN = 0.05
SOURCE_Z_MAX = 2.0
SOURCE_LENS_REDSHIFT_GAP = 0.05
SIGMA_PROXY_FRACTIONAL_SCATTER = 0.0625
FP_FIT_MSTAR_MIN_PHYSICAL = 11.0
FP_PIVOT_MSTAR_PHYSICAL = 11.3
FP_FIDUCIAL_SCATTER = 0.047
FP_SCATTER_ERROR = 0.008
FP_MU_V_ERROR = 0.03
FP_BETA_V_ERROR = 0.03


def shift_physical_mass_location_to_hunits(value: float, h_ref: float) -> float:
    """Convert a physical log-stellar-mass location to the h-units coordinate."""

    return float(value) + 2.0 * float(np.log10(h_ref))


def fp_parent_velocity_mean(log_mstar: float) -> float:
    """Return the reference parent-sample mass-velocity relation."""

    offset = float(log_mstar) - 11.0
    return 2.2577 + 0.3034 * offset - 0.0761 * offset**2


FP_MU_V_PRIOR = fp_parent_velocity_mean(FP_PIVOT_MSTAR_PHYSICAL)
FP_BETA_V_PRIOR = (
    fp_parent_velocity_mean(FP_PIVOT_MSTAR_PHYSICAL + 0.01)
    - fp_parent_velocity_mean(FP_PIVOT_MSTAR_PHYSICAL - 0.01)
) / 0.02


def active_size_relation_coefficients(
    *,
    h_ref: float,
    unit_convention: str,
) -> tuple[float, float, float]:
    """Return Equation 29 coefficients in the active size/mass coordinate."""

    if unit_convention == "legacy_fixed_kpc":
        return SIZE_MU0_PHYSICAL, SIZE_MU1_PHYSICAL, SIZE_MU2_PHYSICAL
    if unit_convention == "h_units_v1":
        log10_h = float(np.log10(h_ref))
        return (
            SIZE_MU0_PHYSICAL + log10_h * (1.0 - 2.0 * SIZE_MU1_PHYSICAL) + 4.0 * SIZE_MU2_PHYSICAL * log10_h**2,
            SIZE_MU1_PHYSICAL - 4.0 * SIZE_MU2_PHYSICAL * log10_h,
            SIZE_MU2_PHYSICAL,
        )
    raise ValueError(f"Unsupported unit_convention '{unit_convention}'.")


__all__ = [
    "GAMMA_TRUNC_HIGH",
    "GAMMA_TRUNC_LOW",
    "FP_BETA_V_ERROR",
    "FP_BETA_V_PRIOR",
    "FP_FIDUCIAL_SCATTER",
    "FP_FIT_MSTAR_MIN_PHYSICAL",
    "FP_MU_V_ERROR",
    "FP_MU_V_PRIOR",
    "FP_PIVOT_MSTAR_PHYSICAL",
    "FP_SCATTER_ERROR",
    "MBAR_PHYSICAL",
    "MSTAR_PIVOT_PHYSICAL",
    "PARENT_ALPHA",
    "PARENT_MSTAR_MAX_PHYSICAL",
    "PARENT_MSTAR_MAX_OFFSET",
    "PARENT_MSTAR_MIN_PHYSICAL",
    "PARENT_MSTAR_MIN_OFFSET",
    "PARENT_ZD_MAX",
    "PARENT_ZD_MIN",
    "SIGMA_PROXY_FRACTIONAL_SCATTER",
    "SIZE_MU0_PHYSICAL",
    "SIZE_MU1_PHYSICAL",
    "SIZE_MU2_PHYSICAL",
    "SIZE_SCATTER",
    "SOURCE_Z_MAX",
    "SOURCE_Z_MIN",
    "SOURCE_LENS_REDSHIFT_GAP",
    "TRUNCATION_MASS_POLYNOMIAL_COEFFICIENTS",
    "TRUNCATION_MASS_SCATTER",
    "active_size_relation_coefficients",
    "fp_parent_velocity_mean",
    "shift_physical_mass_location_to_hunits",
]
