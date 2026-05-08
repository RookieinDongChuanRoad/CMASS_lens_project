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
PARENT_ZD_MAX = 0.95
PARENT_MSTAR_MIN_OFFSET = -0.8
PARENT_MSTAR_MAX_OFFSET = 1.0
SIGMA_PROXY_FRACTIONAL_SCATTER = 0.0625


def shift_physical_mass_location_to_hunits(value: float, h_ref: float) -> float:
    """Convert a physical log-stellar-mass location to the h-units coordinate."""

    return float(value) + 2.0 * float(np.log10(h_ref))


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
    "MBAR_PHYSICAL",
    "MSTAR_PIVOT_PHYSICAL",
    "PARENT_ALPHA",
    "PARENT_MSTAR_MAX_OFFSET",
    "PARENT_MSTAR_MIN_OFFSET",
    "PARENT_ZD_MAX",
    "PARENT_ZD_MIN",
    "SIGMA_PROXY_FRACTIONAL_SCATTER",
    "SIZE_MU0_PHYSICAL",
    "SIZE_MU1_PHYSICAL",
    "SIZE_MU2_PHYSICAL",
    "SIZE_SCATTER",
    "TRUNCATION_MASS_POLYNOMIAL_COEFFICIENTS",
    "TRUNCATION_MASS_SCATTER",
    "active_size_relation_coefficients",
    "shift_physical_mass_location_to_hunits",
]
