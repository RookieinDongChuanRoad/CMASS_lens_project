"""Power-law mass-normalization grid calculations.

This module focuses on the `m5_grid` and `dm5_dthetaein_grid` datasets. The
first implementation intentionally mirrors the legacy script behavior because
the user explicitly asked for compatibility with historical outputs.
"""

from __future__ import annotations

import math

import numpy as np

from interpolation_grids.reference_formulas import (
    legacy_dm5_dthetaein_grid,
    legacy_m5_grid,
)
from interpolation_grids.unit_conventions import H_UNITS_V1, LEGACY_FIXED_KPC


def convert_log_enclosed_mass(
    log_mass: np.ndarray,
    gamma_grid: np.ndarray,
    from_radius_kpc: float,
    to_radius_kpc: float,
) -> np.ndarray:
    """Convert one logarithmic enclosed-mass grid between two radii."""

    if float(from_radius_kpc) == float(to_radius_kpc):
        return np.asarray(log_mass, dtype=float).copy()
    radius_ratio = float(to_radius_kpc) / float(from_radius_kpc)
    return np.asarray(log_mass, dtype=float) + (3.0 - np.asarray(gamma_grid, dtype=float)) * math.log10(radius_ratio)


def compute_mass_grid(
    gamma_grid: np.ndarray,
    sigma_crit: float,
    rein_kpc: float,
    mass_radius_kpc: float,
    *,
    unit_convention: str = LEGACY_FIXED_KPC,
    h_ref: float = 0.7,
) -> np.ndarray:
    """
    Compute the enclosed-mass grid for an arbitrary supported radius.

    The legacy numerical contract is defined at 5 kpc. Other supported radii
    are derived from that exact reference grid using the analytic power-law
    conversion so we preserve historical behavior while avoiding duplicate
    numerical implementations.

    Under `h_units_v1`, `mass_radius_kpc` is the coefficient in
    `R h^-1 kpc`, not a physical kpc aperture. The direct formula uses
    `Sigma_c/h_ref` and `r_ein*h_ref` so the stored mass grid is natively
    `log10[M(<R h^-1 kpc)/(h^-1 Msun)]`.
    """

    normalized_convention = str(unit_convention).strip()
    if normalized_convention == H_UNITS_V1:
        h_value = float(h_ref)
        if not math.isfinite(h_value) or h_value <= 0.0:
            raise ValueError(f"h_ref must be a positive finite value, got {h_ref!r}.")
        gamma_array = np.asarray(gamma_grid, dtype=float)
        return np.asarray(
            np.log10(
                math.pi
                * (float(sigma_crit) / h_value)
                * np.power(float(rein_kpc) * h_value, gamma_array - 1.0)
                * np.power(float(mass_radius_kpc), 3.0 - gamma_array)
            ),
            dtype=float,
        )
    if normalized_convention != LEGACY_FIXED_KPC:
        raise ValueError(
            "Unsupported unit_convention for mass-grid generation: "
            f"{unit_convention!r}. Expected '{LEGACY_FIXED_KPC}' or '{H_UNITS_V1}'."
        )

    m5_grid = legacy_m5_grid(
        gamma_grid=gamma_grid,
        sigma_crit=sigma_crit,
        rein_kpc=rein_kpc,
    )
    return convert_log_enclosed_mass(
        log_mass=m5_grid,
        gamma_grid=gamma_grid,
        from_radius_kpc=5.0,
        to_radius_kpc=mass_radius_kpc,
    )


def compute_m5_grid(
    gamma_grid: np.ndarray,
    sigma_crit: float,
    rein_kpc: float,
) -> np.ndarray:
    """Compute the `m5_grid` used by downstream interpolation.

    Parameters
    ----------
    gamma_grid:
        Sampled 3D density slopes for the power-law profile.
    sigma_crit:
        Critical surface density for the lens-source geometry.
    rein_kpc:
        Einstein radius expressed in physical kpc.

    Returns
    -------
    np.ndarray
        One `m5` value per gamma sample. The output is dimensionless because it
        is stored as a base-10 logarithm.
    """

    return compute_mass_grid(
        gamma_grid=gamma_grid,
        sigma_crit=sigma_crit,
        rein_kpc=rein_kpc,
        mass_radius_kpc=5.0,
    )


def compute_dmass_dthetaein_grid(
    gamma_grid: np.ndarray,
    sigma_crit: float,
    theta_ein_arcsec: float,
    kpc_per_arcsec: float,
    theta_samples: int,
    mass_radius_kpc: float,
) -> np.ndarray:
    """Compute the derivative of `m_R` with respect to Einstein radius.

    Why this uses the dense numerical recipe:
    - The current acceptance criterion is consistency with the historical
      script, not switching to a new derivative convention.
    - Keeping the exact numerical path makes the regression tests meaningful.

    The derivative is invariant under changing the enclosed-mass definition
    radius because the radius-dependent term in `m_R` is additive and does not
    depend on Einstein radius. The `mass_radius_kpc` argument is kept explicit
    so callers do not have to rely on that fact implicitly.
    """

    _ = float(mass_radius_kpc)
    return legacy_dm5_dthetaein_grid(
        gamma_grid=gamma_grid,
        sigma_crit=sigma_crit,
        theta_ein_arcsec=theta_ein_arcsec,
        kpc_per_arcsec=kpc_per_arcsec,
        theta_samples=theta_samples,
    )


def compute_dm5_dthetaein_grid(
    gamma_grid: np.ndarray,
    sigma_crit: float,
    theta_ein_arcsec: float,
    kpc_per_arcsec: float,
    theta_samples: int,
) -> np.ndarray:
    """Backward-compatible wrapper for the historical `dm5` derivative grid."""

    return compute_dmass_dthetaein_grid(
        gamma_grid=gamma_grid,
        sigma_crit=sigma_crit,
        theta_ein_arcsec=theta_ein_arcsec,
        kpc_per_arcsec=kpc_per_arcsec,
        theta_samples=theta_samples,
        mass_radius_kpc=5.0,
    )
