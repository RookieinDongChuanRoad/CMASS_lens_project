"""Power-law mass-normalization grid calculations.

This module focuses on the `m5_grid` and `dm5_dthetaein_grid` datasets. The
first implementation intentionally mirrors the legacy script behavior because
the user explicitly asked for compatibility with historical outputs.
"""

from __future__ import annotations

import numpy as np

from interpolation_grids.reference_formulas import (
    legacy_dm5_dthetaein_grid,
    legacy_m5_grid,
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

    return legacy_m5_grid(
        gamma_grid=gamma_grid,
        sigma_crit=sigma_crit,
        rein_kpc=rein_kpc,
    )


def compute_dm5_dthetaein_grid(
    gamma_grid: np.ndarray,
    sigma_crit: float,
    theta_ein_arcsec: float,
    kpc_per_arcsec: float,
    theta_samples: int,
) -> np.ndarray:
    """Compute the derivative of `m5` with respect to Einstein radius.

    Why this uses the dense numerical recipe:
    - The current acceptance criterion is consistency with the historical
      script, not switching to a new derivative convention.
    - Keeping the exact numerical path makes the regression tests meaningful.
    """

    return legacy_dm5_dthetaein_grid(
        gamma_grid=gamma_grid,
        sigma_crit=sigma_crit,
        theta_ein_arcsec=theta_ein_arcsec,
        kpc_per_arcsec=kpc_per_arcsec,
        theta_samples=theta_samples,
    )

