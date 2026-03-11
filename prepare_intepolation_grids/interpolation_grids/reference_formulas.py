"""Reference implementations copied from the historical standalone scripts.

These helpers intentionally preserve the old numerical behavior. Production
code can call them directly or validate against them, depending on how much
refactoring we do later. Keeping them isolated also makes it obvious which
parts are "compatibility contracts" and which parts are project structure.
"""

from __future__ import annotations

import numpy as np

from spherical_jeans.mass_profiles import powerlaw


def legacy_m5_grid(
    gamma_grid: np.ndarray,
    sigma_crit: float,
    rein_kpc: float,
) -> np.ndarray:
    """Reproduce the `m5_grid` formula from `make_m5_grids.py`."""

    output = np.zeros_like(gamma_grid, dtype=float)
    for index, gamma in enumerate(gamma_grid):
        einstein_mass = np.pi * sigma_crit * rein_kpc**2
        powerlaw_normalization = einstein_mass / powerlaw.M2d(rein_kpc, gamma)
        output[index] = np.log10(powerlaw_normalization * powerlaw.M2d(5.0, gamma))
    return output


def legacy_dm5_dthetaein_grid(
    gamma_grid: np.ndarray,
    sigma_crit: float,
    theta_ein_arcsec: float,
    kpc_per_arcsec: float,
    theta_samples: int,
) -> np.ndarray:
    """Reproduce the derivative-grid recipe from `make_m5_grids.py`.

    The historical script uses dense finite differences in theta-space rather
    than an analytic derivative. We preserve that here because the current
    requirement is compatibility with that output, not algorithmic novelty.
    """

    theta_grid = np.linspace(theta_ein_arcsec * 0.1, theta_ein_arcsec * 2.0, theta_samples)
    rein_kpc_grid = theta_grid * kpc_per_arcsec
    derivative_grid = np.zeros_like(gamma_grid, dtype=float)

    for index, gamma in enumerate(gamma_grid):
        m5_grid = np.zeros_like(rein_kpc_grid, dtype=float)
        for sample_index, rein_kpc in enumerate(rein_kpc_grid):
            einstein_mass = np.pi * sigma_crit * rein_kpc**2
            powerlaw_normalization = einstein_mass / powerlaw.M2d(rein_kpc, gamma)
            m5_grid[sample_index] = np.log10(powerlaw_normalization * powerlaw.M2d(5.0, gamma))

        derivative_values = np.gradient(m5_grid, theta_grid)
        derivative_grid[index] = np.interp(theta_ein_arcsec, theta_grid, derivative_values)

    return derivative_grid

