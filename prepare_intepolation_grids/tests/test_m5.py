"""Regression tests for the power-law mass normalization grids.

These tests intentionally compare the new local implementation against the
behavior encoded in the legacy reference script
`/Users/liurongfu/Desktop/Spectrum_reduction/make_m5_grids.py`.
The goal is not to invent a new numerical convention; the goal is to preserve
the historical output while moving the code into a maintainable structure.
"""

from __future__ import annotations

import numpy as np

from interpolation_grids.config import DEFAULT_DERIVATIVE_THETA_SAMPLES, GAMMA_GRID
from interpolation_grids.physics.m5 import (
    compute_dm5_dthetaein_grid,
    compute_m5_grid,
)
from interpolation_grids.reference_formulas import (
    legacy_dm5_dthetaein_grid,
    legacy_m5_grid,
)


def test_m5_grid_matches_legacy_reference_script_values() -> None:
    """The local `m5_grid` implementation must match legacy script output."""

    sigma_crit = 2_223_801_018.8799353
    rein_kpc = 6.4

    expected = legacy_m5_grid(
        gamma_grid=GAMMA_GRID,
        sigma_crit=sigma_crit,
        rein_kpc=rein_kpc,
    )
    actual = compute_m5_grid(
        gamma_grid=GAMMA_GRID,
        sigma_crit=sigma_crit,
        rein_kpc=rein_kpc,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_dm5_grid_matches_legacy_reference_script_values() -> None:
    """The local derivative grid must follow the legacy numerical recipe."""

    sigma_crit = 2_223_801_018.8799353
    theta_ein_arcsec = 0.929
    kpc_per_arcsec = 6.4 / theta_ein_arcsec

    expected = legacy_dm5_dthetaein_grid(
        gamma_grid=GAMMA_GRID,
        sigma_crit=sigma_crit,
        theta_ein_arcsec=theta_ein_arcsec,
        kpc_per_arcsec=kpc_per_arcsec,
        theta_samples=DEFAULT_DERIVATIVE_THETA_SAMPLES,
    )
    actual = compute_dm5_dthetaein_grid(
        gamma_grid=GAMMA_GRID,
        sigma_crit=sigma_crit,
        theta_ein_arcsec=theta_ein_arcsec,
        kpc_per_arcsec=kpc_per_arcsec,
        theta_samples=DEFAULT_DERIVATIVE_THETA_SAMPLES,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)
