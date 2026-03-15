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
    compute_dmass_dthetaein_grid,
    compute_mass_grid,
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


def test_compute_mass_grid_supports_exact_m10_conversion() -> None:
    """The generic mass-grid helper should reproduce the exact `m10` relation."""

    sigma_crit = 2_223_801_018.8799353
    rein_kpc = 6.4

    m5_grid = compute_m5_grid(
        gamma_grid=GAMMA_GRID,
        sigma_crit=sigma_crit,
        rein_kpc=rein_kpc,
    )
    m10_grid = compute_mass_grid(
        gamma_grid=GAMMA_GRID,
        sigma_crit=sigma_crit,
        rein_kpc=rein_kpc,
        mass_radius_kpc=10.0,
    )

    expected = m5_grid + (3.0 - GAMMA_GRID) * np.log10(2.0)
    np.testing.assert_allclose(m10_grid, expected, rtol=0.0, atol=1e-12)


def test_compute_dmass_grid_is_invariant_between_m5_and_m10() -> None:
    """Changing the enclosed-mass radius should not change the theta derivative."""

    sigma_crit = 2_223_801_018.8799353
    theta_ein_arcsec = 0.929
    kpc_per_arcsec = 6.4 / theta_ein_arcsec

    dm5_grid = compute_dm5_dthetaein_grid(
        gamma_grid=GAMMA_GRID,
        sigma_crit=sigma_crit,
        theta_ein_arcsec=theta_ein_arcsec,
        kpc_per_arcsec=kpc_per_arcsec,
        theta_samples=DEFAULT_DERIVATIVE_THETA_SAMPLES,
    )
    dm10_grid = compute_dmass_dthetaein_grid(
        gamma_grid=GAMMA_GRID,
        sigma_crit=sigma_crit,
        theta_ein_arcsec=theta_ein_arcsec,
        kpc_per_arcsec=kpc_per_arcsec,
        theta_samples=DEFAULT_DERIVATIVE_THETA_SAMPLES,
        mass_radius_kpc=10.0,
    )

    np.testing.assert_allclose(dm10_grid, dm5_grid, rtol=0.0, atol=1e-12)
