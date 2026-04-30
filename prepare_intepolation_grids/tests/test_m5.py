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
from interpolation_grids.unit_conventions import (
    logMstar_h2_from_legacy,
    logRe_hinv_from_legacy,
    logSigmaStar_from_h_units,
    mR_hinv_from_fixed_kpc,
    Sunit_hinv_from_fixed_kpc,
)
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


def test_h_unit_conversion_helpers_match_documented_scalings() -> None:
    """The preparation layer should expose auditable h-unit migration helpers."""

    h_ref = 0.7
    gamma_grid = np.array([1.7, 2.0, 2.3], dtype=float)
    fixed_mass_grid = np.array([11.1, 11.2, 11.3], dtype=float)
    fixed_sunit_grid = np.array([0.4, 0.5, 0.6], dtype=float)

    log_mstar_h2 = logMstar_h2_from_legacy(11.4, h_ref=h_ref)
    log_re_hinv = logRe_hinv_from_legacy(0.8, h_ref=h_ref)

    np.testing.assert_allclose(log_mstar_h2, 11.4 + 2.0 * np.log10(h_ref))
    np.testing.assert_allclose(log_re_hinv, 0.8 + np.log10(h_ref))
    np.testing.assert_allclose(
        logSigmaStar_from_h_units(log_mstar_h2, log_re_hinv),
        11.4 - np.log10(2.0 * np.pi) - 2.0 * 0.8,
    )
    np.testing.assert_allclose(
        mR_hinv_from_fixed_kpc(fixed_mass_grid, gamma_grid, h_ref=h_ref),
        fixed_mass_grid - (2.0 - gamma_grid) * np.log10(h_ref),
    )
    np.testing.assert_allclose(
        Sunit_hinv_from_fixed_kpc(fixed_sunit_grid, gamma_grid, h_ref=h_ref),
        fixed_sunit_grid * np.power(h_ref, 2.0 - gamma_grid),
    )


def test_compute_mass_grid_can_build_h_unit_grid_directly() -> None:
    """The h-unit mass grid should match the analytic migration from fixed kpc."""

    sigma_crit = 2_223_801_018.8799353
    rein_kpc = 6.4
    h_ref = 0.7

    legacy_grid = compute_mass_grid(
        gamma_grid=GAMMA_GRID,
        sigma_crit=sigma_crit,
        rein_kpc=rein_kpc,
        mass_radius_kpc=5.0,
    )
    h_unit_grid = compute_mass_grid(
        gamma_grid=GAMMA_GRID,
        sigma_crit=sigma_crit,
        rein_kpc=rein_kpc,
        mass_radius_kpc=5.0,
        unit_convention="h_units_v1",
        h_ref=h_ref,
    )

    np.testing.assert_allclose(
        h_unit_grid,
        legacy_grid - (2.0 - GAMMA_GRID) * np.log10(h_ref),
        rtol=0.0,
        atol=1.0e-12,
    )
