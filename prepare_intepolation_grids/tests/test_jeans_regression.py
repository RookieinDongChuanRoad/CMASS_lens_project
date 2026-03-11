"""Regression and policy tests for the velocity-dispersion grid calculation.

Two distinct behaviors matter now:

1. We must be able to reproduce the historical `make_jeans_grid.py` behavior
   for the free-Sersic branch. That verifies the new code still understands the
   old reference implementation correctly.
2. The production code has intentionally diverged from that historical script
   by using a fixed 1.6 arcsec aperture width while keeping seeing fixed at
   0.9 arcsec. That policy change must also be tested explicitly.
"""

from __future__ import annotations

import h5py
import numpy as np
from astropy.constants import G, M_sun, kpc
from astropy.cosmology import FlatLambdaCDM
from spherical_jeans import sigma_model, tracer_profiles
from spherical_jeans.mass_profiles import powerlaw

from interpolation_grids.config import (
    APERTURE_HEIGHT_ARCSEC,
    DEFAULT_APERTURE_WIDTH_ARCSEC,
    DEFAULT_RADIAL_GRID_SIZE,
    GAMMA_GRID,
    SEEING_FWHM_ARCSEC,
)
from interpolation_grids.io.hdf5 import build_galaxy_inputs
from interpolation_grids.physics.jeans import compute_s2_grid, kpc_per_arcsec, uses_devaucouleurs_branch


DATA_ROOT = "/Users/liurongfu/Work/CMASS_lens_project/data/raw"
COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3)
SIGMA2_TO_KM2_PER_S2 = (G * M_sun / kpc).to("km2 / s2").value


def _legacy_reference_s2_grid(galaxy, gamma_grid: np.ndarray) -> np.ndarray:
    """Reproduce the historical `make_jeans_grid.py` calculation path.

    This helper intentionally uses the aperture width stored in the input file,
    because the legacy script did the same. It is a test-only compatibility
    oracle, not the current production policy.
    """

    physical_kpc_per_arcsec = COSMOLOGY.kpc_proper_per_arcmin(galaxy.zd).value / 60.0
    aperture_kpc = [
        APERTURE_HEIGHT_ARCSEC * physical_kpc_per_arcsec,
        galaxy.aperture_width_arcsec * physical_kpc_per_arcsec,
    ]
    seeing_kpc = SEEING_FWHM_ARCSEC * physical_kpc_per_arcsec
    radial_anchor_kpc = galaxy.re_arcsec * physical_kpc_per_arcsec
    radial_grid = np.logspace(
        np.log10(radial_anchor_kpc) - 3.0,
        np.log10(radial_anchor_kpc) + 3.0,
        DEFAULT_RADIAL_GRID_SIZE,
    )

    output = np.zeros_like(gamma_grid, dtype=float)
    for index, gamma in enumerate(gamma_grid):
        normalization = 1.0 / powerlaw.M2d(5.0, gamma)
        enclosed_mass_grid = normalization * powerlaw.M3d(radial_grid, gamma)
        output[index] = sigma_model.sigma2(
            (radial_grid, enclosed_mass_grid),
            aperture_kpc,
            [radial_anchor_kpc, galaxy.nser],
            tracer_profiles.sersic,
            seeing=seeing_kpc,
        ) * SIGMA2_TO_KM2_PER_S2
    return output


def _policy_s2_grid(galaxy, gamma_grid: np.ndarray, aperture_width_arcsec: float) -> np.ndarray:
    """Compute `s2_grid` for an explicit aperture policy.

    The helper mirrors the production implementation but keeps the aperture
    width overridable so tests can compare the old 0.8 arcsec behavior against
    the new 1.6 arcsec business rule while leaving seeing fixed.
    """

    physical_kpc_per_arcsec = kpc_per_arcsec(galaxy.zd)
    aperture_kpc = [
        APERTURE_HEIGHT_ARCSEC * physical_kpc_per_arcsec,
        aperture_width_arcsec * physical_kpc_per_arcsec,
    ]
    seeing_kpc = SEEING_FWHM_ARCSEC * physical_kpc_per_arcsec

    if uses_devaucouleurs_branch(galaxy.source_filename):
        radial_anchor_kpc = galaxy.reff_dev_arcsec * physical_kpc_per_arcsec
        tracer_parameters = radial_anchor_kpc
        tracer_profile = tracer_profiles.deVaucouleurs
    else:
        radial_anchor_kpc = galaxy.re_arcsec * physical_kpc_per_arcsec
        tracer_parameters = (radial_anchor_kpc, galaxy.nser)
        tracer_profile = tracer_profiles.sersic

    radial_grid = np.logspace(
        np.log10(radial_anchor_kpc) - 3.0,
        np.log10(radial_anchor_kpc) + 3.0,
        DEFAULT_RADIAL_GRID_SIZE,
    )

    output = np.zeros_like(gamma_grid, dtype=float)
    for index, gamma in enumerate(gamma_grid):
        normalization = 1.0 / powerlaw.M2d(5.0, gamma)
        enclosed_mass_grid = normalization * powerlaw.M3d(radial_grid, gamma)
        output[index] = sigma_model.sigma2(
            (radial_grid, enclosed_mass_grid),
            aperture_kpc,
            tracer_parameters,
            tracer_profile,
            seeing=seeing_kpc,
        ) * SIGMA2_TO_KM2_PER_S2
    return output


def test_sersic_s2_grid_matches_legacy_reference_script_path() -> None:
    """The legacy helper should still reproduce the old 0.8 arcsec recipe."""

    input_path = f"{DATA_ROOT}/observations_with_m5_grids_all.hdf5"
    group_name = "023817-054555"

    with h5py.File(input_path, "r") as handle:
        galaxy = build_galaxy_inputs(
            group_name=group_name,
            group_handle=handle[group_name],
            source_filename=input_path,
        )
    actual = _legacy_reference_s2_grid(galaxy=galaxy, gamma_grid=GAMMA_GRID)
    expected = _policy_s2_grid(
        galaxy=galaxy,
        gamma_grid=GAMMA_GRID,
        aperture_width_arcsec=galaxy.aperture_width_arcsec,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)


def test_production_sersic_s2_grid_uses_fixed_1p6_arcsec_aperture() -> None:
    """Production Sersic output should follow the new 1.6 arcsec policy."""

    input_path = f"{DATA_ROOT}/observations_with_m5_grids_all.hdf5"
    group_name = "023817-054555"

    with h5py.File(input_path, "r") as handle:
        galaxy = build_galaxy_inputs(
            group_name=group_name,
            group_handle=handle[group_name],
            source_filename=input_path,
        )
        stored_production = handle[group_name]["s2_grid"][:]

    actual = compute_s2_grid(galaxy=galaxy, gamma_grid=GAMMA_GRID)
    expected_policy = _policy_s2_grid(
        galaxy=galaxy,
        gamma_grid=GAMMA_GRID,
        aperture_width_arcsec=DEFAULT_APERTURE_WIDTH_ARCSEC,
    )
    old_policy = _policy_s2_grid(
        galaxy=galaxy,
        gamma_grid=GAMMA_GRID,
        aperture_width_arcsec=galaxy.aperture_width_arcsec,
    )

    np.testing.assert_allclose(actual, expected_policy, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(actual, stored_production, rtol=1e-8, atol=1e-12)
    assert np.max(np.abs(actual - old_policy) / old_policy) > 0.05


def test_production_devaucouleurs_s2_grid_uses_fixed_1p6_arcsec_aperture() -> None:
    """The deV branch should also follow the new fixed-aperture policy."""

    input_path = f"{DATA_ROOT}/observations_deV_with_m5_grids.hdf5"
    group_name = "023817-054555"

    with h5py.File(input_path, "r") as handle:
        galaxy = build_galaxy_inputs(
            group_name=group_name,
            group_handle=handle[group_name],
            source_filename=input_path,
        )
        stored_production = handle[group_name]["s2_grid"][:]

    actual = compute_s2_grid(galaxy=galaxy, gamma_grid=GAMMA_GRID)
    expected_policy = _policy_s2_grid(
        galaxy=galaxy,
        gamma_grid=GAMMA_GRID,
        aperture_width_arcsec=DEFAULT_APERTURE_WIDTH_ARCSEC,
    )
    old_policy = _policy_s2_grid(
        galaxy=galaxy,
        gamma_grid=GAMMA_GRID,
        aperture_width_arcsec=galaxy.aperture_width_arcsec,
    )

    np.testing.assert_allclose(actual, expected_policy, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(actual, stored_production, rtol=1e-8, atol=1e-12)
    assert np.max(np.abs(actual - old_policy) / old_policy) > 0.05
