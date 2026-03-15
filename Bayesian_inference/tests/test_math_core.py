"""
Tests for numerical helper behavior that is explicit in the requirements.

These tests intentionally target small deterministic helpers so the first
implementation cycle can establish correct behavior without having to run the
entire sampler.
"""

from __future__ import annotations

import numpy as np
from astropy.cosmology import FlatLambdaCDM as AstropyFlatLambdaCDM

import pytest

from cmass_lens_inference.cosmology import FlatLambdaCDM
from cmass_lens_inference.interpolation import clipped_linear_interp
from cmass_lens_inference.mass_definition import convert_log_enclosed_mass
from cmass_lens_inference.normalization import apply_normalization_guard


def test_clipped_linear_interp_clips_to_boundary_values() -> None:
    """
    The interpolation helper must clip to the edge values outside the input
    range because the requirements explicitly forbid extrapolation.
    """

    x = np.array([0.0, 1.0, 2.0])
    y = np.array([10.0, 20.0, 30.0])
    x_new = np.array([-1.0, 0.5, 5.0])

    interpolated = clipped_linear_interp(x, y, x_new)

    assert np.allclose(interpolated, np.array([10.0, 15.0, 30.0]))


def test_theta_ein_returns_zero_when_deflector_is_not_in_front_of_source() -> None:
    """
    Einstein radius must collapse to zero when `z_d >= z_s`.
    """

    cosmology = FlatLambdaCDM()
    theta_ein = cosmology.theta_ein_from_mass_gamma(
        z_d=1.0,
        z_s=0.9,
        mass_value=11.0,
        gamma=2.0,
        mass_radius_kpc=5.0,
    )

    assert theta_ein == 0.0


def test_theta_ein_is_invariant_under_exact_m5_to_m10_conversion() -> None:
    """The physical Einstein radius should not depend on the chosen mass label."""

    cosmology = FlatLambdaCDM()
    gamma = 2.15
    m5 = 11.24
    m10 = float(
        convert_log_enclosed_mass(
            log_mass=np.array([m5], dtype=float),
            gamma=np.array([gamma], dtype=float),
            from_radius_kpc=5.0,
            to_radius_kpc=10.0,
        )[0]
    )

    theta_from_m5 = cosmology.theta_ein_from_mass_gamma(
        z_d=0.55,
        z_s=1.8,
        mass_value=m5,
        gamma=gamma,
        mass_radius_kpc=5.0,
    )
    theta_from_m10 = cosmology.theta_ein_from_mass_gamma(
        z_d=0.55,
        z_s=1.8,
        mass_value=m10,
        gamma=gamma,
        mass_radius_kpc=10.0,
    )

    assert theta_from_m5 == pytest.approx(theta_from_m10, rel=0.0, abs=1.0e-12)


def test_flat_lcdm_builds_its_distance_table_from_astropy() -> None:
    """
    The wrapper should delegate the base comoving-distance grid to astropy.

    That guarantees the Python helpers and the `numba` kernels both consume the
    same cosmology source of truth instead of a hand-maintained approximation.
    """

    cosmology = FlatLambdaCDM(h0=70.0, omega_m=0.3)
    astropy_cosmology = AstropyFlatLambdaCDM(H0=70.0, Om0=0.3)
    expected_comoving_table = astropy_cosmology.comoving_distance(cosmology.z_table).value

    assert cosmology.z_table.dtype == np.float64
    assert cosmology.comoving_distance_table_mpc.dtype == np.float64
    assert cosmology.z_table.flags.c_contiguous
    assert cosmology.comoving_distance_table_mpc.flags.c_contiguous
    np.testing.assert_array_equal(cosmology.comoving_distance_table_mpc, expected_comoving_table)


def test_angular_diameter_distance_between_clips_non_physical_ordering_to_zero() -> None:
    """
    The project contract must keep front/back ordering guards explicit.

    Astropy returns a signed `D_ls` for `z_s <= z_d`, but the inference code
    expects this invalid geometry to collapse to zero.
    """

    cosmology = FlatLambdaCDM(h0=70.0, omega_m=0.3)

    assert cosmology.angular_diameter_distance_between_mpc(1.0, 0.9) == 0.0


def test_apply_normalization_guard_rejects_tiny_normalization_values() -> None:
    """
    The normalization guard enforces the hard rejection threshold from the
    requirements document.
    """

    assert apply_normalization_guard(1.0e-12) == -np.inf
    assert apply_normalization_guard(1.0e-3) == 0.0
