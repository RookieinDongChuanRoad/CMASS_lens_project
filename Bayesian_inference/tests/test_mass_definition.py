"""
Tests for the shared mass-definition abstraction.

These tests lock the new contract that lets the scientific pipeline switch
between `m5` and `m10` without duplicating the full implementation stack.
The goal is to keep one internal representation while preserving definition-
specific public labels and exact physical conversions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cmass_lens_inference.mass_definition import (
    H_UNITS_V1,
    LEGACY_FIXED_KPC,
    convert_log_enclosed_mass,
    convert_log_mass_fixed_kpc_to_hinv,
    convert_sigma_unit_fixed_kpc_to_hinv,
    get_mass_definition,
    sigma_bundle_filename,
)


def test_get_mass_definition_exposes_public_labels_and_units() -> None:
    """The definition helper should expose the public naming contract."""

    m5 = get_mass_definition(5)
    m10 = get_mass_definition(10)

    assert m5.label == "m5"
    assert m5.radius_kpc == 5.0
    assert m5.public_parameter_names == ("mu5_0", "beta5", "xi5", "sigma5")
    assert m5.sigma_unit_units == "km2 s-2 per 10**m5"

    assert m10.label == "m10"
    assert m10.radius_kpc == 10.0
    assert m10.public_parameter_names == ("mu10_0", "beta10", "xi10", "sigma10")
    assert m10.sigma_unit_units == "km2 s-2 per 10**m10"


def test_get_h_unit_mass_definition_exposes_h_dependent_contract() -> None:
    """The h-units mode must use distinct labels, parameter names, and metadata."""

    m5h = get_mass_definition(5, unit_convention=H_UNITS_V1)
    m10h = get_mass_definition(10, unit_convention=H_UNITS_V1)

    assert m5h.unit_convention == H_UNITS_V1
    assert m5h.label == "m5_hinvkpc"
    assert m5h.subgroup_name == "m5_hinvkpc"
    assert m5h.radius_kpc == 5.0
    assert m5h.aperture_h_power == -1
    assert m5h.mass_h_power == -1
    assert m5h.public_parameter_names == ("mu5h_0", "beta5h", "xi5h", "sigma5h")
    assert m5h.sigma_unit_units == "km2 s-2 per 10**m5_hinvkpc"
    assert m5h.physical_radius_kpc(h_ref=0.7) == pytest.approx(5.0 / 0.7)
    assert m5h.log_mass_physical_offset(h_ref=0.7) == pytest.approx(math.log10(0.7 ** -1))

    assert m10h.label == "m10_hinvkpc"
    assert m10h.public_parameter_names == ("mu10h_0", "beta10h", "xi10h", "sigma10h")


def test_legacy_mass_definition_keeps_existing_contract() -> None:
    """The explicit legacy mode should preserve the fixed-kpc public API."""

    m5 = get_mass_definition(5, unit_convention=LEGACY_FIXED_KPC)

    assert m5.unit_convention == LEGACY_FIXED_KPC
    assert m5.label == "m5"
    assert m5.aperture_h_power == 0
    assert m5.mass_h_power == 0
    assert m5.physical_radius_kpc(h_ref=0.7) == pytest.approx(5.0)
    assert m5.log_mass_physical_offset(h_ref=0.7) == pytest.approx(0.0)


def test_sigma_bundle_filename_exposes_canonical_per_profile_names() -> None:
    """The bundle helper should keep PPT and asset-build naming in lockstep."""

    assert sigma_bundle_filename("devauc") == "jeans_deV_sigma_bundle.h5"
    assert sigma_bundle_filename("sersic") == "jeans_sers_sigma_bundle.h5"


def test_convert_log_enclosed_mass_matches_exact_power_law_relation() -> None:
    """`m5` and `m10` conversion should follow the exact analytic relation."""

    gamma = np.array([1.2, 2.0, 2.8], dtype=float)
    m5_values = np.array([11.3, 11.1, 10.9], dtype=float)

    converted = convert_log_enclosed_mass(
        log_mass=m5_values,
        gamma=gamma,
        from_radius_kpc=5.0,
        to_radius_kpc=10.0,
    )

    expected = m5_values + (3.0 - gamma) * math.log10(2.0)
    np.testing.assert_allclose(converted, expected, rtol=0.0, atol=1.0e-12)


def test_convert_log_enclosed_mass_round_trips_between_m5_and_m10() -> None:
    """Converting to `m10` and back to `m5` should be lossless numerically."""

    gamma = np.linspace(1.2, 2.8, 17)
    m5_values = np.linspace(11.6, 10.8, 17)

    m10_values = convert_log_enclosed_mass(
        log_mass=m5_values,
        gamma=gamma,
        from_radius_kpc=5.0,
        to_radius_kpc=10.0,
    )
    restored = convert_log_enclosed_mass(
        log_mass=m10_values,
        gamma=gamma,
        from_radius_kpc=10.0,
        to_radius_kpc=5.0,
    )

    np.testing.assert_allclose(restored, m5_values, rtol=0.0, atol=1.0e-12)


def test_h_unit_analytic_migration_helpers_preserve_observable_scaling() -> None:
    """The h-units migration helpers should implement the documented formulae."""

    gamma = np.array([1.7, 2.0, 2.3], dtype=float)
    fixed_mass = np.array([11.1, 11.2, 11.3], dtype=float)
    fixed_sigma_unit = np.array([0.4, 0.5, 0.6], dtype=float)
    h_ref = 0.7

    migrated_mass = convert_log_mass_fixed_kpc_to_hinv(
        log_mass_fixed_kpc=fixed_mass,
        gamma=gamma,
        h_ref=h_ref,
    )
    migrated_sigma_unit = convert_sigma_unit_fixed_kpc_to_hinv(
        sigma_unit_fixed_kpc=fixed_sigma_unit,
        gamma=gamma,
        h_ref=h_ref,
    )

    np.testing.assert_allclose(
        migrated_mass,
        fixed_mass - (2.0 - gamma) * math.log10(h_ref),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        migrated_sigma_unit,
        fixed_sigma_unit * np.power(h_ref, 2.0 - gamma),
        rtol=0.0,
        atol=1.0e-12,
    )
