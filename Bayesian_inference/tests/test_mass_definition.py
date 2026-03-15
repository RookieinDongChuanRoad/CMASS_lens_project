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

from cmass_lens_inference.mass_definition import (
    convert_log_enclosed_mass,
    get_mass_definition,
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
