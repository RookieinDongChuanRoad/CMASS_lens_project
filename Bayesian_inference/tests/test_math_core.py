"""
Tests for numerical helper behavior that is explicit in the requirements.

These tests intentionally target small deterministic helpers so the first
implementation cycle can establish correct behavior without having to run the
entire sampler.
"""

from __future__ import annotations

import numpy as np

from cmass_lens_inference.cosmology import FlatLambdaCDM
from cmass_lens_inference.interpolation import clipped_linear_interp
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
    theta_ein = cosmology.theta_ein_from_m5_gamma(
        z_d=1.0,
        z_s=0.9,
        m5=11.0,
        gamma=2.0,
    )

    assert theta_ein == 0.0


def test_apply_normalization_guard_rejects_tiny_normalization_values() -> None:
    """
    The normalization guard enforces the hard rejection threshold from the
    requirements document.
    """

    assert apply_normalization_guard(1.0e-12) == -np.inf
    assert apply_normalization_guard(1.0e-3) == 0.0

