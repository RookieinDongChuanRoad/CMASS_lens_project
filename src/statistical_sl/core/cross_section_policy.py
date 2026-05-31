"""Canonical cross-section source and boundary-policy contracts.

The canonical HDF5 block stores a numerical ``theta_E x gamma`` grid, but that
grid alone does not fully define how runtime code should evaluate values outside
the tabulated domain.  This module keeps the two required pieces explicit:

* ``source`` says what physical or preparation product produced the grid.
* ``boundary_policy`` says how consumers may evaluate outside the finite table.

Keeping those fields separate is intentional.  A real finite-fibre two
dimensional table and a CMASS separable table can both be represented as dense
arrays, but only the separable table has a valid analytic ``theta_E^2``
extension.  Runtime code therefore resolves the pair into a compact integer
mode before entering Numba kernels.
"""

from __future__ import annotations


SOURCE_SEPARABLE_CS_OVER_THETA_EIN = "separable_cs_over_theta_ein"
SOURCE_MUFIBRE3_CS_GRID = "mufibre3_cs_grid"

BOUNDARY_ZERO_OUTSIDE_THETA_CLIP_GAMMA = "zero_outside_theta_clip_gamma"
BOUNDARY_THETA_SQUARED_EXTRAPOLATE_CLIP_GAMMA = "theta_squared_extrapolate_clip_gamma"

CROSS_SECTION_MODE_GRID_ZERO_OUTSIDE = 0
CROSS_SECTION_MODE_SEPARABLE_THETA_SQUARED = 1


class CrossSectionPolicyError(ValueError):
    """Raised when canonical cross-section metadata describes an invalid mode."""


def resolve_cross_section_mode(source: str, boundary_policy: str) -> int:
    """
    Convert canonical cross-section metadata into a Numba-friendly mode code.

    ``source`` and ``boundary_policy`` are deliberately not collapsed into one
    string because they answer different questions.  ``source`` identifies the
    scientific data product; ``boundary_policy`` identifies the allowed
    numerical behavior outside its finite axes.  The pair is what lets CMASS
    separable grids extrapolate analytically without granting the same behavior
    to Sonnenfeld finite-fibre tables.
    """

    normalized_source = str(source).strip()
    normalized_policy = str(boundary_policy).strip()

    if normalized_policy == BOUNDARY_ZERO_OUTSIDE_THETA_CLIP_GAMMA:
        if normalized_source in {SOURCE_SEPARABLE_CS_OVER_THETA_EIN, SOURCE_MUFIBRE3_CS_GRID}:
            return CROSS_SECTION_MODE_GRID_ZERO_OUTSIDE

    if normalized_policy == BOUNDARY_THETA_SQUARED_EXTRAPOLATE_CLIP_GAMMA:
        if normalized_source == SOURCE_SEPARABLE_CS_OVER_THETA_EIN:
            return CROSS_SECTION_MODE_SEPARABLE_THETA_SQUARED
        raise CrossSectionPolicyError(
            "Cross-section source/boundary_policy combination is invalid: "
            f"source={normalized_source!r}, boundary_policy={normalized_policy!r}. "
            "Only separable cs_over_theta_ein grids may use theta-squared extrapolation."
        )

    raise CrossSectionPolicyError(
        "Unsupported cross-section source/boundary_policy combination: "
        f"source={normalized_source!r}, boundary_policy={normalized_policy!r}."
    )


def is_separable_theta_squared_mode(mode_code: int) -> bool:
    """Return whether a resolved mode uses CMASS separable theta-squared scaling."""

    return int(mode_code) == CROSS_SECTION_MODE_SEPARABLE_THETA_SQUARED


__all__ = [
    "BOUNDARY_THETA_SQUARED_EXTRAPOLATE_CLIP_GAMMA",
    "BOUNDARY_ZERO_OUTSIDE_THETA_CLIP_GAMMA",
    "CROSS_SECTION_MODE_GRID_ZERO_OUTSIDE",
    "CROSS_SECTION_MODE_SEPARABLE_THETA_SQUARED",
    "CrossSectionPolicyError",
    "SOURCE_MUFIBRE3_CS_GRID",
    "SOURCE_SEPARABLE_CS_OVER_THETA_EIN",
    "is_separable_theta_squared_mode",
    "resolve_cross_section_mode",
]
