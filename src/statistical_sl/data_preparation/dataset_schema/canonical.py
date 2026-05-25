"""Data-preparation access to canonical inference dataset schema constants.

The canonical schema contract now lives in :mod:`statistical_sl.core` so data
preparation, inference, and posterior predictive workflows share one neutral
definition.  This module re-exports the shared constants for data-preparation
code that naturally groups schema writing helpers under this package.
"""

from __future__ import annotations

from statistical_sl.core.canonical_schema import (
    BLOCK_LENSES,
    BLOCK_LENSING_CROSS_SECTION,
    BLOCK_LENSING_MASS_GRIDS,
    BLOCK_METADATA,
    BLOCK_VELOCITY_DISPERSION_GRIDS,
    CANONICAL_SCHEMA_VERSION,
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1,
    DEFAULT_BOUNDARY_POLICY,
    TOP_LEVEL_BLOCKS,
)

__all__ = [
    "BLOCK_LENSES",
    "BLOCK_LENSING_CROSS_SECTION",
    "BLOCK_LENSING_MASS_GRIDS",
    "BLOCK_METADATA",
    "BLOCK_VELOCITY_DISPERSION_GRIDS",
    "CANONICAL_SCHEMA_VERSION",
    "CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1",
    "CAPABILITY_LENSING_MASS_GRIDS_V1",
    "CAPABILITY_LENS_OBSERVATIONS_V1",
    "CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1",
    "CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1",
    "CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1",
    "DEFAULT_BOUNDARY_POLICY",
    "TOP_LEVEL_BLOCKS",
]
