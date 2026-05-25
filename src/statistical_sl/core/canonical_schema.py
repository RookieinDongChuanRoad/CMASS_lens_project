"""Canonical inference dataset schema constants.

This module intentionally contains names and capability labels only.  It is not
an inference validator.  Keeping these strings in ``statistical_sl.core`` lets
data preparation, inference, and posterior predictive code agree on one stable
HDF5 contract without importing one another's workflow internals.
"""

from __future__ import annotations

CANONICAL_SCHEMA_VERSION = "canonical_inference_dataset_v1"

BLOCK_METADATA = "metadata"
BLOCK_LENSES = "lenses"
BLOCK_LENSING_MASS_GRIDS = "lensing_mass_grids"
BLOCK_LENSING_CROSS_SECTION = "lensing_cross_section"
BLOCK_VELOCITY_DISPERSION_GRIDS = "velocity_dispersion_grids"

TOP_LEVEL_BLOCKS = (
    BLOCK_METADATA,
    BLOCK_LENSES,
    BLOCK_LENSING_MASS_GRIDS,
    BLOCK_LENSING_CROSS_SECTION,
    BLOCK_VELOCITY_DISPERSION_GRIDS,
)

CAPABILITY_LENS_OBSERVATIONS_V1 = "lens_observations.v1"
CAPABILITY_LENSING_MASS_GRIDS_V1 = "lensing_mass_grids.v1"
CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1 = "lensing_cross_section.theta_gamma_grid.v1"
CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1 = "velocity_dispersion.per_lens_s2.v1"
CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1 = "velocity_dispersion.fp_within_re.v1"
CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1 = "velocity_dispersion.population_sigma_unit.v1"

DEFAULT_BOUNDARY_POLICY = "zero_outside_theta_clip_gamma"

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
