"""
Tests for the CMASS deterministic preprocessing boundary.

`models.components.cmass.preprocessing` is allowed to know CMASS formulas and context
fields.  It should be callable directly from a canonical dataset so
`cmass_runtime.py` can remain a small glue module.
"""

from __future__ import annotations

import numpy as np

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.models.components.cmass.context import CMASSModelContext
from cmass_lens_inference.models.components.cmass.preprocessing import (
    build_cmass_context_from_canonical_dataset,
)


def test_cmass_preprocessing_builds_model_context_from_canonical_dataset(
    synthetic_config_path,
) -> None:
    """The CMASS preprocessor should own deterministic context construction."""

    runtime_config = load_runtime_config(synthetic_config_path)
    context_result = build_cmass_context_from_canonical_dataset(runtime_config)

    assert isinstance(context_result.context, CMASSModelContext)
    assert context_result.context.gamma_grid_int.shape == (runtime_config.integration.gamma_points,)
    assert context_result.context.mstar_grid.shape == (
        1,
        runtime_config.integration.mstar_points,
    )
    assert context_result.context.mass_radius_kpc == np.float64(5.0 / 0.7)
    assert context_result.metadata["canonical_schema_version"] == "canonical_inference_dataset_v1"
    assert "lens_observations.v1" in context_result.metadata["canonical_capabilities"]
