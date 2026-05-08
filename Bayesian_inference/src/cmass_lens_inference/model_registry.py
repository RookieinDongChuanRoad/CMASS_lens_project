"""
Scientific-model registry.

All production entrypoints resolve models through this module.  That keeps
configuration parsing, emcee sampling, benchmarks, and tests from importing
CMASS implementation details directly.  Adding a new model should mean adding
a new module under ``models/`` and registering it here.
"""

from __future__ import annotations

from .numba_backend.compiled_model_factory import build_model_definition
from .model_interfaces import ModelDefinition
from .models import (
    cmass,
    cmass_runtime,
    sonnenfeld2024_slacs,
    sonnenfeld2024_slacs_runtime,
    toy_hierarchical,
    toy_hierarchical_runtime,
)
from .models.cmass import posterior as cmass_posterior
from .models.sonnenfeld2024_slacs import posterior as sonnenfeld2024_slacs_posterior
from .models.toy_hierarchical import posterior as toy_hierarchical_posterior


def get_model_definition(model_name: str) -> ModelDefinition:
    """
    Return the model definition registered under ``model_name``.

    Every branch returns a complete ``ModelDefinition`` built from a
    human-authored ``ModelSpec`` and a runtime adapter.  The registry is the
    only production dispatch point, so config parsing, numba kernels, emcee,
    and benchmarks all see the same model boundary.
    """

    if model_name == "cmass":
        return build_model_definition(
            cmass.get_model_spec(),
            cmass_runtime.get_runtime_adapter(),
            cmass_posterior.log_prob,
        )
    if model_name == "sonnenfeld2024_slacs":
        return build_model_definition(
            sonnenfeld2024_slacs.get_model_spec(),
            sonnenfeld2024_slacs_runtime.get_runtime_adapter(),
            sonnenfeld2024_slacs_posterior.log_prob,
        )
    if model_name == "sonnenfeld2024_slacs_hunit":
        return build_model_definition(
            sonnenfeld2024_slacs.get_hunit_model_spec(),
            sonnenfeld2024_slacs_runtime.get_runtime_adapter(),
            sonnenfeld2024_slacs_posterior.log_prob,
        )
    if model_name == "toy_hierarchical":
        return build_model_definition(
            toy_hierarchical.get_model_spec(),
            toy_hierarchical_runtime.get_runtime_adapter(),
            toy_hierarchical_posterior.log_prob,
        )
    raise ValueError(
        "Unsupported model preset "
        f"'{model_name}'. Expected one of: cmass, sonnenfeld2024_slacs, "
        "sonnenfeld2024_slacs_hunit, toy_hierarchical."
    )


__all__ = ["get_model_definition"]
