"""
Scientific-model registry.

All production entrypoints resolve models through this module.  That keeps
configuration parsing, emcee sampling, benchmarks, and tests from importing
CMASS implementation details directly.  Adding a new model should mean adding
a new module under ``models/`` and registering it here.
"""

from __future__ import annotations

from .numba_backend.model_adapter import build_model_definition
from .model_interfaces import ModelDefinition
from .models import (
    cmass,
    cmass_runtime,
    sonnenfeld2024_slacs,
    sonnenfeld2024_slacs_runtime,
)


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
        )
    if model_name == "sonnenfeld2024_slacs":
        return build_model_definition(
            sonnenfeld2024_slacs.get_model_spec(),
            sonnenfeld2024_slacs_runtime.get_runtime_adapter(),
        )
    if model_name == "sonnenfeld2024_slacs_hunit":
        return build_model_definition(
            sonnenfeld2024_slacs.get_hunit_model_spec(),
            sonnenfeld2024_slacs_runtime.get_runtime_adapter(),
        )
    raise ValueError(
        "Unsupported model preset "
        f"'{model_name}'. Expected one of: cmass, sonnenfeld2024_slacs, "
        "sonnenfeld2024_slacs_hunit."
    )


__all__ = ["get_model_definition"]
