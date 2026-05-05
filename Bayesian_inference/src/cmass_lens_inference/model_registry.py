"""
Scientific-model registry.

All production entrypoints resolve models through this module.  That keeps
configuration parsing, NumPyro sampling, benchmarks, and tests from importing
CMASS implementation details directly.  Adding a new model should mean adding
a new module under ``models/`` and registering it here.
"""

from __future__ import annotations

from .jax_backend.model_adapter import build_model_definition
from .model_interfaces import ModelDefinition
from .models import cmass, cmass_runtime, sonnenfeld2024_slacs


def get_model_definition(model_name: str) -> ModelDefinition:
    """
    Return the model definition registered under ``model_name``.

    Sonnenfeld 2024 is intentionally present but disabled.  Raising
    ``NotImplementedError`` here prevents a config typo from silently running
    CMASS equations under a Sonnenfeld label.
    """

    if model_name == "cmass":
        return build_model_definition(
            cmass.get_model_spec(),
            cmass_runtime.get_runtime_adapter(),
        )
    if model_name == "sonnenfeld2024_slacs":
        return sonnenfeld2024_slacs.get_model_definition()
    raise ValueError(
        "Unsupported model preset "
        f"'{model_name}'. Expected one of: cmass, sonnenfeld2024_slacs."
    )


__all__ = ["get_model_definition"]
