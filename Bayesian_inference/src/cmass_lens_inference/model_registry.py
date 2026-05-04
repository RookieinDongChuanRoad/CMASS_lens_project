"""
Scientific-model registry.

All production entrypoints resolve models through this module.  That keeps
configuration parsing, NumPyro sampling, benchmarks, and tests from importing
CMASS implementation details directly.  Adding a new model should mean adding
a new module under ``models/`` and registering it here.
"""

from __future__ import annotations

from .model_interfaces import ModelDefinition
from .models import cmass_current, sonnenfeld2024_slacs


def get_model_definition(model_name: str) -> ModelDefinition:
    """
    Return the model definition registered under ``model_name``.

    Sonnenfeld 2024 is intentionally present but disabled: the module boundary
    exists, while the scientific likelihood and selection normalization are
    still future work.  Raising ``NotImplementedError`` here prevents a config
    typo from silently running the CMASS equations under a Sonnenfeld label.
    """

    if model_name == "cmass_current":
        return cmass_current.get_model_definition()
    if model_name == "sonnenfeld2024_slacs":
        return sonnenfeld2024_slacs.get_model_definition()
    raise ValueError(
        "Unsupported model preset "
        f"'{model_name}'. Expected one of: cmass_current, sonnenfeld2024_slacs."
    )


__all__ = ["get_model_definition"]
