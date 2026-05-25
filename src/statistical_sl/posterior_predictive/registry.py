"""Predictive diagnostics registry for the standalone PPT package.

The inference package already resolves concrete scientific models through
``model.name``.  This registry mirrors that boundary for posterior-predictive
workflows: generic PPT code asks for the active model's predictive definition,
then either receives a model-owned hook or fails before any CMASS-specific
fallback can run.
"""

from __future__ import annotations

from statistical_sl.posterior_predictive.adapters import cmass, sonnenfeld
from statistical_sl.posterior_predictive.interfaces import PredictiveDefinition


class UnsupportedPredictiveModelError(ValueError):
    """Raised when a model has no posterior-predictive diagnostics contract."""


def get_predictive_definition(model_name: str) -> PredictiveDefinition:
    """
    Return the predictive definition registered for ``model_name``.

    The explicit error is part of the contract.  It prevents unsupported models
    from accidentally flowing through CMASS-only diagnostics while their own
    predictive semantics are still being designed.
    """

    if model_name == "cmass":
        return cmass.get_predictive_definition()
    if model_name in sonnenfeld.MODEL_NAMES:
        return sonnenfeld.get_predictive_definition(model_name)
    raise UnsupportedPredictiveModelError(
        "Model "
        f"'{model_name}' does not yet expose posterior predictive diagnostics. "
        "Add a model-specific predictive hook before running PPT for this model."
    )


__all__ = [
    "UnsupportedPredictiveModelError",
    "get_predictive_definition",
]
