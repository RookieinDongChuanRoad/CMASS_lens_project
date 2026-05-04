"""
Sonnenfeld 2024 SLACS debiased model boundary.

This file intentionally defines the module boundary before enabling the model.
The requested refactor is about separating reusable backend code from concrete
scientific models; the Sonnenfeld numerical model still needs its own parent
population, likelihood, and Monte Carlo selection-normalization implementation
before it can be sampled safely.
"""

from __future__ import annotations

def get_model_definition():
    """
    Fail explicitly until the Sonnenfeld likelihood is implemented.

    Keeping this as a real module instead of a missing import gives users and
    future developers a precise extension point while preventing accidental
    execution of CMASS equations under a Sonnenfeld config label.
    """

    raise NotImplementedError(
        "sonnenfeld2024_slacs is registered as a model module boundary, but "
        "its parent-population likelihood and Monte Carlo selection "
        "normalization are not implemented yet."
    )


__all__ = ["get_model_definition"]
