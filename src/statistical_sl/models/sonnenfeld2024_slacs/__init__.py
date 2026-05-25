"""Sonnenfeld 2024 SLACS production model package."""

from __future__ import annotations

from .assembly import (
    COMPONENTS,
    HUNIT_MODEL_NAME,
    INTERNAL_PARAMETER_NAMES,
    MODEL_NAME,
    PARAMETER_SPECS,
    PARAMETERS,
    PUBLIC_PARAMETER_NAMES,
    REQUIRED_CAPABILITIES,
    get_hunit_model_spec,
    get_model_spec,
)

__all__ = [
    "COMPONENTS",
    "HUNIT_MODEL_NAME",
    "INTERNAL_PARAMETER_NAMES",
    "MODEL_NAME",
    "PARAMETER_SPECS",
    "PARAMETERS",
    "PUBLIC_PARAMETER_NAMES",
    "REQUIRED_CAPABILITIES",
    "get_hunit_model_spec",
    "get_model_spec",
]
