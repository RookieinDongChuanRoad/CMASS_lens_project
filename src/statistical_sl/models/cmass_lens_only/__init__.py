"""CMASS lens-only model package."""

from .assembly import get_model_spec
from .runtime import get_runtime_adapter

__all__ = ["get_model_spec", "get_runtime_adapter"]
