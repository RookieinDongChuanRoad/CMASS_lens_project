"""
Common compiled-context dispatch helpers.

The hunit-aware data loading and array preparation still live in
``compiled_context.py`` because they are shared by the retained legacy oracle
and the CMASS JAX model.  This module is the backend-facing wrapper that keeps
new code from importing the legacy file directly.
"""

from __future__ import annotations

from ..compiled_context import build_compiled_context

__all__ = ["build_compiled_context"]
