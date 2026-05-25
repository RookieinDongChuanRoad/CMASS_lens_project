"""Public package facade for the statistical strong-lensing codebase.

This namespace is the reusable package surface for the restructured repository.
Workflow code lives under ``data_preparation``, ``inference``, and
``posterior_predictive``; local configs, data, and outputs live under the
repository-level ``workspace`` tree.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
