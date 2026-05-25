"""Production Numba/emcee inference backend.

This namespace owns inference-specific backend glue for the current production
sampler.  Generic numerical primitives live under ``statistical_sl.numerics``;
model-specific posterior hooks live under ``statistical_sl.models``.
"""

from __future__ import annotations


__all__: list[str] = []
