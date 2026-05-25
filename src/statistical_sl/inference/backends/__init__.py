"""Inference backend namespaces owned by the public ``statistical_sl`` package.

Backends in this package are framework glue: they connect model declarations,
runtime contexts, samplers, and diagnostic artifact contracts.  Reusable
numerical kernels live in ``statistical_sl.numerics`` instead of here, so
posterior predictive workflows can share low-level math without depending on
inference-private sampler machinery.
"""

from __future__ import annotations

__all__: list[str] = []
