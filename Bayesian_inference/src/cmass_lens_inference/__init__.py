"""
Top-level package for the CMASS lens Bayesian inference framework.

The public surface is intentionally small:
- configuration loading
- single-run and resume entrypoints
- typed result objects that callers and tests can inspect
"""

import jax

# Scientific inference in this package compares JAX results against legacy
# NumPy/numba double-precision kernels and uses small likelihood tolerances.
# Enabling x64 at package import makes that precision contract independent of
# whether callers enter through `runner`, `numpyro_sampler`, or `jax_model`.
jax.config.update("jax_enable_x64", True)

from .runner import resume_inference, run_inference
from .types import RunResult

__all__ = [
    "RunResult",
    "resume_inference",
    "run_inference",
]
