"""
Top-level package for the CMASS lens Bayesian inference framework.

The public surface is intentionally small:
- configuration loading
- single-run and resume entrypoints
- typed result objects that callers and tests can inspect
"""

import jax

# Scientific inference uses JAX/NumPyro as the production backend.  Enabling
# x64 at package import keeps likelihood evaluation, posterior sampling, and
# tests on the same double-precision contract no matter which public entrypoint
# imports the package first.
jax.config.update("jax_enable_x64", True)

from .runner import resume_inference, run_inference
from .types import RunResult

__all__ = [
    "RunResult",
    "resume_inference",
    "run_inference",
]
