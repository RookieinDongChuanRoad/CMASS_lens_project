"""
Top-level package for the CMASS lens Bayesian inference framework.

The public surface is intentionally small:
- configuration loading
- single-run and resume entrypoints
- typed result objects that callers and tests can inspect
"""

from .runner import resume_inference, run_inference
from .sampler import build_log_prob_fn
from .types import RunResult

__all__ = [
    "RunResult",
    "build_log_prob_fn",
    "resume_inference",
    "run_inference",
]
