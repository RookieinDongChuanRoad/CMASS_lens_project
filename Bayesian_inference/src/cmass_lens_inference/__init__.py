"""
Top-level package for the CMASS lens Bayesian inference framework.

The public surface is intentionally small:
- configuration loading
- single-run and resume entrypoints
- typed result objects that callers and tests can inspect
"""

from .runner import resume_inference, run_inference
from .types import RunResult

__all__ = [
    "RunResult",
    "resume_inference",
    "run_inference",
]
