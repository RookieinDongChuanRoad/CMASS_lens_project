"""
Top-level package for the CMASS lens Bayesian inference framework.

The public surface is intentionally small:
- configuration loading
- single-run and resume entrypoints
- typed result objects that callers and tests can inspect
"""

from .posterior_predictive import (
    run_posterior_predictive,
    run_posterior_trends,
    wait_for_external_sigma_tables_and_run,
)
from .runner import resume_inference, run_inference
from .sampler import build_log_prob_fn
from .types import PosteriorPredictiveMonitorResult, PosteriorPredictiveResult, PosteriorTrendResult, RunResult

__all__ = [
    "RunResult",
    "PosteriorPredictiveResult",
    "PosteriorPredictiveMonitorResult",
    "PosteriorTrendResult",
    "build_log_prob_fn",
    "run_posterior_predictive",
    "run_posterior_trends",
    "wait_for_external_sigma_tables_and_run",
    "resume_inference",
    "run_inference",
]
