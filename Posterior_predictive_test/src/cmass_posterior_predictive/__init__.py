"""
Standalone posterior-predictive package for CMASS lens analyses.

This package owns all posterior-predictive, trend, monitor, and notebook
comparison workflows. It deliberately reuses the inference engine in
`cmass_lens_inference` as a dependency, but it does not extend that package's
public surface with PPT-specific APIs.
"""

from .notebook_comparison import run_notebook_pipeline_comparison
from .predictive import run_posterior_predictive, wait_for_external_sigma_tables_and_run
from .trends import run_posterior_trends
from .types import (
    NotebookComparisonResult,
    PosteriorPredictiveMonitorResult,
    PosteriorPredictiveResult,
    PosteriorTrendResult,
)

__all__ = [
    "NotebookComparisonResult",
    "PosteriorPredictiveMonitorResult",
    "PosteriorPredictiveResult",
    "PosteriorTrendResult",
    "run_notebook_pipeline_comparison",
    "run_posterior_predictive",
    "wait_for_external_sigma_tables_and_run",
    "run_posterior_trends",
]
