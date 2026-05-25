"""
Standalone posterior-predictive package for CMASS lens analyses.

This package owns posterior-predictive, trend, monitor, and diagnostics
workflows. It consumes saved inference artifacts and package-owned inference
contracts without extending the inference public surface with diagnostics-only
APIs.
"""

from .predictive import run_posterior_diagnostics, run_posterior_predictive, wait_for_external_sigma_tables_and_run
from .predictive import annotate_existing_fig8_like_figures_with_observations
from .trends import run_posterior_trends
from .types import (
    Fig8ObservationAnnotationResult,
    PosteriorPredictiveMonitorResult,
    PosteriorPredictiveResult,
    PosteriorDiagnosticsResult,
    PosteriorTrendResult,
)

__all__ = [
    "Fig8ObservationAnnotationResult",
    "PosteriorPredictiveMonitorResult",
    "PosteriorPredictiveResult",
    "PosteriorDiagnosticsResult",
    "PosteriorTrendResult",
    "annotate_existing_fig8_like_figures_with_observations",
    "run_posterior_diagnostics",
    "run_posterior_predictive",
    "wait_for_external_sigma_tables_and_run",
    "run_posterior_trends",
]
