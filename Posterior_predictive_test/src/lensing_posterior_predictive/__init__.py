"""
Standalone posterior-predictive package for CMASS lens analyses.

This package owns posterior-predictive, trend, monitor, and diagnostics
workflows. It deliberately reuses the inference engine in
`cmass_lens_inference` as a dependency, but it does not extend that package's
public surface with PPT-specific APIs.
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
