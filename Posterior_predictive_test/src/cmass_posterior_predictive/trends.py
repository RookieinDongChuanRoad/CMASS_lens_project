"""
Trend-figure entrypoints for the standalone posterior-predictive package.

The heavy lifting still lives in `predictive.py` because the historical code
was authored as a single PPT-family module. This wrapper isolates the public
trend API so callers do not need to know that internal detail.
"""

from __future__ import annotations

from .predictive import run_posterior_trends

__all__ = ["run_posterior_trends"]
