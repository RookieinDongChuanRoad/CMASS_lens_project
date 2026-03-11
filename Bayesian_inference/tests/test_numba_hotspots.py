"""
Tests that lock in real numba usage for the production numerical hotspots.

The refactor moved the hot path away from per-lens wrappers and into a single
compiled model entrypoint. These tests verify that the public path still
reaches `numba` kernels rather than silently regressing to Python loops.
"""

from __future__ import annotations

from pathlib import Path

from numba.core.registry import CPUDispatcher
import numpy as np

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.kernels.likelihood import log_likelihood_lenses_numba
from cmass_lens_inference.kernels.normalization import normalization_mc_numba
from cmass_lens_inference.model import build_compiled_model, log_prob
from cmass_lens_inference.normalization import build_random_basis, estimate_normalization


def test_log_prob_uses_monolithic_likelihood_kernel(synthetic_config_path: Path) -> None:
    """
    The public `log_prob` path should compile and execute the new all-lens
    kernel instead of dispatching one lens at a time from Python.
    """

    runtime_config = load_runtime_config(synthetic_config_path)
    compiled_model = build_compiled_model(runtime_config)

    likelihood_value, _ = log_prob(
        runtime_config.sampling.initial_center.to_array(),
        compiled_model,
    )

    assert isinstance(log_likelihood_lenses_numba, CPUDispatcher)
    assert log_likelihood_lenses_numba.signatures
    assert np.isfinite(likelihood_value)


def test_estimate_normalization_uses_monolithic_numba_kernel(synthetic_config_path: Path) -> None:
    """
    The standalone normalization wrapper must still execute the production
    `numba` kernel so helper paths and tests share the same implementation.
    """

    runtime_config = load_runtime_config(synthetic_config_path)
    compiled_model = build_compiled_model(runtime_config)
    random_basis = build_random_basis(
        runtime_config.integration.normalization_samples,
        runtime_config.sampling.random_seed,
    )

    normalization_value = estimate_normalization(
        runtime_config.sampling.initial_center,
        compiled_model.profile,
        random_basis,
        compiled_model.cosmology,
        compiled_model.cross_section_grid,
    )

    assert isinstance(normalization_mc_numba, CPUDispatcher)
    assert normalization_mc_numba.signatures
    assert normalization_value >= 0.0
