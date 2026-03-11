"""
Tests for the second-stage performance refactor.

These tests deliberately lock the architectural shape before implementation:
- public numerical primitives must live in `kernels/primitives.py`
- the production log-probability path must build a compiled array context
- likelihood and normalization must execute through monolithic numba kernels

The scientific model is still exercised on tiny synthetic data. That keeps the
tests fast while proving that the new performance-oriented structure is really
in use.
"""

from __future__ import annotations

from numba.core.registry import CPUDispatcher
import numpy as np

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.model import build_compiled_model, log_prob
from cmass_lens_inference.kernels.likelihood import log_likelihood_lenses_numba
from cmass_lens_inference.kernels.normalization import normalization_mc_numba
from cmass_lens_inference.kernels.primitives import (
    interp1d_clip,
    normal_ppf,
    skewnorm_sample,
    theta_ein_arcsec,
    truncnorm_sample,
)


def test_compiled_model_builds_contiguous_array_context(synthetic_config_path) -> None:
    """
    The new compiled model builder must convert the observation list into
    contiguous arrays so that numba kernels can consume the full dataset in one
    pass without Python object dispatch.
    """

    runtime_config = load_runtime_config(synthetic_config_path)
    compiled_model = build_compiled_model(runtime_config)

    assert compiled_model.context.zd.ndim == 1
    assert compiled_model.context.m5_grid_int.ndim == 2
    assert compiled_model.context.mstar_base.ndim == 2
    assert compiled_model.context.base_normals.ndim == 2
    assert compiled_model.context.zd.flags.c_contiguous
    assert compiled_model.context.m5_grid_int.flags.c_contiguous
    assert compiled_model.context.mstar_base.flags.c_contiguous
    assert compiled_model.context.base_normals.flags.c_contiguous


def test_model_log_prob_runs_through_monolithic_numba_kernels(synthetic_config_path) -> None:
    """
    The production `log_prob` entrypoint must compile and execute both the
    all-lens likelihood kernel and the normalization kernel.
    """

    runtime_config = load_runtime_config(synthetic_config_path)
    compiled_model = build_compiled_model(runtime_config)

    log_prob_value, blob = log_prob(
        runtime_config.sampling.initial_center.to_array(),
        compiled_model,
    )

    assert isinstance(log_likelihood_lenses_numba, CPUDispatcher)
    assert isinstance(normalization_mc_numba, CPUDispatcher)
    assert log_likelihood_lenses_numba.signatures
    assert normalization_mc_numba.signatures
    assert np.isfinite(log_prob_value)
    assert blob.dtype.names is not None
    assert set(blob.dtype.names) >= {
        "total_log_prob_seconds",
        "likelihood_seconds",
        "normalization_seconds",
        "normalization_value",
        "parallel_strategy",
    }
    assert blob["parallel_strategy"].decode("utf-8").rstrip("\x00") == compiled_model.parallelism.strategy


def test_kernel_primitives_live_in_shared_module_and_compile() -> None:
    """
    Shared numerical primitives should compile from one place so likelihood and
    normalization reuse identical approximations and sampling transforms.
    """

    interp_value = interp1d_clip(0.5, np.array([0.0, 1.0]), np.array([10.0, 20.0]))
    ppf_value = normal_ppf(0.84)
    skew_value = skewnorm_sample(11.0, 0.2, 1.5, 0.1, -0.2)
    trunc_value = truncnorm_sample(2.0, 0.2, 1.2, 2.8, 0.3)
    theta_value = theta_ein_arcsec(
        0.55,
        1.8,
        11.2,
        2.0,
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.0, 1000.0, 1800.0, 2400.0]),
    )

    assert isinstance(interp1d_clip, CPUDispatcher)
    assert isinstance(normal_ppf, CPUDispatcher)
    assert isinstance(skewnorm_sample, CPUDispatcher)
    assert isinstance(truncnorm_sample, CPUDispatcher)
    assert isinstance(theta_ein_arcsec, CPUDispatcher)
    assert interp1d_clip.signatures
    assert normal_ppf.signatures
    assert skewnorm_sample.signatures
    assert truncnorm_sample.signatures
    assert theta_ein_arcsec.signatures
    assert interp_value == 15.0
    assert np.isfinite(ppf_value)
    assert np.isfinite(skew_value)
    assert np.isfinite(trunc_value)
    assert theta_value >= 0.0
