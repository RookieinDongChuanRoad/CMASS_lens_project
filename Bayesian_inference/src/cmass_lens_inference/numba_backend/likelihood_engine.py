"""
Production Numba likelihood engine.

This module is the bridge between the framework-level sampler and model-owned
posterior adapters.  It owns box-prior rejection and common diagnostic
construction; concrete models own the posterior assembly callable registered in
`ModelDefinition`.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from ..model_registry import get_model_definition
from ..types import CompiledModel, RuntimeConfig
from .diagnostics import (
    NUMBA_DIAGNOSTIC_BLOB_DTYPE,
    build_reject_result,
)


def build_compiled_model(runtime_config: RuntimeConfig) -> CompiledModel:
    """
    Build the Numba compiled model for the configured registry model.

    Concrete runtime adapters still own canonical dataset loading and
    preprocessing.  This common function keeps production entrypoints from
    importing model-specific context builders directly.
    """

    model_definition = get_model_definition(runtime_config.model.name)
    return model_definition.build_compiled_model(runtime_config)


def _model_definition_from_compiled(compiled_model: CompiledModel):
    """Resolve the registry definition that owns a compiled model."""

    return get_model_definition(compiled_model.config.model.name)


def log_prob(theta: np.ndarray, compiled_model: CompiledModel) -> tuple[float, np.void]:
    """
    Evaluate one model posterior for emcee.

    The function first enforces the model's explicit box prior on the host.
    Only in-bounds proposals enter model-specific kernels, which keeps
    rejection behavior clear and avoids wasting kernel time on obviously
    invalid parameter vectors.
    """

    total_start = perf_counter()
    theta = np.asarray(theta, dtype=np.float64)
    parameter_schema = compiled_model.config.parameter_schema
    parameter_schema.validate_theta_shape(theta)
    model_definition = _model_definition_from_compiled(compiled_model)
    kernel = model_definition.backend_kernel

    for index, (_name, (lower, upper)) in enumerate(
        zip(
            parameter_schema.internal_parameter_names,
            parameter_schema.prior_bounds,
            strict=True,
        )
    ):
        value = float(theta[index])
        if value < lower or value > upper:
            return build_reject_result(total_start, compiled_model, kernel)

    return model_definition.evaluate_log_prob(theta, compiled_model, total_start)


__all__ = [
    "NUMBA_DIAGNOSTIC_BLOB_DTYPE",
    "build_compiled_model",
    "log_prob",
]
