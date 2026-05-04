"""
Registry-driven JAX likelihood engine.

This module is the production backend entrypoint.  It deliberately contains no
CMASS equations: the active model is resolved from ``RuntimeConfig.model`` and
the model definition supplies the compiled-context builder and log-probability
functions.  The small wrappers here give NumPyro, runner, benchmarks, and tests
one stable backend API.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..model_registry import get_model_definition
from ..types import CompiledModel, RuntimeConfig


def _model_definition_from_compiled(compiled_model: CompiledModel):
    """Resolve the model that created a compiled model container."""

    return get_model_definition(compiled_model.config.model.name)


def build_compiled_model(runtime_config: RuntimeConfig) -> CompiledModel:
    """
    Build the JAX compiled model for the configured scientific model.

    The concrete model remains responsible for converting observations and
    configuration into its own shape-stable context.  This common function only
    handles registry dispatch.
    """

    model_definition = get_model_definition(runtime_config.model.name)
    return model_definition.build_compiled_model(runtime_config)


def log_prob_value(theta: jnp.ndarray, compiled_model: CompiledModel) -> tuple[jnp.ndarray, ...]:
    """Return JAX posterior components for the compiled model's definition."""

    return _model_definition_from_compiled(compiled_model).log_prob_value(theta, compiled_model)


def log_prob(theta: np.ndarray, compiled_model: CompiledModel) -> tuple[float, np.void]:
    """Evaluate the active model's host-facing log-probability wrapper."""

    return _model_definition_from_compiled(compiled_model).log_prob(theta, compiled_model)


__all__ = ["build_compiled_model", "log_prob", "log_prob_value"]
