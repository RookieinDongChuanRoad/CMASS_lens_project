"""Registry-driven JAX likelihood engine."""

from __future__ import annotations

import math
from functools import lru_cache
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from ..model_registry import get_model_definition
from ..types import CompiledModel, RuntimeConfig

JAX_DIAGNOSTIC_BLOB_DTYPE = np.dtype(
    [
        ("total_log_prob_seconds", np.float64),
        ("likelihood_seconds", np.float64),
        ("normalization_seconds", np.float64),
        ("fp_prior_seconds", np.float64),
        ("normalization_value", np.float64),
        ("fp_prior_log_term", np.float64),
        ("fpfit_mu", np.float64),
        ("fpfit_beta", np.float64),
        ("fpfit_xi", np.float64),
        ("fpfit_scatter", np.float64),
        ("backend", "S16"),
    ]
)


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
    """Return JAX posterior components for the active model."""

    model_definition = _model_definition_from_compiled(compiled_model)
    static_kwargs = dict(model_definition.static_jit_kwargs(compiled_model))
    compiled = _build_log_prob_components_jit(
        model_name=model_definition.name,
        static_items=tuple(sorted(static_kwargs.items())),
    )
    return compiled(
        jnp.asarray(theta, dtype=jnp.float64),
        model_definition.to_jax_context(compiled_model),
    )


def log_prob(theta: np.ndarray, compiled_model: CompiledModel) -> tuple[float, np.void]:
    """Evaluate the active JAX posterior and return a timing blob."""

    total_start = perf_counter()
    theta = np.asarray(theta, dtype=np.float64)
    parameter_schema = compiled_model.config.parameter_schema
    parameter_schema.validate_theta_shape(theta)

    for index, (_name, (lower, upper)) in enumerate(
        zip(
            parameter_schema.internal_parameter_names,
            parameter_schema.prior_bounds,
            strict=True,
        )
    ):
        value = float(theta[index])
        if value < lower or value > upper:
            total_seconds = perf_counter() - total_start
            return -np.inf, _build_timing_blob(
                total_log_prob_seconds=total_seconds,
                likelihood_seconds=0.0,
                normalization_seconds=0.0,
                fp_prior_seconds=0.0,
                normalization_value=0.0,
                fp_prior_log_term=0.0,
                fpfit_mu=math.nan,
                fpfit_beta=math.nan,
                fpfit_xi=math.nan,
                fpfit_scatter=math.nan,
            )

    component_start = perf_counter()
    (
        log_prob_total,
        _likelihood_value,
        normalization_value,
        fp_prior_log_term,
        fpfit_mu,
        fpfit_beta,
        fpfit_xi,
        fpfit_scatter,
    ) = log_prob_value(jnp.asarray(theta, dtype=jnp.float64), compiled_model)
    log_prob_total.block_until_ready()
    component_seconds = perf_counter() - component_start
    total_seconds = perf_counter() - total_start

    blob = _build_timing_blob(
        total_log_prob_seconds=total_seconds,
        likelihood_seconds=component_seconds,
        normalization_seconds=component_seconds,
        fp_prior_seconds=0.0,
        normalization_value=float(normalization_value),
        fp_prior_log_term=float(fp_prior_log_term),
        fpfit_mu=float(fpfit_mu),
        fpfit_beta=float(fpfit_beta),
        fpfit_xi=float(fpfit_xi),
        fpfit_scatter=float(fpfit_scatter),
    )
    return float(log_prob_total), blob


def _build_timing_blob(
    *,
    total_log_prob_seconds: float,
    likelihood_seconds: float,
    normalization_seconds: float,
    fp_prior_seconds: float,
    normalization_value: float,
    fp_prior_log_term: float,
    fpfit_mu: float,
    fpfit_beta: float,
    fpfit_xi: float,
    fpfit_scatter: float,
) -> np.void:
    """Build the structured diagnostic record returned with ``log_prob``."""

    return np.array(
        (
            float(total_log_prob_seconds),
            float(likelihood_seconds),
            float(normalization_seconds),
            float(fp_prior_seconds),
            float(normalization_value),
            float(fp_prior_log_term),
            float(fpfit_mu),
            float(fpfit_beta),
            float(fpfit_xi),
            float(fpfit_scatter),
            b"jax",
        ),
        dtype=JAX_DIAGNOSTIC_BLOB_DTYPE,
    )[()]


@lru_cache(maxsize=16)
def _build_log_prob_components_jit(
    *,
    model_name: str,
    static_items: tuple[tuple[str, int], ...],
):
    """
    Build a model-specialized JIT function.

    Concrete models provide only scientific hooks.  This backend owns the
    common execution shape: normalization vmap, likelihood reduction, extra
    prior composition, and posterior validity checks.
    """

    model_definition = get_model_definition(model_name)
    static = dict(static_items)

    @jax.jit
    def compiled(theta: jnp.ndarray, model_context) -> tuple[jnp.ndarray, ...]:
        theta_parts = model_definition.unpack_theta(theta)
        valid_theta = model_definition.validate_theta(theta, theta_parts, model_context, static)

        def one_normalization_sample(nrm: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            draw = model_definition.draw_population(theta_parts, nrm, model_context, static)
            weight = model_definition.selection_weight(theta_parts, draw, nrm, model_context, static)
            summary = model_definition.summary_row(theta_parts, draw, model_context, static)
            return weight, summary

        normalization_weights, summary_rows = jax.vmap(one_normalization_sample)(
            model_definition.normalization_samples(model_context)
        )
        z_norm = jnp.mean(normalization_weights)
        fp_summary = jnp.sum(summary_rows, axis=0)
        lens_integral_values = model_definition.lens_integrals(theta_parts, model_context, static)
        likelihood_valid = jnp.all(lens_integral_values > 0.0)
        likelihood_value = jnp.where(
            valid_theta & likelihood_valid,
            jnp.sum(jnp.log(lens_integral_values)),
            -jnp.inf,
        )
        log_extra_prior, fpfit_mu, fpfit_beta, fpfit_xi, fpfit_scatter = model_definition.extra_prior(
            fp_summary,
            model_context,
            static,
        )
        normalization_valid = (
            valid_theta
            & jnp.isfinite(z_norm)
            & (z_norm > model_definition.normalization_min_value(model_context))
        )
        prior_valid = jnp.isfinite(log_extra_prior)
        total = likelihood_value - lens_integral_values.shape[0] * jnp.log(z_norm) + log_extra_prior
        total = jnp.where(
            normalization_valid & prior_valid & jnp.isfinite(likelihood_value),
            total,
            -jnp.inf,
        )
        return total, likelihood_value, z_norm, log_extra_prior, fpfit_mu, fpfit_beta, fpfit_xi, fpfit_scatter

    return compiled


__all__ = ["JAX_DIAGNOSTIC_BLOB_DTYPE", "build_compiled_model", "log_prob", "log_prob_value"]
