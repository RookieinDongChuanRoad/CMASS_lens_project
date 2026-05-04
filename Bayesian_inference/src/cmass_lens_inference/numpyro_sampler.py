"""
NumPyro sampler orchestration for the JAX inference backend.

The old sampler module is intentionally left as a legacy test oracle.  New
production inference uses this module: NumPyro owns parameter sampling, while
the registry-driven JAX backend contributes the configured model likelihood
through a `numpyro.factor`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_uniform

from .jax_backend.likelihood_engine import log_prob_value
from .outputs import append_run_log
from .types import RuntimeContext


@dataclass(frozen=True)
class NumPyroSamplerResult:
    """
    In-memory result returned after one NumPyro run.

    `samples_by_chain` keeps the natural NumPyro layout
    `(chain, draw, parameter)`.  `flat_samples` is the same data flattened for
    corner plotting and simple downstream consumers.
    """

    samples_by_chain: np.ndarray
    flat_samples: np.ndarray
    log_prob_by_chain: np.ndarray
    acceptance_fraction_mean: float
    diagnostics: dict[str, Any]
    last_state: Any


def numpyro_model(runtime_context: RuntimeContext) -> None:
    """
    NumPyro model for the configured CMASS component assembly.

    Each hyper-parameter is sampled under its configured box prior.  The
    scientific likelihood is not decomposed into observed NumPyro sites because
    it already exists as a vectorized JAX expression with explicit selection
    normalization; `numpyro.factor` is therefore the correct way to add that
    custom log-density term to the posterior.
    """

    if runtime_context.compiled_model is None:
        raise RuntimeError("RuntimeContext is missing the JAX compiled model required by NumPyro.")

    parameter_schema = runtime_context.config.parameter_schema
    theta_values = []
    for parameter_name, (lower, upper) in zip(
        parameter_schema.internal_parameter_names,
        parameter_schema.prior_bounds,
        strict=True,
    ):
        theta_values.append(
            numpyro.sample(
                parameter_name,
                dist.Uniform(
                    low=jnp.asarray(lower, dtype=jnp.float64),
                    high=jnp.asarray(upper, dtype=jnp.float64),
                ),
            )
        )

    theta = jnp.stack(theta_values)
    (
        total_log_prob,
        likelihood_value,
        normalization_value,
        fp_prior_log_term,
        fpfit_mu,
        fpfit_beta,
        fpfit_xi,
        fpfit_scatter,
    ) = log_prob_value(theta, runtime_context.compiled_model)

    numpyro.factor("cmass_log_likelihood", total_log_prob)
    numpyro.deterministic("likelihood_value", likelihood_value)
    numpyro.deterministic("normalization_value", normalization_value)
    numpyro.deterministic("fp_prior_log_term", fp_prior_log_term)
    numpyro.deterministic("fpfit_mu", fpfit_mu)
    numpyro.deterministic("fpfit_beta", fpfit_beta)
    numpyro.deterministic("fpfit_xi", fpfit_xi)
    numpyro.deterministic("fpfit_scatter", fpfit_scatter)


def _initial_values(runtime_context: RuntimeContext) -> dict[str, float]:
    """
    Return the configured initial center under internal parameter names.

    Using explicit initial values is important for this project because the
    posterior contains hard physical rejection surfaces.  Starting from the
    validated center avoids wasting NumPyro warmup on invalid regions.
    """

    return runtime_context.config.sampling.initial_center.to_dict()


def _bounded_jittered_init(site=None, *, initial_values: dict[str, float], prior_bounds: dict[str, tuple[float, float]], jitter_scale: float):
    """
    NumPyro init strategy that jitters each chain inside the constrained box.

    NumPyro calls an init strategy once per chain when `chain_method` is
    sequential. Using the site's RNG key here gives every chain an independent
    starting point without changing the model's scalar parameter sites. The
    clipping margin keeps Uniform-prior initial values away from exact support
    boundaries, where unconstrained transforms can become numerically awkward.
    """

    if site is None:
        return partial(
            _bounded_jittered_init,
            initial_values=initial_values,
            prior_bounds=prior_bounds,
            jitter_scale=float(jitter_scale),
        )

    if site["type"] == "sample" and not site["is_observed"] and site["name"] in initial_values:
        parameter_name = site["name"]
        center = jnp.asarray(initial_values[parameter_name], dtype=jnp.float64)
        lower, upper = prior_bounds[parameter_name]
        lower_value = jnp.asarray(lower, dtype=jnp.float64)
        upper_value = jnp.asarray(upper, dtype=jnp.float64)
        width = upper_value - lower_value
        margin = jnp.maximum(jnp.asarray(1.0e-12, dtype=jnp.float64), jnp.asarray(1.0e-9, dtype=jnp.float64) * width)
        rng_key = site["kwargs"].get("rng_key")
        if jitter_scale > 0.0 and rng_key is not None:
            proposed = center + jnp.asarray(jitter_scale, dtype=jnp.float64) * jax.random.normal(rng_key, dtype=jnp.float64)
        else:
            proposed = center
        return jnp.clip(proposed, lower_value + margin, upper_value - margin)

    return init_to_uniform(site)


def _build_jittered_initial_strategy(runtime_context: RuntimeContext):
    """
    Build the NUTS initialization strategy for this run.

    NUTS does not use an ensemble geometry, but multi-chain runs still benefit
    from independent, valid initial points for convergence diagnostics.
    """

    runtime_config = getattr(runtime_context, "config", runtime_context)
    parameter_schema = runtime_config.parameter_schema
    return _bounded_jittered_init(
        initial_values=runtime_config.sampling.initial_center.to_dict(),
        prior_bounds={
            name: tuple(bounds)
            for name, bounds in zip(
                parameter_schema.internal_parameter_names,
                parameter_schema.prior_bounds,
                strict=True,
            )
        },
        jitter_scale=runtime_config.sampling.initial_jitter_scale,
    )


def _samples_to_ordered_array(samples: dict[str, np.ndarray], parameter_names: tuple[str, ...]) -> np.ndarray:
    """Convert NumPyro's sample dictionary into the canonical parameter matrix."""

    return np.stack([np.asarray(samples[name], dtype=float) for name in parameter_names], axis=-1)


def _extract_log_prob_by_chain(extra_fields: dict[str, np.ndarray], samples_by_chain: np.ndarray) -> np.ndarray:
    """
    Convert NumPyro extra fields into a log-probability matrix when available.

    NumPyro stores the Hamiltonian potential energy, which is negative log
    posterior up to constants introduced by constrained transforms.  It is
    still the best compact scalar diagnostic for checkpoint compatibility and
    run summaries.
    """

    potential_energy = extra_fields.get("potential_energy")
    if potential_energy is None:
        return np.full(samples_by_chain.shape[:2], np.nan, dtype=float)
    return -np.asarray(potential_energy, dtype=float)


def _acceptance_fraction(extra_fields: dict[str, np.ndarray]) -> float:
    """Return the mean NumPyro acceptance probability when present."""

    accept_prob = extra_fields.get("accept_prob")
    if accept_prob is None:
        return float("nan")
    return float(np.nanmean(np.asarray(accept_prob, dtype=float)))


def run_numpyro_sampler(
    runtime_context: RuntimeContext,
    *,
    logs_dir: Path | None = None,
    post_warmup_state: Any | None = None,
) -> NumPyroSamplerResult:
    """
    Run NumPyro NUTS for the configured number of warmup and sample draws.

    Resume support is implemented through NumPyro's `post_warmup_state`
    contract: when provided, warmup/adaptation is skipped and the sampler
    continues from the serialized state stored in the run checkpoint.
    """

    sampling = runtime_context.config.sampling
    rng_key = jax.random.PRNGKey(int(sampling.random_seed))
    kernel = NUTS(
        numpyro_model,
        init_strategy=_build_jittered_initial_strategy(runtime_context),
        target_accept_prob=0.8,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=int(sampling.num_warmup),
        num_samples=int(sampling.num_samples),
        num_chains=int(sampling.num_chains),
        thinning=int(sampling.thinning),
        chain_method=str(sampling.chain_method),
        progress_bar=bool(runtime_context.config.runtime.progress),
    )
    if post_warmup_state is not None:
        mcmc.post_warmup_state = post_warmup_state

    if logs_dir is not None:
        append_run_log(
            logs_dir,
            "numpyro start | "
            f"chains {sampling.num_chains} | warmup {sampling.num_warmup} | "
            f"samples {sampling.num_samples} | chain_method {sampling.chain_method}",
        )

    mcmc.run(
        rng_key,
        runtime_context,
        extra_fields=("potential_energy", "accept_prob", "diverging"),
    )
    samples_dict = mcmc.get_samples(group_by_chain=True)
    extra_fields = {
        key: np.asarray(value)
        for key, value in mcmc.get_extra_fields(group_by_chain=True).items()
    }
    parameter_names = runtime_context.config.parameter_schema.internal_parameter_names
    samples_by_chain = _samples_to_ordered_array(samples_dict, parameter_names)
    flat_samples = samples_by_chain.reshape(-1, samples_by_chain.shape[-1])
    log_prob_by_chain = _extract_log_prob_by_chain(extra_fields, samples_by_chain)
    acceptance_fraction_mean = _acceptance_fraction(extra_fields)

    diagnostics = {
        "extra_fields": extra_fields,
        "num_divergences": int(np.count_nonzero(extra_fields.get("diverging", np.asarray([], dtype=bool)))),
        "parameter_names": list(parameter_names),
    }
    if logs_dir is not None:
        append_run_log(
            logs_dir,
            "numpyro complete | "
            f"accept_prob {acceptance_fraction_mean:.4f} | "
            f"divergences {diagnostics['num_divergences']}",
        )

    return NumPyroSamplerResult(
        samples_by_chain=samples_by_chain,
        flat_samples=flat_samples,
        log_prob_by_chain=log_prob_by_chain,
        acceptance_fraction_mean=acceptance_fraction_mean,
        diagnostics=diagnostics,
        last_state=mcmc.last_state,
    )


__all__ = [
    "NumPyroSamplerResult",
    "numpyro_model",
    "run_numpyro_sampler",
]
