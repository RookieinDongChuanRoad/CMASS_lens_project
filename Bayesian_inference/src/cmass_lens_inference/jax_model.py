"""
Removed JAX entrypoint.

The production backend is now registry-driven.  Import from
``cmass_lens_inference.jax_backend.likelihood_engine`` instead of using this
legacy module.
"""

from __future__ import annotations


def _removed_entrypoint(*_args, **_kwargs):
    """Raise a targeted error for stale imports during the breaking refactor."""

    raise RuntimeError(
        "cmass_lens_inference.jax_model is no longer a supported public entrypoint. "
        "Use cmass_lens_inference.jax_backend.likelihood_engine instead."
    )


build_jax_model = _removed_entrypoint
log_prob_value = _removed_entrypoint
log_prob = _removed_entrypoint

__all__ = ["build_jax_model", "log_prob", "log_prob_value"]
