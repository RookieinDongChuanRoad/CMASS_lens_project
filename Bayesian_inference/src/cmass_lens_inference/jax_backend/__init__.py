"""
Reusable JAX backend package.

The package ``__init__`` intentionally does not import ``likelihood_engine``.
Model modules import ``jax_backend.primitives`` while the likelihood engine
imports the model registry; eager re-exporting would create a circular import
during configuration loading.  Callers should import concrete backend modules,
for example ``cmass_lens_inference.jax_backend.likelihood_engine``.
"""

__all__: list[str] = []
