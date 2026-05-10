"""
Production Numba backend package.

This package owns the compiled numerical backend used by production inference.
It intentionally keeps sampler orchestration outside the package: Numba kernels
compute posterior components, while `emcee_sampler.py` controls walker
evolution and HDF5 persistence.
"""

__all__: list[str] = []
