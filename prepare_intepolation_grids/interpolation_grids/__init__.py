"""Interpolation grid preparation toolkit.

This package turns the previously scattered grid-generation logic into a small,
testable codebase. The public surface intentionally stays small:

- configuration constants live in :mod:`interpolation_grids.config`
- numerical kernels live in :mod:`interpolation_grids.physics`
- HDF5 batch processing lives in :mod:`interpolation_grids.io.hdf5`
- the command-line entrypoint lives in :mod:`interpolation_grids.cli`
"""

from .config import DEFAULT_INPUT_FILENAMES, DERIVATIVE_DATASET_NAME, GAMMA_GRID

__all__ = [
    "DEFAULT_INPUT_FILENAMES",
    "DERIVATIVE_DATASET_NAME",
    "GAMMA_GRID",
]
