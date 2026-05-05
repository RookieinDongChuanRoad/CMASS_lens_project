"""Dataset preparation toolkit.

This package turns the previously scattered grid-generation and dataset-writing
logic into a small, testable codebase. The public surface intentionally stays
small:

- configuration constants live in :mod:`prepare_dataset.config`
- numerical kernels live in :mod:`prepare_dataset.physics`
- HDF5 batch processing lives in :mod:`prepare_dataset.io.hdf5`
- canonical inference dataset writers live in :mod:`prepare_dataset.dataset_schema`
- the command-line entrypoint lives in :mod:`prepare_dataset.cli`
"""

from .config import DEFAULT_INPUT_FILENAMES, DERIVATIVE_DATASET_NAME, GAMMA_GRID

__all__ = [
    "DEFAULT_INPUT_FILENAMES",
    "DERIVATIVE_DATASET_NAME",
    "GAMMA_GRID",
]
