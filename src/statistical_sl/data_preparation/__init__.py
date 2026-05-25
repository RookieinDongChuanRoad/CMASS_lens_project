"""Dataset preparation toolkit.

This package turns the previously scattered grid-generation and dataset-writing
logic into a small, testable codebase. The public surface intentionally stays
small:

- configuration constants live in :mod:`statistical_sl.data_preparation.config`
- numerical kernels live in :mod:`statistical_sl.data_preparation.physics`
- HDF5 batch processing lives in :mod:`statistical_sl.data_preparation.io.hdf5`
- canonical inference dataset writers live in :mod:`statistical_sl.data_preparation.dataset_schema`
- the command-line entrypoint lives in :mod:`statistical_sl.data_preparation.cli`
"""

from .config import DEFAULT_INPUT_FILENAMES, DERIVATIVE_DATASET_NAME, GAMMA_GRID

__all__ = [
    "DEFAULT_INPUT_FILENAMES",
    "DERIVATIVE_DATASET_NAME",
    "GAMMA_GRID",
]
