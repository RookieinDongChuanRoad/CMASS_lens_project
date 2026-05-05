"""Canonical inference dataset schema helpers.

The schema package belongs to data preparation, not Bayesian inference.  It
defines the names and writer utilities used to turn existing CMASS-style raw
products into one inference-ready HDF5 dataset.
"""

from .canonical import (
    CANONICAL_SCHEMA_VERSION,
    TOP_LEVEL_BLOCKS,
)
from .writer import write_canonical_inference_dataset

__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "TOP_LEVEL_BLOCKS",
    "write_canonical_inference_dataset",
]
