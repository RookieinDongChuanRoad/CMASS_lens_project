"""Shared artifact filename contract for Statistical_SL workflows.

These constants define the filenames that independent workflow stages should
use when exchanging run outputs.  They do not perform any file I/O; concrete
writers remain in their owning workflow modules until later migration phases.
"""

from __future__ import annotations

INFERENCE_CHAIN_FILENAME = "chain.h5"
INFERENCE_RUN_RESULT_FILENAME = "run_result.json"
POSTERIOR_CORNER_FILENAME = "posterior_corner.png"
POSTERIOR_CORNER_RESULT_FILENAME = "posterior_corner_result.json"
PPC_SUMMARY_FILENAME = "ppc_summary.json"
REPLICATED_STATISTICS_FILENAME = "replicated_statistics.npz"

__all__ = [
    "INFERENCE_CHAIN_FILENAME",
    "INFERENCE_RUN_RESULT_FILENAME",
    "POSTERIOR_CORNER_FILENAME",
    "POSTERIOR_CORNER_RESULT_FILENAME",
    "PPC_SUMMARY_FILENAME",
    "REPLICATED_STATISTICS_FILENAME",
]
