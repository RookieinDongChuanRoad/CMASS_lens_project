"""Reusable Numba diagnostics orchestration for posterior predictive workflows.

Model adapters still own their scientific kernels because theta layout,
population state, and output payloads are model-specific.  This module owns the
small but important backend mechanics that should be shared across adapters:
thread-policy application, draw chunking, dense array adaptation, and chunk
output concatenation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from statistical_sl.numerics.numba.runtime import apply_thread_limits


def apply_diagnostics_thread_policy(
    execution: Any | None,
    *,
    thread_limiter=apply_thread_limits,
) -> None:
    """
    Apply the generic PPC execution policy before running model-owned kernels.

    ``DiagnosticsExecution`` is a workflow-level config object, so this helper
    accepts a structural object with ``kernel_threads_per_process``. Keeping the
    dependency structural prevents the backend namespace from importing
    workflow orchestration modules.  ``thread_limiter`` is injectable so
    tests can prevent process-global Numba thread mutations while still
    verifying that adapters hand off the resolved width.
    """

    if execution is None:
        return
    thread_limiter(int(execution.kernel_threads_per_process))


def iter_draw_chunks(n_draws: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    """
    Yield half-open posterior-draw slices for one diagnostics run.

    Chunking is centralized so CMASS and Sonnenfeld diagnostics use the same
    edge-case behavior and validation.  ``chunk_size`` is clamped to at least
    one because a zero-size chunk would otherwise produce an infinite loop.
    """

    resolved_n_draws = max(0, int(n_draws))
    resolved_chunk_size = max(1, int(chunk_size))
    for chunk_start in range(0, resolved_n_draws, resolved_chunk_size):
        yield chunk_start, min(chunk_start + resolved_chunk_size, resolved_n_draws)


def as_float64_contiguous(array: Any) -> np.ndarray:
    """Return a C-contiguous ``float64`` view or copy for Numba kernels."""

    return np.ascontiguousarray(array, dtype=np.float64)


def as_int64_contiguous(array: Any) -> np.ndarray:
    """Return a C-contiguous ``int64`` view or copy for Numba kernels."""

    return np.ascontiguousarray(array, dtype=np.int64)


def concatenate_chunk_outputs(
    chunk_outputs: Sequence[tuple[np.ndarray, ...]],
    array_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """
    Concatenate tuple-based kernel outputs into a named dictionary.

    Numba kernels return tuples for speed and type stability, while artifact
    writers need named arrays.  The validation here fails early if a model
    adapter's output-name contract drifts from its kernel return shape.
    """

    if not chunk_outputs:
        raise ValueError("Cannot concatenate diagnostics outputs because no chunks were produced.")

    expected_width = len(array_names)
    for chunk_index, chunk in enumerate(chunk_outputs):
        if len(chunk) != expected_width:
            raise ValueError(
                "Diagnostics chunk output width mismatch: "
                f"chunk {chunk_index} returned {len(chunk)} arrays, expected {expected_width}."
            )

    return {
        array_name: np.concatenate([chunk[index] for chunk in chunk_outputs], axis=0)
        for index, array_name in enumerate(array_names)
    }


__all__ = [
    "apply_diagnostics_thread_policy",
    "as_float64_contiguous",
    "as_int64_contiguous",
    "concatenate_chunk_outputs",
    "iter_draw_chunks",
]
