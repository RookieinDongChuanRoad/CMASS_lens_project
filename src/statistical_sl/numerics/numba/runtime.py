"""Runtime helpers shared by Numba-backed workflows.

These helpers are deliberately lower-level than inference or posterior
predictive workflows.  They only describe process-local math-library thread
limits, which both emcee inference and PPC diagnostics need when Numba kernels
run alongside Python orchestration.
"""

from __future__ import annotations

import os


THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def apply_thread_limits(num_threads: int) -> None:
    """
    Clamp common math-library thread counts and numba worker counts.

    This helper is intentionally tolerant of optional Numba runtime failures:
    setting environment variables remains useful even if ``numba.set_num_threads``
    is unavailable or rejects the requested value in a particular process.
    """

    clamped_threads = str(max(1, int(num_threads)))
    for variable_name in THREAD_LIMIT_ENV_VARS:
        os.environ[variable_name] = clamped_threads
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    os.environ["KMP_WARNINGS"] = "0"

    try:
        import numba

        numba.set_num_threads(int(clamped_threads))
    except Exception:
        pass


__all__ = ["THREAD_LIMIT_ENV_VARS", "apply_thread_limits"]
