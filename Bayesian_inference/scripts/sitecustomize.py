"""
Script-directory startup hook for local utility entrypoints.

Why this file exists:
- `python scripts/foo.py` starts Python with `scripts/` at the front of the
  import path, so the repository-root `sitecustomize.py` is not guaranteed to
  be the first startup hook Python sees.
- The OpenMP warning cleanup depends on setting environment variables before
  any `numba` import happens.

This hook intentionally stays tiny and side-effect-light:
- it does not import project modules
- it does not set thread counts
- it only establishes safe OpenMP defaults if the user did not already export
  their own values
"""

from __future__ import annotations

import os


os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
os.environ.setdefault("KMP_WARNINGS", "0")
