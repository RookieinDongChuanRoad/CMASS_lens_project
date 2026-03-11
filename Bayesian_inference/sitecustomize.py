"""
Development-time Python path bootstrap.

This project uses a `src/` layout. When commands are executed directly from the
repository root with `python -m ...`, Python does not automatically add
`src/` to `sys.path`. Importing `sitecustomize` is standard Python behavior, so
this file provides a zero-configuration bridge that keeps local CLI execution
working before the package is installed into an environment.

This file is also the safest place to establish OpenMP runtime defaults that
must exist before `numba` is imported. Setting them here prevents the noisy
`omp_set_nested` deprecation info line without changing the actual thread
budgeting logic, which still lives in `parallel.py`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# Use `setdefault` so an explicit user override still wins. The point of this
# hook is only to make the common local-development path quiet and stable.
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
os.environ.setdefault("KMP_WARNINGS", "0")

REPOSITORY_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
