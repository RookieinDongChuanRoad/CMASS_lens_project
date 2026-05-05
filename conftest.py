"""
Pytest bootstrap for the multi-package CMASS lens workspace.

The repository contains several Python projects that are normally tested from
their own package roots.  Running ``pytest`` from the repository root is useful
for migration verification, but the root directory does not install those
projects as editable packages first.  This bootstrap makes the root-level test
run explicit and reproducible by adding only the local source directories that
the checked-in tests import directly.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent

# Keep this list narrow: each entry corresponds to a project-local package or a
# test-support module that is imported by the root-level test collection.  The
# order preserves the existing package-local behavior while avoiding accidental
# imports from unrelated generated output directories.
LOCAL_IMPORT_ROOTS = (
    REPOSITORY_ROOT / "Bayesian_inference" / "src",
    REPOSITORY_ROOT / "Posterior_predictive_test" / "src",
    REPOSITORY_ROOT / "prepare_dataset",
    REPOSITORY_ROOT / "key_tests",
)

for import_root in reversed(LOCAL_IMPORT_ROOTS):
    # Missing directories should not happen in this repository, but the guard
    # keeps pytest collection readable if a future partial checkout omits one of
    # the optional workflow folders.
    if import_root.exists() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
