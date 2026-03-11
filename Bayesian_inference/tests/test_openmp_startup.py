"""
Tests for OpenMP startup-hook behavior.

The goal is intentionally narrow: make sure OpenMP warning suppression happens
early enough in Python startup that `numba.set_num_threads()` does not emit the
deprecated `omp_set_nested` runtime info message. These tests do not touch the
scientific model or thread-budget policy.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _execute_module_from_path(module_path: Path, module_name: str) -> None:
    """
    Execute a Python file under an isolated module name.

    This helper lets the tests inspect startup-hook behavior without relying on
    Python's implicit `sitecustomize` import cache.
    """

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_root_sitecustomize_sets_openmp_defaults_without_overwriting(monkeypatch) -> None:
    """
    The repository-root startup hook should establish safe OpenMP defaults via
    `setdefault`, preserving any values the user explicitly exported.
    """

    monkeypatch.delenv("OMP_MAX_ACTIVE_LEVELS", raising=False)
    monkeypatch.delenv("KMP_WARNINGS", raising=False)

    _execute_module_from_path(PROJECT_ROOT / "sitecustomize.py", "test_root_sitecustomize_defaults")

    assert os.environ["OMP_MAX_ACTIVE_LEVELS"] == "1"
    assert os.environ["KMP_WARNINGS"] == "0"

    monkeypatch.setenv("OMP_MAX_ACTIVE_LEVELS", "7")
    monkeypatch.setenv("KMP_WARNINGS", "custom")
    _execute_module_from_path(PROJECT_ROOT / "sitecustomize.py", "test_root_sitecustomize_preserve")

    assert os.environ["OMP_MAX_ACTIVE_LEVELS"] == "7"
    assert os.environ["KMP_WARNINGS"] == "custom"


def test_scripts_sitecustomize_matches_root_openmp_defaults(monkeypatch) -> None:
    """
    Script-direct execution needs its own startup hook because `python
    scripts/foo.py` starts with `scripts/` on the import path before the repo
    root. The script-level hook must therefore exist and apply the same safe
    defaults.
    """

    monkeypatch.delenv("OMP_MAX_ACTIVE_LEVELS", raising=False)
    monkeypatch.delenv("KMP_WARNINGS", raising=False)

    _execute_module_from_path(PROJECT_ROOT / "scripts" / "sitecustomize.py", "test_scripts_sitecustomize_defaults")

    assert os.environ["OMP_MAX_ACTIVE_LEVELS"] == "1"
    assert os.environ["KMP_WARNINGS"] == "0"


def test_repo_root_python_startup_no_longer_emits_omp_set_nested_warning() -> None:
    """
    Running Python from the repository root should import the startup hook
    early enough that the OpenMP deprecation info line does not appear.
    """

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import numba; numba.set_num_threads(12); print('ok')",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "ok"
    assert "omp_set_nested routine deprecated" not in completed.stderr
