from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the package CLI from the current checkout with isolated environment."""

    repository_root = Path(__file__).resolve().parents[1]
    subprocess_environment = os.environ.copy()
    subprocess_environment["PYTHONPATH"] = str(repository_root / "src")
    return subprocess.run(
        [sys.executable, "-m", "statistical_sl.cli", *arguments],
        check=False,
        capture_output=True,
        env=subprocess_environment,
        text=True,
    )


def test_import_statistical_sl() -> None:
    """The public package namespace must be importable from the root project."""

    import statistical_sl  # noqa: F401


def test_model_and_predictive_registries_import() -> None:
    """Core registries must resolve representative model contracts."""

    from statistical_sl.models.registry import get_model_definition
    from statistical_sl.posterior_predictive.registry import get_predictive_definition

    assert get_model_definition("toy_hierarchical").name == "toy_hierarchical"
    assert get_predictive_definition("cmass").model_name == "cmass"


def test_statistical_sl_cli_help() -> None:
    """The root CLI should expose the three canonical workflow groups."""

    result = _run_cli("--help")

    assert result.returncode == 0
    assert "prepare-dataset" in result.stdout
    assert "inference" in result.stdout
    assert "posterior-predictive" in result.stdout


def test_statistical_sl_workflow_help_pages() -> None:
    """Each workflow command should forward help to its package-owned parser."""

    for workflow_name in ("prepare-dataset", "inference", "posterior-predictive"):
        result = _run_cli(workflow_name, "--help")
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()
