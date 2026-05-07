"""Regression tests for the physical removal of JAX and NumPyro backends.

These tests deliberately inspect source files instead of importing optional
packages.  The goal is to prevent a retired backend from silently returning as
an import-time dependency, production module, or optional packaging extra.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src" / "cmass_lens_inference"
PYPROJECT_PATH = PACKAGE_ROOT / "pyproject.toml"
RETIRED_TOP_LEVEL_IMPORTS = {"jax", "numpyro"}


def _python_source_files() -> list[Path]:
    """Return every production Python source file under the package root."""

    return sorted(SOURCE_ROOT.rglob("*.py"))


def _top_level_module_name(imported_name: str) -> str:
    """Normalize `jax.numpy` and similar imports to their root package name."""

    return imported_name.split(".", maxsplit=1)[0]


def test_production_source_does_not_import_jax_or_numpyro() -> None:
    """Production package source must not import retired JAX/NumPyro modules."""

    offenders: list[str] = []
    for source_path in _python_source_files():
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _top_level_module_name(alias.name) in RETIRED_TOP_LEVEL_IMPORTS:
                        offenders.append(f"{source_path.relative_to(PACKAGE_ROOT)} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if _top_level_module_name(node.module) in RETIRED_TOP_LEVEL_IMPORTS:
                    offenders.append(f"{source_path.relative_to(PACKAGE_ROOT)} imports from {node.module}")

    assert offenders == []


def test_package_metadata_does_not_expose_jax_or_numpyro_extras() -> None:
    """The install metadata should expose only numba/emcee production deps."""

    payload = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    dependencies = payload["project"].get("dependencies", [])
    optional_dependencies = payload["project"].get("optional-dependencies", {})
    serialized = "\n".join(
        [str(item) for item in dependencies]
        + [
            f"{extra_name}: {dependency}"
            for extra_name, extra_dependencies in optional_dependencies.items()
            for dependency in extra_dependencies
        ]
    ).lower()

    assert "jax" not in optional_dependencies
    assert "jax" not in serialized
    assert "numpyro" not in serialized
