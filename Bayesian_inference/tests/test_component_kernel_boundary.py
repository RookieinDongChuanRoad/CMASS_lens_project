"""Static boundary tests for the locked component/kernel/model split."""

from __future__ import annotations

from pathlib import Path
import ast


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "cmass_lens_inference"
MODEL_TOKENS = ("cmass", "CMASS", "sonnenfeld", "Sonnenfeld", "SLACS")


def _python_sources(root: Path) -> list[Path]:
    """Return Python sources under a package root, excluding bytecode caches."""

    return [
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def test_components_do_not_contain_production_model_content() -> None:
    """The reusable component repository must not mention production models."""

    offenders: list[tuple[str, str]] = []
    for source_path in _python_sources(PACKAGE_ROOT / "components"):
        text = source_path.read_text(encoding="utf-8")
        for token in MODEL_TOKENS:
            if token in text or token in source_path.name:
                offenders.append((str(source_path.relative_to(PACKAGE_ROOT)), token))

    assert offenders == []


def test_numba_backend_does_not_contain_production_model_content() -> None:
    """The backend and shared kernels must stay model-agnostic."""

    offenders: list[tuple[str, str]] = []
    for source_path in _python_sources(PACKAGE_ROOT / "numba_backend"):
        text = source_path.read_text(encoding="utf-8")
        for token in MODEL_TOKENS:
            if token in text or token in source_path.name:
                offenders.append((str(source_path.relative_to(PACKAGE_ROOT)), token))

    assert offenders == []


def test_models_do_not_use_legacy_component_or_production_paths() -> None:
    """Model-specific code should live under concrete model packages."""

    assert list((PACKAGE_ROOT / "models" / "components").rglob("*.py")) == []
    assert list((PACKAGE_ROOT / "models").rglob("production.py")) == []


def test_model_posteriors_are_single_file_assemblies() -> None:
    """A model-owned posterior should keep its private kernels in the same file."""

    assert list((PACKAGE_ROOT / "models").rglob("posterior_kernels.py")) == []


def test_sonnenfeld_posterior_reuses_generic_population_kernels() -> None:
    """Sonnenfeld posterior should not duplicate generic population math."""

    posterior_path = PACKAGE_ROOT / "models" / "sonnenfeld2024_slacs" / "posterior.py"
    source_tree = ast.parse(posterior_path.read_text(encoding="utf-8"))
    function_names = {
        node.name
        for node in source_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "_size_relation_mean" not in function_names
    assert "_parent_density_for_draw" not in function_names
    assert "_active_truncation_mass_threshold" not in function_names
