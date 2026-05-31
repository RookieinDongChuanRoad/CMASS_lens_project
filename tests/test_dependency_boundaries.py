from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ForbiddenDependency:
    """A source token that would prove the new package still depends on old code.

    The integration plan defines the old workflow directories and the archived
    historical tree as non-production material.  This test turns that
    architectural rule into a cheap static guard.  It deliberately checks text
    rather than importing modules, because import-time checks can miss dormant
    wrappers, fallback paths, and configuration references that only activate
    under specific CLI arguments.
    """

    token: str
    reason: str


def _forbidden_dependencies() -> tuple[ForbiddenDependency, ...]:
    """Return forbidden tokens without making this test fail on its own source.

    The test file is the only place allowed to spell the policy tokens, so the
    strings are assembled from fragments.  Every other production, workspace,
    and default-test file must avoid these tokens literally.
    """

    old_inference_package = "cmass" + "_lens_inference"
    old_ppc_package = "lensing" + "_posterior_predictive"
    old_data_package = "prepare" + "_dataset"
    legacy_path = "legacy" + "/"

    return (
        ForbiddenDependency(old_inference_package, "old Bayesian inference package import"),
        ForbiddenDependency(old_ppc_package, "old posterior predictive package import"),
        ForbiddenDependency(old_data_package, "old data-preparation package identity"),
        ForbiddenDependency("Bayesian" + "_inference", "old Bayesian inference directory"),
        ForbiddenDependency("Posterior" + "_predictive_test", "old posterior predictive directory"),
        ForbiddenDependency("_legacy" + "_paths", "temporary sys.path bootstrap"),
        ForbiddenDependency("_legacy" + "_imports", "temporary re-export helper"),
        ForbiddenDependency("reexport" + "_module", "temporary legacy implementation wrapper"),
        ForbiddenDependency(legacy_path, "legacy cold archive dependency"),
    )


def _source_files_to_check(repository_root: Path) -> list[Path]:
    """Collect files that participate in the current package and workflow.

    ``docs/`` and the root README may keep historical migration notes, so they
    are intentionally outside this static gate.  Current workflow docs should be
    checked separately during final review because a human has to distinguish
    historical mentions from recommended runbook steps.
    """

    candidate_roots = (
        repository_root / "src",
        repository_root / "tests",
        repository_root / "workspace",
    )
    checked_suffixes = {".py", ".toml", ".yaml", ".yml", ".json", ".md"}
    boundary_test_path = Path(__file__).resolve()
    files: list[Path] = [repository_root / "pyproject.toml"]
    optional_root_files = (repository_root / "conftest.py",)
    files.extend(path for path in optional_root_files if path.exists())

    for candidate_root in candidate_roots:
        if not candidate_root.exists():
            continue
        for path in candidate_root.rglob("*"):
            if path == boundary_test_path:
                continue
            if path.is_file() and path.suffix in checked_suffixes:
                files.append(path)

    return sorted(files)


def test_current_code_paths_do_not_depend_on_old_or_legacy_sources() -> None:
    """Reject production/test/workspace dependencies on retired source trees."""

    repository_root = Path(__file__).resolve().parents[1]
    violations: list[str] = []

    for path in _source_files_to_check(repository_root):
        relative_path = path.relative_to(repository_root)
        source = path.read_text(encoding="utf-8")

        for forbidden_dependency in _forbidden_dependencies():
            if forbidden_dependency.token in source:
                violations.append(
                    f"{relative_path}: contains {forbidden_dependency.token!r} "
                    f"({forbidden_dependency.reason})"
                )

    assert not violations, "\n".join(violations)


def test_non_backend_packages_do_not_import_inference_private_backends() -> None:
    """Keep model and PPC packages off inference-private backend modules.

    The integration plan allows the inference workflow to own the active
    Numba/emcee backend, but model declarations and posterior-predictive
    adapters must not reach through that private namespace.  Shared helpers
    needed by both sides belong in public inference modules, ``numerics``, or
    model-owned code.
    """

    repository_root = Path(__file__).resolve().parents[1]
    checked_roots = (
        repository_root / "src" / "statistical_sl" / "models",
        repository_root / "src" / "statistical_sl" / "posterior_predictive",
    )
    forbidden_import = "statistical_sl.inference.backends"
    violations: list[str] = []

    for checked_root in checked_roots:
        for module_path in checked_root.rglob("*.py"):
            module_source = module_path.read_text(encoding="utf-8")
            if forbidden_import in module_source:
                violations.append(str(module_path.relative_to(repository_root)))

    assert not violations, "\n".join(violations)
