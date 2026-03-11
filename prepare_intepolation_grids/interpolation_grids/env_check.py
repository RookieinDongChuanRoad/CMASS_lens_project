"""Standard environment validation for the interpolation-grid project.

This module exists because the project now has an explicit operational
contract: the only supported runtime and verification environment is the conda
environment named `cmass_lens`.

The goal is not to auto-switch environments from Python. Instead, the checker
fails fast with actionable messages when someone runs the tool outside the
agreed environment or when critical dependencies are missing from that
environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys


REQUIRED_RUNTIME_MODULES = ("numpy", "scipy", "astropy", "h5py", "spherical_jeans")
REQUIRED_TEST_MODULES = ("pytest",)


@dataclass(frozen=True)
class EnvironmentCheckResult:
    """Result of validating the active Python environment."""

    is_valid: bool
    detected_env_name: str | None
    messages: list[str]


def detect_conda_env_name(current_prefix: str | None) -> str | None:
    """Infer the active conda environment name from `sys.prefix`.

    Why we use `sys.prefix`:
    - It is available even when the shell did not export environment variables.
    - It gives a stable answer under `conda run -n <env> ...`.
    """

    if not current_prefix:
        return None
    normalized = current_prefix.rstrip("/")
    if "/envs/" in normalized:
        return normalized.rsplit("/", 1)[-1]
    if normalized.endswith("/anaconda3") or normalized.endswith("/miniconda3"):
        return "base"
    return normalized.rsplit("/", 1)[-1] or None


def _default_import_checker(module_name: str) -> bool:
    """Return whether `module_name` can be imported in the active interpreter."""

    try:
        importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - user-facing env diagnostics should be broad.
        return False
    return True


def check_environment(
    expected_env_name: str = "cmass_lens",
    current_prefix: str | None = None,
    import_checker=None,
) -> EnvironmentCheckResult:
    """Validate that the active interpreter satisfies the project contract.

    Parameters
    ----------
    expected_env_name:
        The one supported conda environment name for this project.
    current_prefix:
        Optional override for `sys.prefix`, mainly used by tests.
    import_checker:
        Optional injectable import predicate for tests.
    """

    prefix = current_prefix if current_prefix is not None else sys.prefix
    import_checker = import_checker or _default_import_checker
    detected_env_name = detect_conda_env_name(prefix)
    messages: list[str] = []

    if detected_env_name != expected_env_name:
        messages.append(
            f"Expected conda environment '{expected_env_name}', but detected '{detected_env_name or 'unknown'}'."
        )

    for module_name in REQUIRED_RUNTIME_MODULES + REQUIRED_TEST_MODULES:
        if not import_checker(module_name):
            messages.append(
                f"Missing required module '{module_name}' in environment '{expected_env_name}'."
            )

    return EnvironmentCheckResult(
        is_valid=not messages,
        detected_env_name=detected_env_name,
        messages=messages,
    )


def main() -> int:
    """CLI entrypoint for `python -m interpolation_grids.env_check`."""

    result = check_environment()
    if result.is_valid:
        print("Environment check passed: cmass_lens is active and all required modules are importable.")
        return 0

    print("Environment check failed:")
    for message in result.messages:
        print(f"- {message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
