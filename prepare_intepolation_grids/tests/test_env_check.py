"""Tests for the standard-environment checks.

The project now treats the `cmass_lens` conda environment as the only standard
runtime and verification environment. These tests lock down the user-facing
checks so environment failures are reported early and clearly.
"""

from __future__ import annotations

from interpolation_grids.env_check import check_environment


def test_check_environment_reports_missing_dependencies() -> None:
    """The checker should explain exactly which imports are missing."""

    result = check_environment(
        expected_env_name="cmass_lens",
        current_prefix="/opt/homebrew/anaconda3/envs/cmass_lens",
        import_checker=lambda name: name not in {"pytest", "spherical_jeans"},
    )

    assert not result.is_valid
    assert result.detected_env_name == "cmass_lens"
    assert any("pytest" in message for message in result.messages)
    assert any("spherical_jeans" in message for message in result.messages)


def test_check_environment_accepts_the_expected_environment_when_imports_exist() -> None:
    """A healthy `cmass_lens` environment should pass with no messages."""

    result = check_environment(
        expected_env_name="cmass_lens",
        current_prefix="/opt/homebrew/anaconda3/envs/cmass_lens",
        import_checker=lambda name: True,
    )

    assert result.is_valid
    assert result.detected_env_name == "cmass_lens"
    assert result.messages == []


def test_check_environment_rejects_non_standard_conda_environment() -> None:
    """The checker should fail fast when the active environment is not `cmass_lens`."""

    result = check_environment(
        expected_env_name="cmass_lens",
        current_prefix="/opt/homebrew/anaconda3/envs/other_env",
        import_checker=lambda name: True,
    )

    assert not result.is_valid
    assert result.detected_env_name == "other_env"
    assert any("cmass_lens" in message for message in result.messages)
