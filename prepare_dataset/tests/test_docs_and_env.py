"""Tests that keep the documented standard environment contract in sync."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_environment_file_declares_cmass_lens_and_required_dependencies() -> None:
    """The repo should declare the standard conda environment explicitly."""

    content = (PROJECT_ROOT / "environment.yml").read_text()

    assert "name: cmass_lens" in content
    assert "- numpy" in content
    assert "- scipy" in content
    assert "- astropy" in content
    assert "- h5py" in content
    assert "- pytest" in content


def test_readme_documents_cmass_lens_commands() -> None:
    """The README should teach users to run through the standard conda env."""

    content = (PROJECT_ROOT / "README.md").read_text()

    assert "cmass_lens" in content
    assert "conda run -n cmass_lens python -m prepare_dataset.env_check" in content
    assert "conda run -n cmass_lens pytest -q" in content
    assert "spherical_jeans" in content
