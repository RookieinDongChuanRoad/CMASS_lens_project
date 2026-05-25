"""
Prepare the isolated `key_tests` workspace for dual-pipeline comparisons.

This setup step is intentionally idempotent. Re-running it should refresh the
reference code copy, regenerate current-pipeline configs, and keep the data
links pointing at the canonical CMASS project data directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from workspace_support import (
    CURRENT_PROJECT_ROOT,
    REFERENCE_REQUIRED_FILES,
    REFERENCE_SOURCE_ROOT,
    WORKSPACE_ROOT,
    build_current_config_payload,
    build_reference_run_spec,
    workspace_subdirs,
)


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    """Write one configuration file with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _copy_reference_sources(reference_pipeline_dir: Path) -> list[Path]:
    """Copy the minimum viable legacy source files into the isolated workspace."""

    copied_paths: list[Path] = []
    for filename in REFERENCE_REQUIRED_FILES:
        source_path = REFERENCE_SOURCE_ROOT / filename
        destination_path = reference_pipeline_dir / filename
        shutil.copy2(source_path, destination_path)
        copied_paths.append(destination_path)
    return copied_paths


def _refresh_data_link(destination_path: Path, source_path: Path) -> None:
    """Create or replace one symlink used by the copied legacy scripts."""

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        destination_path.unlink()
    destination_path.symlink_to(source_path)


def prepare_workspace() -> dict[str, object]:
    """
    Create every deterministic artifact required before any run starts.

    The function prepares directory skeletons, writes current-pipeline configs,
    copies the legacy reference code, and maps the expected legacy `./data/...`
    filenames onto the canonical CMASS project data files.
    """

    subdirs = workspace_subdirs()
    for directory in subdirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    copied_reference_files = _copy_reference_sources(subdirs["reference_pipeline"])

    configs_written: list[Path] = []
    for profile_name in ("sersic", "devauc"):
        for mode_name in ("smoke", "compare"):
            current_output_root = WORKSPACE_ROOT / "output" / "current" / profile_name / mode_name
            config_path = subdirs["configs"] / f"current_{profile_name}_{mode_name}.yaml"
            _write_yaml(
                config_path,
                build_current_config_payload(
                    profile_name=profile_name,
                    mode_name=mode_name,
                    output_root=current_output_root,
                ),
            )
            configs_written.append(config_path)

            reference_spec = build_reference_run_spec(
                profile_name=profile_name,
                mode_name=mode_name,
                output_root=WORKSPACE_ROOT / "output" / "reference" / profile_name / mode_name,
            )
            for filename, target_path in reference_spec.data_links.items():
                _refresh_data_link(subdirs["reference_pipeline"] / "data" / filename, target_path)

    return {
        "workspace_root": str(WORKSPACE_ROOT),
        "current_project_root": str(CURRENT_PROJECT_ROOT),
        "reference_source_root": str(REFERENCE_SOURCE_ROOT),
        "copied_reference_files": [str(path) for path in copied_reference_files],
        "configs_written": [str(path) for path in configs_written],
    }


if __name__ == "__main__":
    import json

    print(json.dumps(prepare_workspace(), indent=2, sort_keys=True))
