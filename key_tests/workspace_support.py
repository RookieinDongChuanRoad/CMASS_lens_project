"""
Workspace support helpers for the isolated `key_tests` harness.

This module centralizes every convention that would otherwise be duplicated
across wrappers, drivers, notebook utilities, and reports:

- fixed filesystem roots
- per-profile input data paths
- smoke/compare runtime settings
- current-pipeline YAML payload generation
- reference-pipeline entrypoint mapping

Keeping these rules in one importable module makes the harness easier to test
and safer to evolve. The user specifically wants the `key_tests` workspace to
be isolated from the original inference project, so this module becomes the
single source of truth for all paths and run layouts inside that workspace.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final


WORKSPACE_ROOT: Final[Path] = Path("/Users/liurongfu/Work/CMASS_lens_project/key_tests")
CURRENT_PROJECT_ROOT: Final[Path] = Path("/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference")
REFERENCE_SOURCE_ROOT: Final[Path] = Path("/Users/liurongfu/Desktop/Spectrum_reduction")
DATA_ROOT: Final[Path] = Path("/Users/liurongfu/Work/CMASS_lens_project/data")
TOOLS_ROOT: Final[Path] = Path("/Users/liurongfu/tools")


@dataclass(frozen=True)
class ModeSettings:
    """
    Runtime controls for one execution mode.

    `smoke` verifies the pipeline wiring with the smallest useful chain.
    `compare` produces the long-run chain used for the actual side-by-side
    comparison between the current and legacy implementations.
    """

    name: str
    n_steps: int
    warmup: int
    checkpoint_every: int
    discard: int


@dataclass(frozen=True)
class ProfileSettings:
    """
    Stable input locations for one profile branch.

    The harness always points at the CMASS project data directory instead of
    the reference tree's bundled data copies so both implementations consume
    the same inputs.
    """

    name: str
    observation_path: Path
    cross_section_path: Path


@dataclass(frozen=True)
class ReferenceRunSpec:
    """
    Fully resolved execution spec for one reference-pipeline run.

    The driver consumes this structure directly so profile selection, output
    layout, and legacy-module dispatch stay explicit and testable.
    """

    profile_name: str
    mode_name: str
    module_name: str
    log_prob_function_name: str
    n_walkers: int
    n_steps: int
    pool_processes: int
    output_dir: Path
    backend_path: Path
    run_summary_path: Path
    data_links: dict[str, Path]


CURRENT_INITIAL_CENTER: Final[dict[str, float]] = {
    "mu5_0": 11.32,
    "beta5": 0.59,
    "xi5": -0.11,
    "sigma5": 0.06,
    "mu_gamma_0": 1.99,
    "beta_gamma": 0.10,
    "xi_gamma": -0.67,
    "sigma_gamma": 0.149,
    "mu_zs": 1.8,
    "sigma_zs": 0.215,
    "theta0": 0.93,
    "loga": 1.0,
}

CURRENT_GAMMA_MODE: Final[str] = "dependent"

CURRENT_PARAMETER_ORDER: Final[tuple[str, ...]] = (
    "mu5_0",
    "beta5",
    "xi5",
    "sigma5",
    "mu_gamma_0",
    "beta_gamma",
    "xi_gamma",
    "sigma_gamma",
    "mu_zs",
    "sigma_zs",
    "theta0",
    "loga",
)

REFERENCE_PARAMETER_ORDER: Final[tuple[str, ...]] = (
    "mu5_0",
    "beta5",
    "xi5",
    "sigma5",
    "mu_gamma_0",
    "beta_gamma",
    "xi_gamma",
    "sigma_gamma",
    "mu_zs",
    "sigma_zs",
    "loga",
    "theta0",
)

NOTEBOOK_PARAMETER_LABELS: Final[list[str]] = [
    r"$\mu_{5,0}$",
    r"$\beta_5$",
    r"$\xi_5$",
    r"$\sigma_5$",
    r"$\mu_\gamma$",
    r"$\beta_\gamma$",
    r"$\xi_\gamma$",
    r"$\sigma_\gamma$",
    r"$\mu_s$",
    r"$\sigma_s$",
    r"$\theta_0$",
    r"$\log a$",
]

MODE_SETTINGS: Final[dict[str, ModeSettings]] = {
    "smoke": ModeSettings(
        name="smoke",
        n_steps=4,
        warmup=0,
        checkpoint_every=2,
        discard=0,
    ),
    "compare": ModeSettings(
        name="compare",
        n_steps=10_000,
        warmup=2_000,
        checkpoint_every=500,
        discard=2_000,
    ),
}

PROFILE_SETTINGS: Final[dict[str, ProfileSettings]] = {
    "sersic": ProfileSettings(
        name="sersic",
        observation_path=DATA_ROOT / "raw" / "observations_with_m5_grids_all.hdf5",
        cross_section_path=DATA_ROOT / "external" / "cs_grid_power.h5",
    ),
    "devauc": ProfileSettings(
        name="devauc",
        observation_path=DATA_ROOT / "raw" / "observations_deV_with_m5_grids.hdf5",
        cross_section_path=DATA_ROOT / "external" / "cs_grid_power.h5",
    ),
}

REFERENCE_MODULE_BY_PROFILE: Final[dict[str, tuple[str, str]]] = {
    "sersic": ("Population_model", "P_d_eta_all"),
    "devauc": ("Population_deV_model", "P_d_eta"),
}

REFERENCE_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "Foreground_models.py",
    "Foreground_deV_models.py",
    "Population_model.py",
    "Population_deV_model.py",
    "constants.py",
    "constants_deV.py",
)

COMPARE_ARCHIVE_LABEL: Final[str] = "compare_80step"
COMPARE_FIGURE_FILENAMES: Final[tuple[str, ...]] = (
    "sersic_compare_corner.png",
    "devauc_compare_corner.png",
)


def _require_profile(profile_name: str) -> ProfileSettings:
    """Return the validated profile settings or raise a clear error."""

    normalized_name = profile_name.strip().lower()
    try:
        return PROFILE_SETTINGS[normalized_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported profile: {profile_name}") from exc


def _require_mode(mode_name: str) -> ModeSettings:
    """Return the validated mode settings or raise a clear error."""

    normalized_name = mode_name.strip().lower()
    try:
        return MODE_SETTINGS[normalized_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode: {mode_name}") from exc


def build_current_config_payload(profile_name: str, mode_name: str, output_root: Path) -> dict[str, object]:
    """
    Build the isolated current-pipeline YAML payload.

    The current implementation already supports a rich configuration surface, so
    the harness only needs to supply a deterministic, workspace-local variant of
    that config. The payload mirrors the production schema exactly to avoid
    special cases in the wrapper script.
    """

    profile = _require_profile(profile_name)
    mode = _require_mode(mode_name)

    return {
        "profile": {
            "name": profile.name,
        },
        "gamma_model": {
            "mode": CURRENT_GAMMA_MODE,
        },
        "data": {
            "observation_path": str(profile.observation_path),
            "cross_section_path": str(profile.cross_section_path),
        },
        "sampling": {
            "n_walkers": 24,
            "n_steps": mode.n_steps,
            "warmup": mode.warmup,
            "random_seed": 7,
            "initial_center": CURRENT_INITIAL_CENTER,
            "initial_jitter_scale": 1.0e-3,
        },
        "integration": {
            "gamma_points": 200,
            "mstar_points": 200,
            "normalization_samples": 100000,
        },
        "runtime": {
            "distance_table_max_z": 5.0,
            "distance_table_size": 8001,
            "checkpoint_every": mode.checkpoint_every,
            "parallel_strategy": "kernel_only",
            "progress": False,
            "progress_summary_every": 20,
            "show_stage_timing": False,
            "disable_hdf5_file_locking": False,
            "num_threads": 12,
            "reserve_cores": 2,
        },
        "output": {
            "root_dir": str(output_root),
            "run_label": mode.name,
            "overwrite_latest": True,
        },
    }


def build_reference_run_spec(profile_name: str, mode_name: str, output_root: Path) -> ReferenceRunSpec:
    """
    Resolve the legacy reference-pipeline execution contract for one run.

    The copied legacy modules still expect specific filenames under their local
    `./data` directory. This function translates the user-facing profile/mode
    choice into the exact module/function pair and data-link layout that the
    driver must prepare.
    """

    profile = _require_profile(profile_name)
    mode = _require_mode(mode_name)
    module_name, function_name = REFERENCE_MODULE_BY_PROFILE[profile.name]

    data_links = {
        "cs_grid_power.h5": profile.cross_section_path,
    }
    if profile.name == "sersic":
        data_links["observations_with_m5_grids_all.hdf5"] = profile.observation_path
    else:
        data_links["observations_deV_with_m5_grids.hdf5"] = profile.observation_path

    return ReferenceRunSpec(
        profile_name=profile.name,
        mode_name=mode.name,
        module_name=module_name,
        log_prob_function_name=function_name,
        n_walkers=24,
        n_steps=mode.n_steps,
        pool_processes=12,
        output_dir=output_root,
        backend_path=output_root / "reference_chain.h5",
        run_summary_path=output_root / "reference_run_summary.json",
        data_links=data_links,
    )


def current_center_as_reference_vector() -> list[float]:
    """
    Reorder the shared initial center into the legacy reference parameter order.

    The current implementation stores the final two parameters as
    `(theta0, loga)`, while the reference scripts expect `(loga, theta0)`.
    Returning the explicitly reordered vector here prevents silent mistakes in
    the driver and keeps the conversion logic unit-tested.
    """

    return [CURRENT_INITIAL_CENTER[name] for name in REFERENCE_PARAMETER_ORDER]


def workspace_subdirs() -> dict[str, Path]:
    """
    Return the canonical top-level directories for the isolated workspace.

    Keeping this in code avoids hand-maintained string duplication between the
    setup script, wrappers, notebook generator, and report writer.
    """

    return {
        "current_pipeline": WORKSPACE_ROOT / "current_pipeline",
        "reference_pipeline": WORKSPACE_ROOT / "reference_pipeline",
        "configs": WORKSPACE_ROOT / "configs",
        "output": WORKSPACE_ROOT / "output",
        "notebooks": WORKSPACE_ROOT / "notebooks",
        "reports": WORKSPACE_ROOT / "reports",
        "figures": WORKSPACE_ROOT / "output" / "figures",
        "archive": WORKSPACE_ROOT / "output" / "archive",
    }


def _summary_filename_for_implementation(implementation_name: str) -> str:
    """Return the canonical summary filename for one implementation branch."""

    if implementation_name == "current":
        return "current_run_summary.json"
    if implementation_name == "reference":
        return "reference_run_summary.json"
    raise ValueError(f"Unsupported implementation: {implementation_name}")


def _requested_steps_from_summary(summary_path: Path) -> int | None:
    """
    Read the requested-step count from one summary JSON when it exists.

    Returning `None` allows callers to treat incomplete directories as stale
    without crashing on partially written or missing metadata.
    """

    if not summary_path.exists():
        return None

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    requested_steps = payload.get("requested_steps")
    if requested_steps is None:
        return None
    return int(requested_steps)


def _move_to_archive(source_path: Path, workspace_root: Path, archive_root: Path) -> Path:
    """
    Move one workspace-local artifact into the fixed compare archive tree.

    The destination mirrors the original relative layout so archived outputs are
    still easy to inspect manually after the canonical compare paths are
    regenerated with long-run artifacts.
    """

    relative_path = source_path.relative_to(workspace_root)
    destination_path = archive_root / relative_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        if destination_path.is_dir() and not destination_path.is_symlink():
            shutil.rmtree(destination_path)
        else:
            destination_path.unlink()
    shutil.move(str(source_path), str(destination_path))
    return destination_path


def archive_existing_compare_artifacts(workspace_root: Path = WORKSPACE_ROOT) -> dict[str, object]:
    """
    Archive stale compare artifacts before regenerating long-run outputs.

    The user explicitly asked to keep the old 80-step compare outputs, but not
    under the canonical `output/.../compare` paths. This helper detects compare
    directories whose `requested_steps` no longer match the configured long-run
    value and moves those directories, plus the report/manifest/figure assets
    derived from them, under `output/archive/compare_80step/`.
    """

    archive_root = workspace_root / "output" / "archive" / COMPARE_ARCHIVE_LABEL
    archived_paths: list[str] = []
    stale_compare_found = False
    target_compare_steps = MODE_SETTINGS["compare"].n_steps

    for implementation_name in ("current", "reference"):
        summary_filename = _summary_filename_for_implementation(implementation_name)
        for profile_name in PROFILE_SETTINGS:
            compare_dir = workspace_root / "output" / implementation_name / profile_name / "compare"
            summary_path = compare_dir / summary_filename
            if not compare_dir.exists():
                continue

            requested_steps = _requested_steps_from_summary(summary_path)
            is_stale_compare = requested_steps != target_compare_steps
            if not is_stale_compare:
                continue

            stale_compare_found = True
            archived_paths.append(
                str(_move_to_archive(compare_dir, workspace_root=workspace_root, archive_root=archive_root))
            )

    if stale_compare_found:
        report_candidates = (
            workspace_root / "reports" / "pipeline_comparison.md",
            workspace_root / "reports" / "run_manifest.json",
        )
        for report_path in report_candidates:
            if report_path.exists():
                archived_paths.append(
                    str(_move_to_archive(report_path, workspace_root=workspace_root, archive_root=archive_root))
                )

        for figure_filename in COMPARE_FIGURE_FILENAMES:
            figure_path = workspace_root / "output" / "figures" / figure_filename
            if figure_path.exists():
                archived_paths.append(
                    str(_move_to_archive(figure_path, workspace_root=workspace_root, archive_root=archive_root))
                )

    return {
        "archived": stale_compare_found,
        "archive_root": str(archive_root),
        "archived_paths": archived_paths,
        "target_compare_steps": target_compare_steps,
    }
