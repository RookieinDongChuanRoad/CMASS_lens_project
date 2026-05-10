#!/usr/bin/env python3
"""Audit and compare Sonnenfeld/SLACS against the external reference code.

The first implementation stage is intentionally audit-only.  The reference
repository contains scripts such as ``fit_full.py`` that execute data
preparation and MCMC at import time, so this harness must not import those
scripts just to learn whether the required artifacts exist.  Instead it records
a manifest that later comparison phases can trust before any numerical oracle
is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SRC = PROJECT_ROOT / "Bayesian_inference" / "src"
if str(INFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(INFERENCE_SRC))

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_SKIPPED_DATA_GATED = "skipped_data_gated"
STATUS_NOT_COMPARABLE = "not_comparable"
STATUS_OPTIONS = (
    STATUS_PASSED,
    STATUS_FAILED,
    STATUS_SKIPPED_DATA_GATED,
    STATUS_NOT_COMPARABLE,
)

REFERENCE_ARTIFACTS: tuple[tuple[str, bool], ...] = (
    ("SLACS_table.cat", True),
    ("parent_sample.fits", True),
    ("full_inference.hdf5", False),
    ("slonly_inference.hdf5", False),
    ("fibre_crosssect_grid.hdf5", True),
    ("slacs_lensing_grids.hdf5", True),
    ("slacs_jeans_grids.hdf5", True),
)

REFERENCE_SCRIPTS: tuple[str, ...] = (
    "scripts/fit_full.py",
    "scripts/fit_slonly.py",
    "scripts/mz_distribution.py",
    "scripts/fitpars.py",
    "scripts/make_crosssect_grid.py",
    "scripts/make_slacs_lensing_grids.py",
    "scripts/make_slacs_jeans_grids.py",
)

REFERENCE_PARAMETER_NAMES: tuple[str, ...] = (
    "mu_m5",
    "sigma_m5",
    "beta_m5",
    "xi_m5",
    "mu_gamma",
    "sigma_gamma",
    "beta_gamma",
    "xi_gamma",
    "mu_zs",
    "sigma_zs",
    "t_find",
    "la_find",
)

LOCAL_PARAMETER_NAMES: tuple[str, ...] = (
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

REFERENCE_TO_LOCAL_PARAMETER: dict[str, str] = {
    "mu_m5": "mu5_0",
    "sigma_m5": "sigma5",
    "beta_m5": "beta5",
    "xi_m5": "xi5",
    "mu_gamma": "mu_gamma_0",
    "sigma_gamma": "sigma_gamma",
    "beta_gamma": "beta_gamma",
    "xi_gamma": "xi_gamma",
    "mu_zs": "mu_zs",
    "sigma_zs": "sigma_zs",
    "t_find": "theta0",
    "la_find": "loga",
}


def parse_args() -> argparse.Namespace:
    """Parse the command-line interface for the comparison harness."""

    parser = argparse.ArgumentParser(
        description="Audit Sonnenfeld external-reference artifacts before numerical comparison.",
    )
    parser.add_argument(
        "--reference-root",
        required=True,
        help="Path to strong_lensing_tools/papers/slacs_selection.",
    )
    parser.add_argument(
        "--candidate-root",
        required=True,
        help="Path to the local CMASS_lens_project worktree.",
    )
    parser.add_argument(
        "--model",
        default="sonnenfeld2024_slacs",
        help="Local model name. Phase 0 accepts only sonnenfeld2024_slacs.",
    )
    parser.add_argument(
        "--candidate-dataset",
        default=None,
        help="Optional local canonical Sonnenfeld dataset to include in the manifest.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Write artifact manifest only; do not run numerical comparison.",
    )
    parser.add_argument(
        "--stages",
        default="",
        help="Comma-separated comparison stages. Phase 3 supports 'primitive'.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where manifest.json and summary.md will be written.",
    )
    return parser.parse_args()


def _close_enough(reference: float, local: float, *, rtol: float = 1.0e-10, atol: float = 1.0e-12) -> bool:
    """Return whether two scalar primitive values satisfy the default tolerance."""

    return bool(math.isclose(float(reference), float(local), rel_tol=rtol, abs_tol=atol))


def _scalar_comparison(name: str, reference: float, local: float) -> dict[str, Any]:
    """Build one scalar comparison record with absolute and relative errors."""

    abs_diff = abs(float(reference) - float(local))
    rel_diff = abs_diff / max(abs(float(reference)), 1.0e-300)
    return {
        "name": name,
        "status": STATUS_PASSED if _close_enough(reference, local) else STATUS_FAILED,
        "reference": float(reference),
        "local": float(local),
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "rtol": 1.0e-10,
        "atol": 1.0e-12,
    }


def reference_pfind(theta_est: float, theta0: float, la_find: float) -> float:
    """Return the reference discovery probability from ``fit_full.py``."""

    slope = 10.0**float(la_find)
    x = -slope * (float(theta_est) - float(theta0))
    if x > 60.0:
        return 0.0
    if x < -60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


def reference_source_redshift_mask(zd: float, zs: float) -> bool:
    """Return the source-redshift part of the reference population good mask."""

    return bool(float(zs) > float(zd) + 0.05 and float(zs) > 0.05 and float(zs) < 2.0)


def reference_size_relation(log_mstar: float) -> float:
    """Return Hyde-Bernardi quadratic size relation used by the reference."""

    value = float(log_mstar)
    return 7.55 - 1.84 * value + 0.110 * value**2


def reference_fp_prior_defaults() -> dict[str, float]:
    """Return FP-prior defaults from ``fitpars.py`` without importing it."""

    def mu_v_parent(log_mstar: float) -> float:
        offset = float(log_mstar) - 11.0
        return 2.2577 + 0.3034 * offset - 0.0761 * offset**2

    return {
        "fit_mstar_min": 11.0,
        "pivot_mstar": 11.3,
        "fiducial_scatter": 0.047,
        "scatter_error": 0.008,
        "mu_v_prior": mu_v_parent(11.3),
        "mu_v_error": 0.03,
        "beta_v_prior": (mu_v_parent(11.31) - mu_v_parent(11.29)) / 0.02,
        "beta_v_error": 0.03,
    }


def reference_fp_ols2_from_rows(
    *,
    mstar: np.ndarray,
    delta_r: np.ndarray,
    log_sigma: np.ndarray,
    pivot_mstar: float,
) -> tuple[float, float, float, float]:
    """Fit the two-predictor FP relation the same way as the reference script."""

    x1 = np.asarray(mstar, dtype=np.float64) - float(pivot_mstar)
    x2 = np.asarray(delta_r, dtype=np.float64)
    y = np.asarray(log_sigma, dtype=np.float64)
    design = np.column_stack([np.ones_like(x1), x1, x2])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = design @ coeffs - y
    scatter = float(np.sqrt(np.mean(residual * residual)))
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), scatter


def _fp_summary_from_rows(
    *,
    mstar: np.ndarray,
    delta_r: np.ndarray,
    log_sigma: np.ndarray,
    pivot_mstar: float,
) -> dict[str, float]:
    """Build sufficient moments expected by the local two-predictor FP solver."""

    x1 = np.asarray(mstar, dtype=np.float64) - float(pivot_mstar)
    x2 = np.asarray(delta_r, dtype=np.float64)
    y = np.asarray(log_sigma, dtype=np.float64)
    return {
        "sample_count": float(y.size),
        "sum_x1": float(np.sum(x1)),
        "sum_x2": float(np.sum(x2)),
        "sum_x1x1": float(np.sum(x1 * x1)),
        "sum_x1x2": float(np.sum(x1 * x2)),
        "sum_x2x2": float(np.sum(x2 * x2)),
        "sum_y": float(np.sum(y)),
        "sum_x1y": float(np.sum(x1 * y)),
        "sum_x2y": float(np.sum(x2 * y)),
        "sum_yy": float(np.sum(y * y)),
    }


def build_primitive_payload() -> dict[str, Any]:
    """
    Compare reference formulas against local primitive implementations.

    This phase deliberately avoids large grid artifacts and MCMC.  Its job is
    to catch formula drift early, before later phases mix in interpolation,
    sampling variance, or per-lens data products.
    """

    from cmass_lens_inference.models.sonnenfeld2024_slacs import paper_constants
    from cmass_lens_inference.models.sonnenfeld2024_slacs.posterior import (
        _passes_reference_source_redshift_mask,
        solve_fundamental_plane_ols2,
    )
    from cmass_lens_inference.numba_backend.kernels.selection import p_find
    from cmass_lens_inference.types import FPPriorConfig

    comparisons: dict[str, dict[str, Any]] = {}

    pfind_reference = reference_pfind(theta_est=1.23, theta0=0.9, la_find=1.37)
    pfind_local = float(p_find(1.23, 0.9, 1.37))
    comparisons["pfind"] = _scalar_comparison("pfind", pfind_reference, pfind_local)

    source_cases = [(0.2, 0.26), (0.2, 0.24), (0.2, 2.1), (0.2, 0.05)]
    source_reference = [reference_source_redshift_mask(zd, zs) for zd, zs in source_cases]
    source_local = [
        bool(_passes_reference_source_redshift_mask(zd, zs, 0.05, 2.0, 0.05))
        for zd, zs in source_cases
    ]
    comparisons["source_redshift_mask"] = {
        "status": STATUS_PASSED if source_reference == source_local else STATUS_FAILED,
        "reference": source_reference,
        "local": source_local,
        "cases": [list(case) for case in source_cases],
    }

    size_reference = reference_size_relation(11.4)
    size_local = (
        paper_constants.SIZE_MU0_PHYSICAL
        + paper_constants.SIZE_MU1_PHYSICAL * 11.4
        + paper_constants.SIZE_MU2_PHYSICAL * 11.4**2
    )
    comparisons["size_relation"] = _scalar_comparison("size_relation", size_reference, size_local)

    default_reference = reference_fp_prior_defaults()
    default_config = FPPriorConfig(enabled=True)
    default_local = {
        "fit_mstar_min": default_config.fit_mstar_min,
        "pivot_mstar": default_config.pivot_mstar,
        "fiducial_scatter": default_config.fiducial_scatter,
        "scatter_error": default_config.scatter_error,
        "mu_v_prior": default_config.mu_v_prior,
        "mu_v_error": default_config.mu_v_error,
        "beta_v_prior": default_config.beta_v_prior,
        "beta_v_error": default_config.beta_v_error,
    }
    fp_default_diffs = {
        key: abs(float(default_reference[key]) - float(default_local[key]))
        for key in default_reference
    }
    comparisons["fp_prior_defaults"] = {
        "status": (
            STATUS_PASSED
            if all(_close_enough(default_reference[key], default_local[key]) for key in default_reference)
            else STATUS_FAILED
        ),
        "reference": default_reference,
        "local": default_local,
        "abs_diff": fp_default_diffs,
    }

    mstar = np.asarray([11.05, 11.25, 11.55, 11.85], dtype=np.float64)
    delta_r = np.asarray([-0.08, 0.03, 0.01, 0.11], dtype=np.float64)
    log_sigma = 2.31 + 0.24 * (mstar - 11.3) - 0.18 * delta_r
    reference_fit = reference_fp_ols2_from_rows(
        mstar=mstar,
        delta_r=delta_r,
        log_sigma=log_sigma,
        pivot_mstar=11.3,
    )
    local_fit = solve_fundamental_plane_ols2(
        **_fp_summary_from_rows(
            mstar=mstar,
            delta_r=delta_r,
            log_sigma=log_sigma,
            pivot_mstar=11.3,
        )
    )
    fp_fit_abs_diff = [abs(float(reference) - float(local)) for reference, local in zip(reference_fit, local_fit, strict=True)]
    comparisons["fp_ols2"] = {
        "status": (
            STATUS_PASSED
            if all(_close_enough(reference, local) for reference, local in zip(reference_fit, local_fit, strict=True))
            else STATUS_FAILED
        ),
        "reference": {
            "mu": reference_fit[0],
            "beta": reference_fit[1],
            "xi": reference_fit[2],
            "scatter": reference_fit[3],
        },
        "local": {
            "mu": local_fit[0],
            "beta": local_fit[1],
            "xi": local_fit[2],
            "scatter": local_fit[3],
        },
        "abs_diff": fp_fit_abs_diff,
    }

    overall_status = (
        STATUS_PASSED
        if all(record["status"] == STATUS_PASSED for record in comparisons.values())
        else STATUS_FAILED
    )
    return {
        "schema_version": "sonnenfeld_reference_comparison_primitive_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "phase": "primitive",
        "status": overall_status,
        "status_options": list(STATUS_OPTIONS),
        "comparisons": comparisons,
    }


def build_data_grid_payload(
    *,
    reference_root: Path,
    candidate_root: Path,
    candidate_dataset_path: Path | None,
) -> dict[str, Any]:
    """
    Build the Phase-4 data/grid comparison payload.

    In the absence of the required grid artifacts this stage must stop at a
    data gate.  That behavior is intentional: comparing posterior numbers
    without proving that the reference and local HDF5 products describe the
    same physical grids would create a misleading validation result.
    """

    audit_payload = build_audit_payload(
        reference_root=reference_root,
        candidate_root=candidate_root,
        model="sonnenfeld2024_slacs",
        candidate_dataset_path=candidate_dataset_path,
    )
    missing = _required_artifacts_missing(audit_payload)
    if missing:
        return {
            "schema_version": "sonnenfeld_reference_comparison_data_grid_v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "phase": "data_grid",
            "status": STATUS_SKIPPED_DATA_GATED,
            "status_options": list(STATUS_OPTIONS),
            "missing_required_artifacts": missing,
            "reference": audit_payload["reference"],
            "candidate": audit_payload["candidate"],
            "comparisons": {},
            "reasons": [f"missing required artifact: {name}" for name in missing],
        }

    return {
        "schema_version": "sonnenfeld_reference_comparison_data_grid_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "phase": "data_grid",
        "status": STATUS_NOT_COMPARABLE,
        "status_options": list(STATUS_OPTIONS),
        "missing_required_artifacts": [],
        "reference": audit_payload["reference"],
        "candidate": audit_payload["candidate"],
        "comparisons": {},
        "reasons": [
            "all required artifacts are present, but numeric grid oracle is not implemented in this phase batch"
        ],
    }


def map_reference_theta_to_local(reference_theta: dict[str, float] | list[float] | tuple[float, ...]) -> dict[str, Any]:
    """
    Convert one reference 12D theta into the local Sonnenfeld order.

    The two implementations use the same scientific parameters but not the same
    array order.  This explicit name-based mapper is therefore part of the
    comparison contract; downstream phases should never compare raw position
    zero against raw position zero without passing through this function.
    """

    if isinstance(reference_theta, dict):
        missing = [name for name in REFERENCE_PARAMETER_NAMES if name not in reference_theta]
        if missing:
            raise ValueError(
                "missing reference theta parameters: "
                + ", ".join(missing)
            )
        reference_values = {
            name: float(reference_theta[name])
            for name in REFERENCE_PARAMETER_NAMES
        }
    else:
        if len(reference_theta) != len(REFERENCE_PARAMETER_NAMES):
            raise ValueError(
                f"expected 12 reference theta values, got {len(reference_theta)}"
            )
        reference_values = {
            name: float(value)
            for name, value in zip(REFERENCE_PARAMETER_NAMES, reference_theta, strict=True)
        }

    local_values = {
        local_name: reference_values[reference_name]
        for reference_name, local_name in REFERENCE_TO_LOCAL_PARAMETER.items()
    }
    local_theta = [local_values[name] for name in LOCAL_PARAMETER_NAMES]
    return {
        "reference_order": list(REFERENCE_PARAMETER_NAMES),
        "local_order": list(LOCAL_PARAMETER_NAMES),
        "reference_values": reference_values,
        "local_values": local_values,
        "local_theta": local_theta,
    }


def _sha256_file(path: Path) -> str:
    """Return a stable SHA256 checksum for one local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _describe_hdf5(path: Path) -> dict[str, Any]:
    """
    Return a lightweight HDF5 schema description.

    Invalid placeholder files are still useful audit facts.  They are reported
    as ``unreadable`` instead of raising, so the caller can classify the later
    comparison as data-gated rather than crashing during manifest creation.
    """

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - cmass_lens normally has h5py
        return {"status": "unavailable", "error": str(exc)}

    try:
        with h5py.File(path, "r") as handle:
            datasets: dict[str, dict[str, Any]] = {}
            groups: list[str] = []

            def visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    datasets[name] = {
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                    }
                elif isinstance(obj, h5py.Group):
                    groups.append(name)

            handle.visititems(visit)
            return {
                "status": "readable",
                "datasets": datasets,
                "groups": sorted(groups),
                "attrs": {key: _json_scalar(value) for key, value in handle.attrs.items()},
            }
    except OSError as exc:
        return {"status": "unreadable", "error": str(exc)}


def _json_scalar(value: Any) -> Any:
    """Convert small HDF5/numpy scalar values into JSON-serializable objects."""

    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def describe_artifact(path: Path, *, required_for_full_comparison: bool) -> dict[str, Any]:
    """
    Describe one file without interpreting its scientific contents.

    The manifest intentionally separates physical comparability from file
    existence.  A file can be present and hashed while still being unusable for
    later phases because its HDF5 schema is unreadable or incomplete.
    """

    resolved_path = Path(path)
    artifact: dict[str, Any] = {
        "path": str(resolved_path),
        "required_for_full_comparison": bool(required_for_full_comparison),
    }
    if not resolved_path.exists():
        artifact["status"] = "missing"
        return artifact
    if not resolved_path.is_file():
        artifact["status"] = "not_file"
        return artifact

    stat = resolved_path.stat()
    artifact.update(
        {
            "status": "present",
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": _sha256_file(resolved_path),
        }
    )
    if resolved_path.suffix.lower() in {".h5", ".hdf5"}:
        artifact["hdf5"] = _describe_hdf5(resolved_path)
    return artifact


def _build_reference_manifest(reference_root: Path) -> dict[str, Any]:
    """Build the reference side of the audit manifest."""

    artifacts = {
        relative_path: describe_artifact(
            reference_root / relative_path,
            required_for_full_comparison=required,
        )
        for relative_path, required in REFERENCE_ARTIFACTS
    }
    scripts = {
        relative_path: describe_artifact(
            reference_root / relative_path,
            required_for_full_comparison=False,
        )
        for relative_path in REFERENCE_SCRIPTS
    }
    return {
        "root": str(reference_root),
        "artifacts": artifacts,
        "scripts": scripts,
    }


def _build_candidate_manifest(
    candidate_root: Path,
    candidate_dataset_path: Path | None,
) -> dict[str, Any]:
    """Build the candidate side of the audit manifest."""

    dataset_artifact = (
        describe_artifact(candidate_dataset_path, required_for_full_comparison=True)
        if candidate_dataset_path is not None
        else {
            "path": None,
            "status": "missing",
            "required_for_full_comparison": True,
        }
    )
    return {
        "root": str(candidate_root),
        "canonical_dataset": dataset_artifact,
    }


def _required_artifacts_missing(payload: dict[str, Any]) -> list[str]:
    """Return names of required full-comparison artifacts that are not present."""

    missing: list[str] = []
    for name, artifact in payload["reference"]["artifacts"].items():
        if artifact["required_for_full_comparison"] and artifact["status"] != "present":
            missing.append(f"reference:{name}")
    candidate_dataset = payload["candidate"]["canonical_dataset"]
    if (
        candidate_dataset["required_for_full_comparison"]
        and candidate_dataset["status"] != "present"
    ):
        missing.append("candidate:canonical_dataset")
    return missing


def build_audit_payload(
    *,
    reference_root: Path,
    candidate_root: Path,
    model: str,
    candidate_dataset_path: Path | None,
) -> dict[str, Any]:
    """
    Build the machine-readable Phase-0 audit payload.

    ``sonnenfeld2024_slacs`` is the only comparable model in the first phase.
    Other variants, including h-unit, are deliberately rejected as
    ``not_comparable`` so unit-coordinate shifts do not pollute the first
    paper-native reference comparison.
    """

    payload: dict[str, Any] = {
        "schema_version": "sonnenfeld_reference_comparison_audit_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "phase": "artifact_audit",
        "status_options": list(STATUS_OPTIONS),
        "model": model,
        "reference": _build_reference_manifest(reference_root),
        "candidate": _build_candidate_manifest(candidate_root, candidate_dataset_path),
    }
    if model != "sonnenfeld2024_slacs":
        payload["status"] = STATUS_NOT_COMPARABLE
        payload["reasons"] = [f"model '{model}' is not part of Phase-0 paper-native comparison"]
        return payload

    missing = _required_artifacts_missing(payload)
    if missing:
        payload["status"] = STATUS_SKIPPED_DATA_GATED
        payload["reasons"] = [f"missing required artifact: {name}" for name in missing]
    else:
        payload["status"] = STATUS_PASSED
        payload["reasons"] = []
    return payload


def write_audit_outputs(payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write audit JSON and a short Markdown summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.md"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_lines = [
        "# Sonnenfeld Reference Comparison Audit",
        "",
        f"- status: `{payload['status']}`",
        f"- model: `{payload['model']}`",
        f"- reference root: `{payload['reference']['root']}`",
        f"- candidate root: `{payload['candidate']['root']}`",
        "",
        "## Reasons",
        "",
    ]
    if payload["reasons"]:
        summary_lines.extend(f"- {reason}" for reason in payload["reasons"])
    else:
        summary_lines.append("- none")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return manifest_path, summary_path


def write_comparison_outputs(payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write comparison JSON and a concise Markdown summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "comparison.json"
    summary_path = output_dir / "summary.md"
    comparison_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_lines = [
        "# Sonnenfeld Reference Comparison",
        "",
        f"- phase: `{payload['phase']}`",
        f"- status: `{payload['status']}`",
        "",
        "## Comparisons",
        "",
    ]
    for name, record in payload.get("comparisons", {}).items():
        summary_lines.append(f"- `{name}`: `{record['status']}`")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return comparison_path, summary_path


def main() -> None:
    """Run the requested comparison-harness mode."""

    args = parse_args()
    if args.audit_only:
        payload = build_audit_payload(
            reference_root=Path(args.reference_root).expanduser().resolve(),
            candidate_root=Path(args.candidate_root).expanduser().resolve(),
            model=args.model,
            candidate_dataset_path=(
                Path(args.candidate_dataset).expanduser().resolve()
                if args.candidate_dataset is not None
                else None
            ),
        )
        manifest_path, summary_path = write_audit_outputs(
            payload,
            Path(args.output_dir).expanduser().resolve(),
        )
        print(
            json.dumps(
                {
                    "manifest_path": str(manifest_path),
                    "summary_path": str(summary_path),
                    "status": payload["status"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    requested_stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    if requested_stages not in ({"primitive"}, {"grid"}):
        raise SystemExit("Current implementation supports exactly --stages primitive or --stages grid.")
    if args.model != "sonnenfeld2024_slacs":
        payload = {
            "schema_version": "sonnenfeld_reference_comparison_primitive_v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "phase": ",".join(sorted(requested_stages)),
            "status": STATUS_NOT_COMPARABLE,
            "status_options": list(STATUS_OPTIONS),
            "model": args.model,
            "comparisons": {},
            "reasons": [f"model '{args.model}' is not part of paper-native comparison"],
        }
    elif requested_stages == {"grid"}:
        payload = build_data_grid_payload(
            reference_root=Path(args.reference_root).expanduser().resolve(),
            candidate_root=Path(args.candidate_root).expanduser().resolve(),
            candidate_dataset_path=(
                Path(args.candidate_dataset).expanduser().resolve()
                if args.candidate_dataset is not None
                else None
            ),
        )
        payload["model"] = args.model
    else:
        payload = build_primitive_payload()
        payload["model"] = args.model
    comparison_path, summary_path = write_comparison_outputs(
        payload,
        Path(args.output_dir).expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "comparison_path": str(comparison_path),
                "summary_path": str(summary_path),
                "status": payload["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
