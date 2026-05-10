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
import tempfile
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

DEFAULT_REFERENCE_THETA: dict[str, float] = {
    "mu_m5": 11.33,
    "sigma_m5": 0.07,
    "beta_m5": 0.62,
    "xi_m5": -0.13,
    "mu_gamma": 2.00,
    "sigma_gamma": 0.18,
    "beta_gamma": 0.31,
    "xi_gamma": -0.78,
    "mu_zs": 0.48,
    "sigma_zs": 0.215,
    "t_find": 0.77,
    "la_find": 1.37,
}

REFERENCE_COMPARISON_BOX_PRIOR: dict[str, tuple[float, float]] = {
    "mu5_0": (10.5, 12.2),
    "beta5": (-3.0, 3.0),
    "xi5": (-3.0, 3.0),
    "sigma5": (0.01, 0.3),
    "mu_gamma_0": (1.2, 2.8),
    "beta_gamma": (-3.0, 3.0),
    "xi_gamma": (-3.0, 3.0),
    "sigma_gamma": (0.01, 0.8),
    "mu_zs": (0.0, 2.0),
    "sigma_zs": (0.001, 1.0),
    "theta0": (0.0, 3.0),
    "loga": (-1.0, 3.0),
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
        help=(
            "Comma-separated comparison stages. Supported values: "
            "primitive, grid, normalization, fp, per-lens, posterior."
        ),
    )
    parser.add_argument(
        "--lens-index",
        type=int,
        default=0,
        help="Lens index used by the per-lens diagnostic stage.",
    )
    parser.add_argument(
        "--normalization-samples",
        type=int,
        default=1000,
        help=(
            "Parent-population samples used by local diagnostic decomposition. "
            "This does not change external reference chain files."
        ),
    )
    parser.add_argument(
        "--gamma-points",
        type=int,
        default=32,
        help="Gamma quadrature points used by local diagnostic decomposition.",
    )
    parser.add_argument(
        "--mstar-points",
        type=int,
        default=32,
        help="Stellar-mass quadrature points used by local diagnostic decomposition.",
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


def _read_slacs_table_by_name(catalog_path: Path) -> dict[str, dict[str, float]]:
    """
    Read the external SLACS table into a name-keyed dictionary.

    The reference scripts address lenses by string IDs, while the canonical
    dataset stores a sorted `lens_id` array.  Comparing by explicit name avoids
    silently treating two different lens orderings as a numerical difference.
    """

    rows: dict[str, dict[str, float]] = {}
    for raw_line in catalog_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = stripped.split()
        if len(values) < 13:
            raise ValueError(f"{catalog_path} contains a malformed SLACS row: {stripped!r}")
        rows[values[0]] = {
            "z_d": float(values[3]),
            "z_s": float(values[4]),
            "re_arcsec": float(values[5]),
            "log_re_kpc": math.log10(float(values[6])),
            "theta_e_obs": float(values[7]),
            "log_mstar_obs": float(values[9]),
            "log_mstar_err": float(values[10]),
            "sigma_obs": float(values[11]),
            "sigma_err": float(values[12]),
        }
    if not rows:
        raise ValueError(f"{catalog_path} contains no SLACS data rows.")
    return rows


def _decode_hdf5_string_array(values: np.ndarray) -> tuple[str, ...]:
    """Decode an HDF5 string array without depending on one h5py string mode."""

    decoded: list[str] = []
    for value in np.atleast_1d(values):
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return tuple(decoded)


def _max_abs_rel_diff(reference: np.ndarray, local: np.ndarray) -> tuple[float, float]:
    """Return maximum absolute and relative difference for two numeric arrays."""

    reference_values = np.asarray(reference, dtype=np.float64)
    local_values = np.asarray(local, dtype=np.float64)
    diff = np.abs(reference_values - local_values)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    denom = np.maximum(np.abs(reference_values), 1.0e-300)
    max_rel = float(np.max(diff / denom)) if diff.size else 0.0
    return max_abs, max_rel


def _array_comparison(
    *,
    name: str,
    reference: np.ndarray,
    local: np.ndarray,
    rtol: float,
    atol: float,
    note: str,
) -> dict[str, Any]:
    """
    Build one array-comparison record with explicit tolerance metadata.

    Phase-4 checks mix products from reference scripts and local data-prep
    generators.  Recording shape and tolerance beside every result keeps later
    failures diagnosable without reopening the HDF5 files by hand.
    """

    reference_values = np.asarray(reference, dtype=np.float64)
    local_values = np.asarray(local, dtype=np.float64)
    same_shape = reference_values.shape == local_values.shape
    if same_shape:
        max_abs, max_rel = _max_abs_rel_diff(reference_values, local_values)
        passed = bool(np.allclose(reference_values, local_values, rtol=rtol, atol=atol, equal_nan=False))
    else:
        max_abs = math.inf
        max_rel = math.inf
        passed = False
    return {
        "name": name,
        "status": STATUS_PASSED if passed else STATUS_FAILED,
        "reference_shape": list(reference_values.shape),
        "local_shape": list(local_values.shape),
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "rtol": float(rtol),
        "atol": float(atol),
        "note": note,
    }


def _axis_subset_comparison(
    *,
    name: str,
    reference_axis: np.ndarray,
    local_axis: np.ndarray,
    atol: float,
    note: str,
) -> dict[str, Any]:
    """
    Check whether a lower-density local axis is represented by a reference axis.

    Sonnenfeld's reference per-lens lensing product stores 81 gamma samples,
    while the local canonical fixed-m5 product stores the 17-point dynamics
    axis.  This is still comparable for deterministic mass/Jacobian values as
    long as every local point lies on the reference axis.
    """

    reference_values = np.asarray(reference_axis, dtype=np.float64)
    local_values = np.asarray(local_axis, dtype=np.float64)
    nearest_distances = np.asarray(
        [np.min(np.abs(reference_values - value)) for value in local_values],
        dtype=np.float64,
    )
    max_abs = float(np.max(nearest_distances)) if nearest_distances.size else 0.0
    return {
        "name": name,
        "status": STATUS_PASSED if max_abs <= float(atol) else STATUS_FAILED,
        "reference_shape": list(reference_values.shape),
        "local_shape": list(local_values.shape),
        "max_nearest_abs_diff": max_abs,
        "atol": float(atol),
        "note": note,
    }


def _interp_rows_to_axis(
    *,
    source_axis: np.ndarray,
    source_values: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
    """Interpolate each lens row from one gamma axis onto another gamma axis."""

    return np.vstack(
        [
            np.interp(
                np.asarray(target_axis, dtype=np.float64),
                np.asarray(source_axis, dtype=np.float64),
                np.asarray(row, dtype=np.float64),
            )
            for row in np.asarray(source_values, dtype=np.float64)
        ]
    )


def _compare_slacs_table_to_candidate(
    *,
    reference_root: Path,
    candidate_dataset_path: Path,
) -> dict[str, Any]:
    """Compare raw SLACS table columns against the canonical `/lenses` block."""

    import h5py

    table = _read_slacs_table_by_name(reference_root / "SLACS_table.cat")
    with h5py.File(candidate_dataset_path, "r") as handle:
        lens_ids = _decode_hdf5_string_array(handle["lenses/lens_id"][()])
        comparisons: dict[str, dict[str, Any]] = {}
        scalar_fields = (
            "z_d",
            "z_s",
            "log_mstar_obs",
            "log_mstar_err",
            "log_re_kpc",
            "theta_e_obs",
        )
        candidate_field_map = {
            "z_d": "z_d",
            "z_s": "z_s",
            "log_mstar_obs": "log_mstar_obs",
            "log_mstar_err": "log_mstar_err",
            "log_re_kpc": "log_re_obs",
            "theta_e_obs": "theta_e_obs",
        }
        for field_name in scalar_fields:
            reference_values = np.asarray([table[lens_id][field_name] for lens_id in lens_ids], dtype=np.float64)
            local_values = np.asarray(handle[f"lenses/{candidate_field_map[field_name]}"][()], dtype=np.float64)
            comparisons[field_name] = _array_comparison(
                name=f"slacs_table.{field_name}",
                reference=reference_values,
                local=local_values,
                rtol=1.0e-12,
                atol=1.0e-12,
                note="Raw SLACS_table.cat value compared by lens_id against canonical /lenses block.",
            )

        reference_sigma = np.asarray([table[lens_id]["sigma_obs"] for lens_id in lens_ids], dtype=np.float64)
        reference_sigma_err = np.asarray([table[lens_id]["sigma_err"] for lens_id in lens_ids], dtype=np.float64)
        comparisons["sigma_obs"] = _array_comparison(
            name="slacs_table.sigma_obs",
            reference=reference_sigma,
            local=np.asarray(handle["lenses/sigma_obs"][()])[:, 0],
            rtol=1.0e-12,
            atol=1.0e-12,
            note="First canonical sigma slot should preserve the SLACS velocity-dispersion measurement.",
        )
        comparisons["sigma_err"] = _array_comparison(
            name="slacs_table.sigma_err",
            reference=reference_sigma_err,
            local=np.asarray(handle["lenses/sigma_err"][()])[:, 0],
            rtol=1.0e-12,
            atol=1.0e-12,
            note="First canonical sigma-error slot should preserve the SLACS table uncertainty.",
        )

    status = STATUS_PASSED if all(item["status"] == STATUS_PASSED for item in comparisons.values()) else STATUS_FAILED
    return {
        "status": status,
        "lens_count": len(lens_ids),
        "comparisons": comparisons,
    }


def _compare_reference_grids_to_candidate(
    *,
    reference_root: Path,
    candidate_dataset_path: Path,
) -> dict[str, Any]:
    """
    Compare generated reference HDF5 grids against canonical data blocks.

    This function is intentionally strict for axes and mass grids, because those
    quantities are deterministic.  The per-lens finite-fibre cross-section grid
    is recorded separately because the current local posterior uses the global
    theta_E x gamma cross-section lookup for per-lens likelihood terms, while
    the reference stores a per-lens spline in `slacs_lensing_grids.hdf5`.
    """

    import h5py

    comparisons: dict[str, dict[str, Any]] = {}
    with (
        h5py.File(candidate_dataset_path, "r") as candidate,
        h5py.File(reference_root / "slacs_lensing_grids.hdf5", "r") as lensing,
        h5py.File(reference_root / "slacs_jeans_grids.hdf5", "r") as jeans,
        h5py.File(reference_root / "fibre_crosssect_grid.hdf5", "r") as fibre,
    ):
        lens_ids = _decode_hdf5_string_array(candidate["lenses/lens_id"][()])
        reference_gamma_lensing = np.asarray(lensing["gamma_grid"][()], dtype=np.float64)
        reference_gamma_jeans = np.asarray(jeans["gamma_grid"][()], dtype=np.float64)
        local_gamma = np.asarray(candidate["lensing_mass_grids/gamma_grid"][()], dtype=np.float64)
        local_gamma_axis = local_gamma[0] if local_gamma.ndim == 2 else local_gamma

        comparisons["lensing_gamma_axis"] = _axis_subset_comparison(
            name="lensing_gamma_axis",
            reference_axis=reference_gamma_lensing,
            local_axis=local_gamma_axis,
            atol=1.0e-12,
            note=(
                "Reference per-lens lensing grid has 81 gamma points; canonical fixed-m5 mass grids "
                "may use the 17-point dynamics subset."
            ),
        )
        comparisons["jeans_gamma_axis"] = _array_comparison(
            name="jeans_gamma_axis",
            reference=reference_gamma_jeans,
            local=local_gamma_axis,
            rtol=1.0e-12,
            atol=1.0e-12,
            note="Reference Jeans grid gamma axis versus canonical mass-grid axis.",
        )

        reference_m5_full = np.vstack([np.asarray(lensing[lens_id]["m5_grid"][()], dtype=np.float64) for lens_id in lens_ids])
        reference_dm5_full = np.vstack([np.asarray(lensing[lens_id]["dm5drein_grid"][()], dtype=np.float64) for lens_id in lens_ids])
        reference_m5 = _interp_rows_to_axis(
            source_axis=reference_gamma_lensing,
            source_values=reference_m5_full,
            target_axis=local_gamma_axis,
        )
        reference_dm5 = _interp_rows_to_axis(
            source_axis=reference_gamma_lensing,
            source_values=reference_dm5_full,
            target_axis=local_gamma_axis,
        )
        reference_s2 = np.vstack([np.asarray(jeans[lens_id]["s2_grid"][()], dtype=np.float64) for lens_id in lens_ids])
        comparisons["m5_grid"] = _array_comparison(
            name="m5_grid",
            reference=reference_m5,
            local=np.asarray(candidate["lensing_mass_grids/log_enclosed_mass_grid"][()], dtype=np.float64),
            rtol=5.0e-8,
            atol=5.0e-10,
            note="Reference make_slacs_lensing_grids.py m5_grid versus canonical log_enclosed_mass_grid.",
        )
        comparisons["dm5drein_grid"] = _array_comparison(
            name="dm5drein_grid",
            reference=reference_dm5,
            local=np.asarray(candidate["lensing_mass_grids/dmass_dthetaein_grid"][()], dtype=np.float64),
            rtol=5.0e-5,
            atol=5.0e-7,
            note="Reference finite-difference Jacobian versus canonical dmass_dthetaein_grid.",
        )
        comparisons["s2_grid"] = _array_comparison(
            name="s2_grid",
            reference=reference_s2,
            local=np.asarray(candidate["lensing_mass_grids/s2_grid"][()], dtype=np.float64),
            rtol=5.0e-6,
            atol=1.0e-8,
            note="Reference make_slacs_jeans_grids.py s2_grid versus canonical per-lens s2 grid.",
        )
        comparisons["global_cross_section_theta_axis"] = _array_comparison(
            name="global_cross_section_theta_axis",
            reference=np.asarray(fibre["tein_grid"][()], dtype=np.float64),
            local=np.asarray(candidate["lensing_cross_section/theta_e_axis"][()], dtype=np.float64),
            rtol=1.0e-12,
            atol=1.0e-12,
            note="Reference finite-fibre theta_E axis versus canonical cross-section axis.",
        )
        comparisons["global_cross_section_gamma_axis"] = _array_comparison(
            name="global_cross_section_gamma_axis",
            reference=np.asarray(fibre["gamma_grid"][()], dtype=np.float64),
            local=np.asarray(candidate["lensing_cross_section/gamma_axis"][()], dtype=np.float64),
            rtol=1.0e-12,
            atol=1.0e-12,
            note="Reference finite-fibre gamma axis versus canonical cross-section axis.",
        )
        comparisons["global_mufibre3_cross_section"] = _array_comparison(
            name="global_mufibre3_cross_section",
            reference=np.asarray(fibre["mufibre3_cs_grid"][()], dtype=np.float64),
            local=np.asarray(candidate["lensing_cross_section/cross_section_grid"][()], dtype=np.float64),
            rtol=1.0e-12,
            atol=1.0e-12,
            note="Sonnenfeld finite-fibre areas must be copied directly, not converted as CMASS separable areas.",
        )

        population_group = candidate["velocity_dispersion_grids/population_sigma_unit"]
        population_grid = np.asarray(population_group["s_unit_grid"][()], dtype=np.float64)
        comparisons["population_sigma_unit_schema"] = {
            "name": "population_sigma_unit_schema",
            "status": STATUS_PASSED if population_grid.ndim in (3, 4) and np.all(np.isfinite(population_grid)) else STATUS_FAILED,
            "shape": list(population_grid.shape),
            "gamma_axis_shape": list(population_group["gamma_axis"].shape),
            "zd_axis_shape": list(population_group["zd_axis"].shape),
            "log_re_kpc_axis_shape": list(population_group["log_re_kpc_axis"].shape),
            "n_axis_shape": list(population_group["n_axis"].shape) if "n_axis" in population_group else [],
            "mass_definition_label": _json_scalar(population_group.attrs.get("mass_definition_label", "")),
            "note": "Runtime-required sigma^2 / 10**m5 interpolation table exists and is finite.",
        }

    status = STATUS_PASSED if all(item["status"] == STATUS_PASSED for item in comparisons.values()) else STATUS_FAILED
    return {
        "status": status,
        "lens_count": len(lens_ids),
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

    assert candidate_dataset_path is not None
    table_payload = _compare_slacs_table_to_candidate(
        reference_root=reference_root,
        candidate_dataset_path=candidate_dataset_path,
    )
    grid_payload = _compare_reference_grids_to_candidate(
        reference_root=reference_root,
        candidate_dataset_path=candidate_dataset_path,
    )
    overall_status = (
        STATUS_PASSED
        if table_payload["status"] == STATUS_PASSED and grid_payload["status"] == STATUS_PASSED
        else STATUS_FAILED
    )
    return {
        "schema_version": "sonnenfeld_reference_comparison_data_grid_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "phase": "data_grid",
        "status": overall_status,
        "status_options": list(STATUS_OPTIONS),
        "missing_required_artifacts": [],
        "reference": audit_payload["reference"],
        "candidate": audit_payload["candidate"],
        "comparisons": {
            "slacs_table": table_payload,
            "reference_grids": grid_payload,
        },
        "reasons": [] if overall_status == STATUS_PASSED else ["one or more data/grid oracle comparisons failed"],
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


def _build_reference_comparison_runtime_config(
    *,
    candidate_dataset_path: Path,
    normalization_samples: int,
    gamma_points: int,
    mstar_points: int,
):
    """
    Construct a minimal paper-native Sonnenfeld RuntimeConfig for diagnostics.

    The production YAML currently points at the h-unit variant.  Reference
    comparison Phase 5-8 must instead run `sonnenfeld2024_slacs` with
    `legacy_fixed_kpc/m5`, so this helper builds the typed config directly
    from the registry rather than editing a user-facing config file.
    """

    from cmass_lens_inference.model_registry import get_model_definition
    from cmass_lens_inference.types import (
        CosmologyConfig,
        DataConfig,
        FPPriorConfig,
        HyperParams,
        IntegrationConfig,
        ModelConfig,
        OutputConfig,
        ProfileConfig,
        RuntimeConfig,
        RuntimeOptions,
        SamplingConfig,
    )

    model_definition = get_model_definition("sonnenfeld2024_slacs")
    mass_definition = model_definition.resolve_mass_definition("legacy_fixed_kpc")
    parameter_schema = model_definition.build_parameter_schema(
        mass_definition=mass_definition,
        public_box_prior=REFERENCE_COMPARISON_BOX_PRIOR,
    )
    mapped_theta = map_reference_theta_to_local(DEFAULT_REFERENCE_THETA)
    initial_center = HyperParams.from_public_dict(
        mapped_theta["local_values"],
        parameter_schema,
    )
    return RuntimeConfig(
        unit_convention="legacy_fixed_kpc",
        h_ref=0.7,
        profile=ProfileConfig(name="devauc"),
        model=ModelConfig(name="sonnenfeld2024_slacs"),
        mass_definition=mass_definition,
        parameter_schema=parameter_schema,
        fp_prior=FPPriorConfig(enabled=True),
        data=DataConfig(inference_dataset_path=Path(candidate_dataset_path).expanduser().resolve()),
        sampling=SamplingConfig(
            random_seed=7,
            initial_center=initial_center,
            initial_jitter_scale=1.0e-3,
            n_walkers=24,
            n_steps=10,
            burn_in=0,
        ),
        integration=IntegrationConfig(
            gamma_points=int(gamma_points),
            mstar_points=int(mstar_points),
            normalization_samples=int(normalization_samples),
        ),
        cosmology=CosmologyConfig(h0=70.0, omega_m=0.3),
        runtime=RuntimeOptions(
            checkpoint_every=100,
            parallel_strategy="off",
            progress=False,
            progress_summary_every=25,
            show_stage_timing=False,
            disable_hdf5_file_locking=False,
            num_threads=1,
            reserve_cores=0,
        ),
        output=OutputConfig(
            root_dir=Path(tempfile.gettempdir()).resolve(),
            run_label="sonnenfeld_reference_comparison",
            overwrite_latest=True,
        ),
    )


def _build_local_compiled_model(
    *,
    candidate_dataset_path: Path,
    normalization_samples: int,
    gamma_points: int,
    mstar_points: int,
):
    """Build the local compiled Sonnenfeld model used by Phase 5-8 diagnostics."""

    from cmass_lens_inference.numba_backend.likelihood_engine import build_compiled_model

    runtime_config = _build_reference_comparison_runtime_config(
        candidate_dataset_path=candidate_dataset_path,
        normalization_samples=normalization_samples,
        gamma_points=gamma_points,
        mstar_points=mstar_points,
    )
    return build_compiled_model(runtime_config)


def _reference_theta_from_full_chain(reference_root: Path, *, selector: str) -> dict[str, Any]:
    """
    Select one actual reference-chain sample and return it in named order.

    The reference chain can supply theta values and stored `logp`, but it does
    not store the full decomposition required by Phase 8.  This helper keeps the
    chain readout honest by returning only what the HDF5 file actually contains.
    """

    import h5py

    chain_path = reference_root / "full_inference.hdf5"
    with h5py.File(chain_path, "r") as handle:
        logp = np.asarray(handle["logp"][()], dtype=np.float64)
        finite = np.isfinite(logp)
        if not np.any(finite):
            raise ValueError(f"{chain_path} contains no finite logp values.")
        if selector == "max_logp":
            flat_index = int(np.nanargmax(logp))
        elif selector == "low_logp":
            finite_flat = np.flatnonzero(finite.ravel())
            ordered = finite_flat[np.argsort(logp.ravel()[finite_flat])]
            flat_index = int(ordered[max(0, int(0.05 * (ordered.size - 1)))])
        elif selector == "theta0_high":
            theta0_values = np.asarray(handle["t_find"][()], dtype=np.float64)
            usable = np.flatnonzero(finite.ravel())
            flat_index = int(usable[np.argmax(theta0_values.ravel()[usable])])
        else:
            raise ValueError(f"Unsupported theta selector: {selector}")

        walker_index, step_index = np.unravel_index(flat_index, logp.shape)
        reference_values = {
            name: float(handle[name][walker_index, step_index])
            for name in REFERENCE_PARAMETER_NAMES
        }
        return {
            "selector": selector,
            "walker_index": int(walker_index),
            "step_index": int(step_index),
            "reference_logp": float(logp[walker_index, step_index]),
            "reference_values": reference_values,
            "mapped": map_reference_theta_to_local(reference_values),
        }


def _default_theta_record(reference_root: Path) -> dict[str, Any]:
    """Return a stable median-near theta source for Phase 5-7 diagnostics."""

    try:
        return _reference_theta_from_full_chain(reference_root, selector="max_logp")
    except Exception:
        return {
            "selector": "default_reference_guess",
            "walker_index": None,
            "step_index": None,
            "reference_logp": None,
            "reference_values": dict(DEFAULT_REFERENCE_THETA),
            "mapped": map_reference_theta_to_local(DEFAULT_REFERENCE_THETA),
        }


def _local_population_diagnostics(theta: np.ndarray, compiled_model) -> dict[str, Any]:
    """
    Evaluate local normalization and FP summary using the production kernels.

    The returned payload is deliberately explicit about reference
    non-comparability: external `fit_full.py` also needs `rein_grid.hdf5`,
    `sigma2_grid.hdf5`, and `mz_inference.hdf5`, none of which are present in
    the supplied reference tree.  The local numbers are still valuable because
    they prove the posterior decomposition path is wired and finite.
    """

    from cmass_lens_inference.models.sonnenfeld2024_slacs.posterior import (
        _evaluate_fundamental_plane_prior,
        _fit_fundamental_plane_from_summary,
        normalization_mc_numba,
        population_summary_mc_numba,
    )

    context = compiled_model.context
    normalization_only = normalization_mc_numba(
        theta=theta,
        base_normals=context.base_normals,
        parent_sample_zd=context.parent_sample_zd,
        parent_sample_mstar=context.parent_sample_mstar,
        parent_sample_log_re=context.parent_sample_log_re,
        parent_sample_delta_r=context.parent_sample_delta_r,
        z_grid=context.z_grid,
        chi_kpc_grid=context.chi_kpc_grid,
        cs_theta_e_axis=context.cs_theta_e_axis,
        cs_gamma_axis=context.cs_gamma_axis,
        cs_cross_section_grid=context.cs_cross_section_grid,
        population_gamma_axis=context.population_gamma_axis,
        population_zd_axis=context.population_zd_axis,
        population_log_re_kpc_axis=context.population_log_re_kpc_axis,
        population_n_axis=context.population_n_axis,
        population_sigma_unit_grid=context.population_sigma_unit_grid,
        mass_radius_kpc=context.mass_radius_kpc,
        mass_log_physical_offset=context.mass_log_physical_offset,
        mstar_pivot=context.mstar_pivot,
        n_fixed=context.n_fixed,
        use_sersic_index=context.use_sersic_index,
        gamma_trunc_low=context.gamma_trunc_low,
        gamma_trunc_high=context.gamma_trunc_high,
        source_z_min=context.source_z_min,
        source_z_max=context.source_z_max,
        source_lens_redshift_gap=context.source_lens_redshift_gap,
        sigma_proxy_fractional_scatter=context.sigma_proxy_fractional_scatter,
    )
    normalization_with_fp, fp_summary = population_summary_mc_numba(
        theta=theta,
        base_normals=context.base_normals,
        parent_sample_zd=context.parent_sample_zd,
        parent_sample_mstar=context.parent_sample_mstar,
        parent_sample_log_re=context.parent_sample_log_re,
        parent_sample_delta_r=context.parent_sample_delta_r,
        z_grid=context.z_grid,
        chi_kpc_grid=context.chi_kpc_grid,
        cs_theta_e_axis=context.cs_theta_e_axis,
        cs_gamma_axis=context.cs_gamma_axis,
        cs_cross_section_grid=context.cs_cross_section_grid,
        population_gamma_axis=context.population_gamma_axis,
        population_zd_axis=context.population_zd_axis,
        population_log_re_kpc_axis=context.population_log_re_kpc_axis,
        population_n_axis=context.population_n_axis,
        population_sigma_unit_grid=context.population_sigma_unit_grid,
        mass_radius_kpc=context.mass_radius_kpc,
        mass_log_physical_offset=context.mass_log_physical_offset,
        mstar_pivot=context.mstar_pivot,
        n_fixed=context.n_fixed,
        use_sersic_index=context.use_sersic_index,
        gamma_trunc_low=context.gamma_trunc_low,
        gamma_trunc_high=context.gamma_trunc_high,
        source_z_min=context.source_z_min,
        source_z_max=context.source_z_max,
        source_lens_redshift_gap=context.source_lens_redshift_gap,
        sigma_proxy_fractional_scatter=context.sigma_proxy_fractional_scatter,
        fp_fit_mstar_min=context.fp_fit_mstar_min,
        fp_pivot_mstar=context.fp_pivot_mstar,
    )
    fpfit_mu, fpfit_beta, fpfit_xi, fpfit_scatter = _fit_fundamental_plane_from_summary(fp_summary)
    fp_prior_log_term, prior_mu, prior_beta, prior_xi, prior_scatter = _evaluate_fundamental_plane_prior(
        fp_summary,
        compiled_model,
    )
    return {
        "normalization_only": float(normalization_only),
        "normalization_with_fp_pass": float(normalization_with_fp),
        "normalization_internal_abs_diff": float(abs(normalization_only - normalization_with_fp)),
        "parent_sample_count": int(context.base_normals.shape[0]),
        "fp_summary_count": float(fp_summary[0]),
        "fpfit": {
            "mu": float(fpfit_mu),
            "beta": float(fpfit_beta),
            "xi": float(fpfit_xi),
            "scatter": float(fpfit_scatter),
        },
        "fp_prior": {
            "log_term": float(fp_prior_log_term),
            "mu": float(prior_mu),
            "beta": float(prior_beta),
            "xi": float(prior_xi),
            "scatter": float(prior_scatter),
        },
    }


def _local_log_likelihood_sum(theta: np.ndarray, compiled_model) -> float:
    """Evaluate the local sum of per-lens log-likelihood terms."""

    from cmass_lens_inference.models.sonnenfeld2024_slacs.posterior import log_likelihood_lenses_numba

    context = compiled_model.context
    return float(
        log_likelihood_lenses_numba(
            theta=theta,
            z_grid=context.z_grid,
            chi_kpc_grid=context.chi_kpc_grid,
            cs_theta_e_axis=context.cs_theta_e_axis,
            cs_gamma_axis=context.cs_gamma_axis,
            cs_cross_section_grid=context.cs_cross_section_grid,
            gamma_grid_int=context.gamma_grid_int,
            mass_grid_int=context.mass_grid_int,
            dmass_dthetaein_grid_int=context.dmass_dthetaein_grid_int,
            s2_grid_int=context.s2_grid_int,
            has_s2=context.has_s2,
            num_sigma=context.num_sigma,
            sigma_obs=context.sigma_obs,
            sigma_err=context.sigma_err,
            zd=context.zd,
            zs=context.zs,
            parent_mstar_density_grid=context.parent_mstar_density_grid,
            size_density_grid=context.size_density_grid,
            delta_r_grid=context.delta_r_grid,
            mstar_shift_grid=context.mstar_shift_grid,
            mstar_grid=context.mstar_grid,
            mass_radius_kpc=context.mass_radius_kpc,
            mass_log_physical_offset=context.mass_log_physical_offset,
        )
    )


def build_population_normalization_payload(
    *,
    reference_root: Path,
    candidate_dataset_path: Path,
    normalization_samples: int,
    gamma_points: int,
    mstar_points: int,
) -> dict[str, Any]:
    """Build Phase-5 normalization diagnostics with explicit reference gating."""

    theta_record = _default_theta_record(reference_root)
    compiled_model = _build_local_compiled_model(
        candidate_dataset_path=candidate_dataset_path,
        normalization_samples=normalization_samples,
        gamma_points=gamma_points,
        mstar_points=mstar_points,
    )
    theta = np.asarray(theta_record["mapped"]["local_theta"], dtype=np.float64)
    local = _local_population_diagnostics(theta, compiled_model)
    internal_passed = local["normalization_internal_abs_diff"] <= 1.0e-12
    return {
        "schema_version": "sonnenfeld_reference_comparison_normalization_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "phase": "normalization",
        "status": STATUS_NOT_COMPARABLE,
        "status_options": list(STATUS_OPTIONS),
        "theta": theta_record,
        "local": local,
        "comparisons": {
            "local_normalization_internal_consistency": {
                "status": STATUS_PASSED if internal_passed else STATUS_FAILED,
                "abs_diff": local["normalization_internal_abs_diff"],
                "note": "normalization_mc_numba and population_summary_mc_numba should agree on z_norm.",
            }
        },
        "reasons": [
            "External exact normalization is not comparable from the supplied reference tree because "
            "fit_full.py also requires rein_grid.hdf5, sigma2_grid.hdf5, and mz_inference.hdf5."
        ],
    }


def build_fp_prior_payload(
    *,
    reference_root: Path,
    candidate_dataset_path: Path,
    normalization_samples: int,
    gamma_points: int,
    mstar_points: int,
) -> dict[str, Any]:
    """Build Phase-6 FP diagnostics and prior-term payload."""

    theta_record = _default_theta_record(reference_root)
    compiled_model = _build_local_compiled_model(
        candidate_dataset_path=candidate_dataset_path,
        normalization_samples=normalization_samples,
        gamma_points=gamma_points,
        mstar_points=mstar_points,
    )
    theta = np.asarray(theta_record["mapped"]["local_theta"], dtype=np.float64)
    local = _local_population_diagnostics(theta, compiled_model)
    diagnostics_finite = all(
        math.isfinite(float(value))
        for value in (
            local["fpfit"]["mu"],
            local["fpfit"]["beta"],
            local["fpfit"]["xi"],
            local["fpfit"]["scatter"],
            local["fp_prior"]["log_term"],
        )
    )
    return {
        "schema_version": "sonnenfeld_reference_comparison_fp_prior_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "phase": "fp",
        "status": STATUS_NOT_COMPARABLE,
        "status_options": list(STATUS_OPTIONS),
        "theta": theta_record,
        "local": {
            "fpfit_mu": local["fpfit"]["mu"],
            "fpfit_beta": local["fpfit"]["beta"],
            "fpfit_xi": local["fpfit"]["xi"],
            "fpfit_scatter": local["fpfit"]["scatter"],
            "fp_prior_log_term": local["fp_prior"]["log_term"],
            "fp_summary_count": local["fp_summary_count"],
        },
        "comparisons": {
            "local_fp_diagnostics_are_finite": {
                "status": STATUS_PASSED if diagnostics_finite else STATUS_FAILED,
                "note": "The local FP-prior implementation returns all four requested diagnostics plus prior term.",
            }
        },
        "reasons": [
            "Reference FP diagnostics depend on the exact parent-population random stream and sigma2_grid.hdf5; "
            "the stored full_inference.hdf5 has fpfit blobs but not enough intermediate state for strict replay."
        ],
    }


def build_per_lens_payload(
    *,
    reference_root: Path,
    candidate_dataset_path: Path,
    normalization_samples: int,
    gamma_points: int,
    mstar_points: int,
    lens_index: int,
) -> dict[str, Any]:
    """Build Phase-7 per-lens diagnostics for one selected lens."""

    import h5py

    compiled_model = _build_local_compiled_model(
        candidate_dataset_path=candidate_dataset_path,
        normalization_samples=normalization_samples,
        gamma_points=gamma_points,
        mstar_points=mstar_points,
    )
    context = compiled_model.context
    if lens_index < 0 or lens_index >= context.zd.shape[0]:
        raise ValueError(f"lens_index {lens_index} is outside [0, {context.zd.shape[0] - 1}].")
    with h5py.File(candidate_dataset_path, "r") as candidate:
        lens_id = _decode_hdf5_string_array(candidate["lenses/lens_id"][()])[lens_index]
    with h5py.File(reference_root / "slacs_lensing_grids.hdf5", "r") as lensing:
        reference_cs = np.asarray(lensing[lens_id]["mufibre3_cs_grid"][()], dtype=np.float64)
        reference_gamma = np.asarray(lensing["gamma_grid"][()], dtype=np.float64)
    global_cs_at_observed_theta = np.interp(
        context.theta_e_obs[lens_index],
        context.cs_theta_e_axis,
        context.cs_cross_section_grid[:, 0],
    )
    cs_difference_record = {
        "status": STATUS_NOT_COMPARABLE,
        "lens_id": lens_id,
        "reference_per_lens_cs_shape": list(reference_cs.shape),
        "reference_gamma_shape": list(reference_gamma.shape),
        "local_uses_global_theta_gamma_grid": True,
        "observed_theta_e": float(context.theta_e_obs[lens_index]),
        "global_cs_at_observed_theta_gamma_min": float(global_cs_at_observed_theta),
        "note": (
            "Reference per-lens likelihood reads mufibre3_cs_grid(gamma) from slacs_lensing_grids.hdf5; "
            "the current local likelihood uses the global theta_E x gamma cross-section grid."
        ),
    }
    return {
        "schema_version": "sonnenfeld_reference_comparison_per_lens_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "phase": "per-lens",
        "status": STATUS_NOT_COMPARABLE,
        "status_options": list(STATUS_OPTIONS),
        "lens_index": int(lens_index),
        "lens_id": lens_id,
        "local": {
            "z_d": float(context.zd[lens_index]),
            "z_s": float(context.zs[lens_index]),
            "theta_e_obs": float(context.theta_e_obs[lens_index]),
            "num_sigma": int(context.num_sigma[lens_index]),
            "sigma_obs": np.asarray(context.sigma_obs[lens_index], dtype=float).tolist(),
            "sigma_err": np.asarray(context.sigma_err[lens_index], dtype=float).tolist(),
        },
        "comparisons": {
            "per_lens_cross_section_contract": cs_difference_record,
        },
        "reasons": [
            "A strict per-lens likelihood replay needs the reference stochastic importance samples "
            "ms_impsamp/gamma_impsamp and the per-lens cs_lens_splines from fit_full.py."
        ],
    }


def build_full_posterior_payload(
    *,
    reference_root: Path,
    candidate_dataset_path: Path,
    normalization_samples: int,
    gamma_points: int,
    mstar_points: int,
) -> dict[str, Any]:
    """Build Phase-8 local posterior decomposition for three reference-chain theta points."""

    from cmass_lens_inference.numba_backend.likelihood_engine import log_prob

    compiled_model = _build_local_compiled_model(
        candidate_dataset_path=candidate_dataset_path,
        normalization_samples=normalization_samples,
        gamma_points=gamma_points,
        mstar_points=mstar_points,
    )
    theta_records = [
        _reference_theta_from_full_chain(reference_root, selector=selector)
        for selector in ("max_logp", "low_logp", "theta0_high")
    ]
    rows: list[dict[str, Any]] = []
    for theta_record in theta_records:
        theta = np.asarray(theta_record["mapped"]["local_theta"], dtype=np.float64)
        local_population = _local_population_diagnostics(theta, compiled_model)
        likelihood_value = _local_log_likelihood_sum(theta, compiled_model)
        normalization_value = local_population["normalization_with_fp_pass"]
        normalization_term = -compiled_model.context.zd.shape[0] * math.log(normalization_value)
        local_total_from_parts = (
            likelihood_value
            + normalization_term
            + local_population["fp_prior"]["log_term"]
        )
        direct_total, blob = log_prob(theta, compiled_model)
        rows.append(
            {
                "theta_id": theta_record["selector"],
                "theta": theta_record,
                "local": {
                    "sum_lens_log_likelihood": float(likelihood_value),
                    "selection_normalization": float(normalization_value),
                    "normalization_term": float(normalization_term),
                    "fp_prior_log_term": float(local_population["fp_prior"]["log_term"]),
                    "total_log_prob_from_parts": float(local_total_from_parts),
                    "direct_total_log_prob": float(direct_total),
                    "direct_blob_normalization": float(blob["normalization_value"]),
                    "direct_blob_fp_prior_log_term": float(blob["fp_prior_log_term"]),
                },
                "reference": {
                    "stored_chain_logp": theta_record["reference_logp"],
                    "decomposition_available": False,
                },
                "diff": {
                    "local_parts_vs_direct_abs": float(abs(local_total_from_parts - direct_total)),
                    "chain_total_abs": (
                        float(abs(local_total_from_parts - theta_record["reference_logp"]))
                        if theta_record["reference_logp"] is not None
                        else None
                    ),
                },
            }
        )
    internal_passed = all(row["diff"]["local_parts_vs_direct_abs"] <= 1.0e-8 for row in rows)
    return {
        "schema_version": "sonnenfeld_reference_comparison_full_posterior_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "phase": "posterior",
        "status": STATUS_NOT_COMPARABLE,
        "status_options": list(STATUS_OPTIONS),
        "comparisons": {
            "local_decomposition_internal_consistency": {
                "status": STATUS_PASSED if internal_passed else STATUS_FAILED,
                "max_abs_diff": max(row["diff"]["local_parts_vs_direct_abs"] for row in rows),
                "note": "Local decomposition should reconstruct the direct production log_prob value.",
            }
        },
        "theta_results": rows,
        "reasons": [
            "full_inference.hdf5 stores chain logp and FP blobs, but not selection normalization "
            "or per-lens terms; external full posterior can therefore be compared only as a "
            "non-exact total-logp diagnostic until reference replay artifacts are available."
        ],
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
    supported_stages = {"primitive", "grid", "normalization", "fp", "per-lens", "posterior"}
    unsupported_stages = sorted(requested_stages.difference(supported_stages))
    if not requested_stages or unsupported_stages:
        raise SystemExit(
            "Supported --stages values are primitive, grid, normalization, fp, per-lens, posterior. "
            f"Unsupported values: {unsupported_stages}"
        )
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
    else:
        reference_root = Path(args.reference_root).expanduser().resolve()
        candidate_root = Path(args.candidate_root).expanduser().resolve()
        candidate_dataset_path = (
            Path(args.candidate_dataset).expanduser().resolve()
            if args.candidate_dataset is not None
            else None
        )
        stage_payloads: dict[str, dict[str, Any]] = {}
        if "primitive" in requested_stages:
            stage_payloads["primitive"] = build_primitive_payload()
        if "grid" in requested_stages:
            stage_payloads["grid"] = build_data_grid_payload(
                reference_root=reference_root,
                candidate_root=candidate_root,
                candidate_dataset_path=candidate_dataset_path,
            )
        data_dependent_stages = {"normalization", "fp", "per-lens", "posterior"}
        if requested_stages.intersection(data_dependent_stages):
            audit_payload = build_audit_payload(
                reference_root=reference_root,
                candidate_root=candidate_root,
                model=args.model,
                candidate_dataset_path=candidate_dataset_path,
            )
            missing = _required_artifacts_missing(audit_payload)
            if missing:
                gated_payload = {
                    "schema_version": "sonnenfeld_reference_comparison_data_dependent_v1",
                    "created_at_utc": datetime.now(UTC).isoformat(),
                    "phase": ",".join(sorted(requested_stages.intersection(data_dependent_stages))),
                    "status": STATUS_SKIPPED_DATA_GATED,
                    "status_options": list(STATUS_OPTIONS),
                    "missing_required_artifacts": missing,
                    "comparisons": {},
                    "reasons": [f"missing required artifact: {name}" for name in missing],
                }
                for stage in sorted(requested_stages.intersection(data_dependent_stages)):
                    stage_payloads[stage] = gated_payload | {"phase": stage}
            else:
                assert candidate_dataset_path is not None
                if "normalization" in requested_stages:
                    stage_payloads["normalization"] = build_population_normalization_payload(
                        reference_root=reference_root,
                        candidate_dataset_path=candidate_dataset_path,
                        normalization_samples=args.normalization_samples,
                        gamma_points=args.gamma_points,
                        mstar_points=args.mstar_points,
                    )
                if "fp" in requested_stages:
                    stage_payloads["fp"] = build_fp_prior_payload(
                        reference_root=reference_root,
                        candidate_dataset_path=candidate_dataset_path,
                        normalization_samples=args.normalization_samples,
                        gamma_points=args.gamma_points,
                        mstar_points=args.mstar_points,
                    )
                if "per-lens" in requested_stages:
                    stage_payloads["per-lens"] = build_per_lens_payload(
                        reference_root=reference_root,
                        candidate_dataset_path=candidate_dataset_path,
                        normalization_samples=args.normalization_samples,
                        gamma_points=args.gamma_points,
                        mstar_points=args.mstar_points,
                        lens_index=args.lens_index,
                    )
                if "posterior" in requested_stages:
                    stage_payloads["posterior"] = build_full_posterior_payload(
                        reference_root=reference_root,
                        candidate_dataset_path=candidate_dataset_path,
                        normalization_samples=args.normalization_samples,
                        gamma_points=args.gamma_points,
                        mstar_points=args.mstar_points,
                    )

        if len(stage_payloads) == 1:
            payload = next(iter(stage_payloads.values()))
            payload["model"] = args.model
        else:
            status_priority = {
                STATUS_FAILED: 3,
                STATUS_SKIPPED_DATA_GATED: 2,
                STATUS_NOT_COMPARABLE: 1,
                STATUS_PASSED: 0,
            }
            overall_status = max(
                (stage_payload["status"] for stage_payload in stage_payloads.values()),
                key=lambda status: status_priority[status],
            )
            payload = {
                "schema_version": "sonnenfeld_reference_comparison_multi_stage_v1",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "phase": ",".join(sorted(requested_stages)),
                "status": overall_status,
                "status_options": list(STATUS_OPTIONS),
                "model": args.model,
                "stages": stage_payloads,
                "comparisons": {
                    stage: {"status": stage_payload["status"]}
                    for stage, stage_payload in stage_payloads.items()
                },
                "reasons": [
                    reason
                    for stage_payload in stage_payloads.values()
                    for reason in stage_payload.get("reasons", [])
                ],
            }
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
