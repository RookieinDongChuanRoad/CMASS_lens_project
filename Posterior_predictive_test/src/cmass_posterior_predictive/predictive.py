"""
Posterior predictive test workflow for the CMASS strong-lens inference model.

This module turns completed MCMC runs into posterior predictive summaries.
The implementation deliberately mirrors the model's selection-normalization
logic at a higher level:

- draw candidate lenses from the same latent population model
- weight them by the same strong-lens selection factor
- sample explicit replicated lens sets for downstream goodness-of-fit checks

The PPC workflow differs from the normalization kernel in one critical way:
normalization only estimates the scalar expectation of the selection weight,
whereas this module converts that weighted population into concrete replicated
samples that can be compared against the observed 23-lens and 7-lens
statistics.
"""

from __future__ import annotations

import os
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import ndtr, ndtri

from cmass_lens_inference.compiled_context import build_compiled_context
from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.mass_definition import MassDefinition, get_mass_definition, mass_definition_metadata
from cmass_lens_inference.parameter_schema import (
    GAMMA_MODE_DEPENDENT_CODE,
    GAMMA_MODE_INDEPENDENT_CODE,
)
from cmass_lens_inference.parallel import apply_thread_limits
from cmass_lens_inference.types import (
    ObservationRecord,
    ProfileSpec,
    ResolvedParallelism,
)
from .types import (
    PosteriorPredictiveMonitorResult,
    PosteriorPredictiveResult,
    PosteriorTrendResult,
)


THETA_SAMPLE_SIZE = 23
SIGMA_SAMPLE_SIZE = 7
DEFAULT_RANDOM_SEED = 20260309
DEFAULT_MAX_CANDIDATE_POOL_SIZE = 100000
DEFAULT_CANONICAL_POSTERIOR_DRAW_CAP = 192000
DEFAULT_N_REPLICATES: int | None = None
SIGMA_RELATIVE_NOISE = 0.0625
SUMMARY_STAT_NAMES = ("median", "std", "p10", "p90")
DEFAULT_PPC_HISTOGRAM_BIN_COUNT = 24
STD_PANEL_LEFT_PADDING_FRACTION = 0.025
SIGMA_STD_UPPER_PERCENTILE = 99.5
SIGMA_STD_UPPER_PADDING_FACTOR = 1.03
DEFAULT_EXTERNAL_SIGMA_DIR = Path("/Users/liurongfu/Work/CMASS_lens_project/data/external")
DEFAULT_MONITOR_NOT_BEFORE = datetime(2026, 3, 9, 15, 27, 7, tzinfo=timezone(timedelta(hours=8)))
_MAX_ALLOWED_NEGATIVE_FRACTION = 0.05
_MAX_ALLOWED_NEGATIVE_ABSOLUTE_VALUE = 1.0e-4
DEFAULT_TREND_POSTERIOR_DRAWS: int | None = None
DEFAULT_TREND_PARENT_SAMPLE_SIZE = 100000
DEFAULT_TREND_MASS_BIN_COUNT = 19
DEFAULT_TREND_MASS_BIN_MIN = 10.15
DEFAULT_TREND_MASS_BIN_MAX = 12.05
TREND_CATEGORY_NAMES = ("parent", "detectable", "selected")


def _trend_quantity_names(mass_definition: MassDefinition) -> tuple[str, str, str]:
    """Return the public quantity names for one trend run."""

    return (mass_definition.label, "gamma", "sigma_ap")


def _sigma_table_metadata_defaults() -> MassDefinition:
    """Return the legacy sigma-table definition used when metadata is absent."""

    return get_mass_definition(5)


@dataclass(frozen=True)
class SigmaUnitTable:
    """
    Interpolation table for the unit-mass Jeans response.

    The table stores `S_unit = sigma^2 / 10**m_R` for one explicit mass
    definition. Keeping that metadata on the table object lets PPC validate
    that a run using `m5` never silently consumes an `m10` table or vice versa.
    """

    profile_name: str
    mass_definition_label: str
    mass_radius_kpc: float
    units: str
    gamma_axis: np.ndarray
    zd_axis: np.ndarray
    log_re_kpc_axis: np.ndarray
    values: np.ndarray
    n_axis: np.ndarray | None = None

    def __post_init__(self) -> None:
        axes = (self.gamma_axis, self.zd_axis, self.log_re_kpc_axis)
        if self.n_axis is None:
            interpolator = RegularGridInterpolator(
                axes,
                self.values,
                bounds_error=False,
                fill_value=None,
            )
        else:
            interpolator = RegularGridInterpolator(
                axes + (self.n_axis,),
                self.values,
                bounds_error=False,
                fill_value=None,
            )
        object.__setattr__(self, "_interpolator", interpolator)

    @classmethod
    def from_path(cls, table_path: str | Path) -> "SigmaUnitTable":
        """
        Load a sigma-unit interpolation table from `.npz` or HDF5.

        The consumer intentionally requires a small, explicit schema because
        the PPC code should fail fast if the upstream interpolation producer
        changes axis names or array ranks unexpectedly.
        """

        path = Path(table_path).expanduser().resolve()
        if path.suffix.lower() == ".npz":
            return cls._from_npz(path)
        if path.suffix.lower() in {".h5", ".hdf5"}:
            return cls._from_hdf5(path)
        raise ValueError(f"Unsupported sigma table format for '{path}'. Expected .npz, .h5, or .hdf5.")

    @classmethod
    def _from_npz(cls, path: Path) -> "SigmaUnitTable":
        """Load the explicit PPC-native `.npz` schema."""

        with np.load(path) as payload:
            profile_name = payload["profile_name"].item()
            default_mass_definition = _sigma_table_metadata_defaults()
            raw_mass_label = payload["mass_definition_label"].item() if "mass_definition_label" in payload.files else default_mass_definition.label
            raw_mass_radius = (
                payload["mass_radius_kpc"].item()
                if "mass_radius_kpc" in payload.files
                else float(default_mass_definition.radius_kpc)
            )
            raw_units = (
                payload["units"].item()
                if "units" in payload.files
                else default_mass_definition.sigma_unit_units
            )
            n_axis = payload["n_axis"] if "n_axis" in payload.files else None
            return cls(
                profile_name=str(profile_name),
                mass_definition_label=str(raw_mass_label),
                mass_radius_kpc=float(raw_mass_radius),
                units=str(raw_units),
                gamma_axis=np.asarray(payload["gamma_axis"], dtype=float),
                zd_axis=np.asarray(payload["zd_axis"], dtype=float),
                log_re_kpc_axis=np.asarray(payload["log_re_kpc_axis"], dtype=float),
                values=_validate_sigma_unit_grid(np.asarray(payload["s_unit_grid"], dtype=float), source_path=path),
                n_axis=None if n_axis is None else np.asarray(n_axis, dtype=float),
            )

    @classmethod
    def _from_hdf5(cls, path: Path) -> "SigmaUnitTable":
        """Load the explicit HDF5 sigma-table schema shared with the producer."""

        with h5py.File(path, "r") as handle:
            dataset_names = set(handle.keys())
            required_dataset_names = {"profile_name", "gamma_axis", "zd_axis", "log_re_kpc_axis", "s_unit_grid"}
            missing = sorted(required_dataset_names.difference(dataset_names))
            if missing:
                raise ValueError(
                    f"HDF5 sigma table '{path}' does not match the required sigma-unit schema. "
                    f"Missing datasets: {missing}."
                )

            raw_profile_name = handle["profile_name"][()]
            profile_name = _decode_hdf5_string(raw_profile_name)
            default_mass_definition = _sigma_table_metadata_defaults()
            raw_mass_label = handle.attrs.get("mass_definition_label", default_mass_definition.label)
            raw_mass_radius = handle.attrs.get("mass_radius_kpc", float(default_mass_definition.radius_kpc))
            raw_units = handle.attrs.get("units", default_mass_definition.sigma_unit_units)
            n_axis = np.asarray(handle["n_axis"], dtype=float) if "n_axis" in handle else None
            return cls(
                profile_name=profile_name,
                mass_definition_label=_decode_hdf5_string(raw_mass_label),
                mass_radius_kpc=float(raw_mass_radius),
                units=_decode_hdf5_string(raw_units),
                gamma_axis=np.asarray(handle["gamma_axis"], dtype=float),
                zd_axis=np.asarray(handle["zd_axis"], dtype=float),
                log_re_kpc_axis=np.asarray(handle["log_re_kpc_axis"], dtype=float),
                values=_validate_sigma_unit_grid(np.asarray(handle["s_unit_grid"], dtype=float), source_path=path),
                n_axis=n_axis,
            )

    def evaluate(
        self,
        gamma: np.ndarray,
        zd: np.ndarray,
        log_re_kpc: np.ndarray,
        n_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Interpolate `S_unit` at the requested lens coordinates with clipping.

        Clipping is intentional and mirrors the rest of the codebase's
        interpolation policy: the PPC pipeline should stay numerically stable
        even when replicated draws land just outside the tabulated range.
        """

        gamma_clipped = np.clip(gamma, self.gamma_axis[0], self.gamma_axis[-1])
        zd_clipped = np.clip(zd, self.zd_axis[0], self.zd_axis[-1])
        log_re_clipped = np.clip(log_re_kpc, self.log_re_kpc_axis[0], self.log_re_kpc_axis[-1])

        if self.n_axis is None:
            query_points = np.column_stack((gamma_clipped, zd_clipped, log_re_clipped))
        else:
            if n_values is None:
                raise ValueError("Sersic sigma interpolation requires `n_values`.")
            n_clipped = np.clip(n_values, self.n_axis[0], self.n_axis[-1])
            query_points = np.column_stack((gamma_clipped, zd_clipped, log_re_clipped, n_clipped))

        return np.asarray(self._interpolator(query_points), dtype=float)


def _assert_sigma_table_matches_run(
    sigma_table: SigmaUnitTable,
    profile_name: str,
    mass_definition: MassDefinition,
) -> None:
    """Fail fast when the loaded sigma table does not match the active run."""

    if sigma_table.profile_name != profile_name:
        raise ValueError(
            f"Sigma table profile '{sigma_table.profile_name}' does not match run profile '{profile_name}'."
        )
    if sigma_table.mass_definition_label != mass_definition.label or not np.isclose(
        sigma_table.mass_radius_kpc,
        float(mass_definition.radius_kpc),
    ):
        raise ValueError(
            f"Sigma table mass definition '{sigma_table.mass_definition_label}' ({sigma_table.mass_radius_kpc:g} kpc) "
            f"does not match run mass definition '{mass_definition.label}' ({mass_definition.radius_kpc:g} kpc)."
        )


def _decode_hdf5_string(raw_value: Any) -> str:
    """Normalize the different scalar string encodings that HDF5 may return."""

    if isinstance(raw_value, bytes):
        return raw_value.decode("utf-8")
    if isinstance(raw_value, np.ndarray) and raw_value.shape == ():
        return _decode_hdf5_string(raw_value.item())
    return str(raw_value)


def _validate_sigma_unit_grid(values: np.ndarray, source_path: Path) -> np.ndarray:
    """
    Reject clearly broken sigma grids while tolerating tiny negative noise.

    The external interpolation producer may emit values that are numerically
    just below zero because of floating-point interpolation artefacts. Those
    values are harmless if they are tiny compared with the table scale and can
    be clipped to zero. Large negative regions, however, would invalidate the
    Jeans response interpretation and must fail fast.
    """

    if not np.isfinite(values).all():
        raise ValueError(f"Sigma table '{source_path}' contains non-finite values.")

    negative_mask = values < 0.0
    negative_fraction = float(np.mean(negative_mask))
    minimum_value = float(np.min(values))
    if negative_fraction > _MAX_ALLOWED_NEGATIVE_FRACTION or minimum_value < -_MAX_ALLOWED_NEGATIVE_ABSOLUTE_VALUE:
        raise ValueError(
            f"Sigma table '{source_path}' contains materially negative values "
            f"(minimum {minimum_value:.6e}, negative_fraction={negative_fraction:.6%})."
        )
    return np.maximum(values, 0.0)


def _parse_not_before(not_before: datetime | str | None) -> datetime:
    """
    Normalize the monitor trigger time into a timezone-aware datetime.

    We keep the parser strict because ambiguous local timestamps would make the
    file-mtime gate unreliable. If the caller passes a naive datetime, it is
    interpreted in the same `+08:00` zone as the agreed baseline.
    """

    if not_before is None:
        return DEFAULT_MONITOR_NOT_BEFORE
    if isinstance(not_before, datetime):
        return not_before if not_before.tzinfo is not None else not_before.replace(tzinfo=DEFAULT_MONITOR_NOT_BEFORE.tzinfo)
    parsed = datetime.fromisoformat(str(not_before))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=DEFAULT_MONITOR_NOT_BEFORE.tzinfo)
    return parsed


def _resolve_external_sigma_table_paths(
    external_dir: str | Path,
    devauc_mass_definition: MassDefinition,
    sersic_mass_definition: MassDefinition,
) -> dict[str, Path]:
    """Return the monitored table paths implied by the two run definitions."""

    resolved_dir = Path(external_dir).expanduser().resolve()
    return {
        "devauc": resolved_dir / devauc_mass_definition.sigma_table_filename("devauc"),
        "sersic": resolved_dir / sersic_mass_definition.sigma_table_filename("sersic"),
    }


def _inspect_sigma_table_candidate(
    table_path: Path,
    expected_profile: str,
    expected_mass_definition: MassDefinition,
    not_before: datetime,
) -> dict[str, Any]:
    """
    Validate one monitored table candidate without launching the PPC pipeline.

    This function intentionally calls the production loader. That is the
    required "double check": readiness is defined by the current source code's
    ability to read the table, not by separate test fixtures or assumptions.
    """

    if not table_path.exists():
        raise FileNotFoundError(f"Required sigma table '{table_path}' does not exist.")

    mtime = datetime.fromtimestamp(table_path.stat().st_mtime, tz=not_before.tzinfo)
    if mtime <= not_before:
        raise TimeoutError(
            f"Sigma table '{table_path}' was not updated after {not_before.isoformat()} "
            f"(mtime={mtime.isoformat()})."
        )

    table = SigmaUnitTable.from_path(table_path)
    _assert_sigma_table_matches_run(
        sigma_table=table,
        profile_name=expected_profile,
        mass_definition=expected_mass_definition,
    )

    axis_summary = {
        "gamma_length": int(table.gamma_axis.size),
        "zd_length": int(table.zd_axis.size),
        "log_re_kpc_length": int(table.log_re_kpc_axis.size),
        "n_length": None if table.n_axis is None else int(table.n_axis.size),
        "grid_shape": tuple(int(size) for size in table.values.shape),
        "min_value": float(np.min(table.values)),
        "max_value": float(np.max(table.values)),
        "mass_definition": mass_definition_metadata(expected_mass_definition),
    }
    return {
        "path": table_path.resolve(),
        "mtime": mtime,
        "table": table,
        "axis_summary": axis_summary,
    }


def wait_for_external_sigma_tables_and_run(
    output_root_dir: str,
    devauc_run_dir: str,
    sersic_run_dir: str,
    external_dir: str | Path = DEFAULT_EXTERNAL_SIGMA_DIR,
    not_before: datetime | str | None = None,
    poll_interval_seconds: float = 30.0,
    timeout_seconds: float | None = None,
    n_replicates: int | None = DEFAULT_N_REPLICATES,
    burn_in: str | int = "auto",
    random_seed: int = DEFAULT_RANDOM_SEED,
    candidate_pool_size: int | None = None,
    worker_processes: int | None = None,
) -> PosteriorPredictiveMonitorResult:
    """
    Wait for externally produced sigma tables, then launch both PPT runs.

    Why this workflow exists:
    - the external Jeans-grid thread may overwrite fixed filenames in place
    - PPC must ignore stale files from earlier runs
    - readiness must be proven using the current source loader before we trust
      the tables enough to start expensive real-data PPT runs
    """

    resolved_not_before = _parse_not_before(not_before)
    devauc_runtime_config = load_runtime_config(Path(devauc_run_dir).expanduser().resolve() / "config_snapshot.yaml")
    sersic_runtime_config = load_runtime_config(Path(sersic_run_dir).expanduser().resolve() / "config_snapshot.yaml")
    table_paths = _resolve_external_sigma_table_paths(
        external_dir,
        devauc_mass_definition=devauc_runtime_config.mass_definition,
        sersic_mass_definition=sersic_runtime_config.mass_definition,
    )
    started_at = time.monotonic()
    last_error_message = "monitor has not inspected any candidate tables yet"

    while True:
        try:
            devauc_candidate = _inspect_sigma_table_candidate(
                table_path=table_paths["devauc"],
                expected_profile="devauc",
                expected_mass_definition=devauc_runtime_config.mass_definition,
                not_before=resolved_not_before,
            )
            sersic_candidate = _inspect_sigma_table_candidate(
                table_path=table_paths["sersic"],
                expected_profile="sersic",
                expected_mass_definition=sersic_runtime_config.mass_definition,
                not_before=resolved_not_before,
            )
            break
        except (FileNotFoundError, TimeoutError, ValueError) as exc:
            last_error_message = str(exc)
            elapsed_seconds = time.monotonic() - started_at
            if timeout_seconds is not None and elapsed_seconds >= timeout_seconds:
                raise TimeoutError(last_error_message) from exc
            time.sleep(max(poll_interval_seconds, 0.0))

    aligned_n_replicates = n_replicates
    effective_tail_cap = DEFAULT_CANONICAL_POSTERIOR_DRAW_CAP
    if aligned_n_replicates is None:
        devauc_burn_in = _resolve_burn_in(burn_in, devauc_runtime_config.sampling.warmup)
        sersic_burn_in = _resolve_burn_in(burn_in, sersic_runtime_config.sampling.warmup)
        devauc_available = _load_flattened_posterior_chain(
            Path(devauc_run_dir).expanduser().resolve() / "chain.h5",
            burn_in=devauc_burn_in,
        ).shape[0]
        sersic_available = _load_flattened_posterior_chain(
            Path(sersic_run_dir).expanduser().resolve() / "chain.h5",
            burn_in=sersic_burn_in,
        ).shape[0]
        aligned_n_replicates = min(
            DEFAULT_CANONICAL_POSTERIOR_DRAW_CAP,
            int(devauc_available),
            int(sersic_available),
        )
        effective_tail_cap = int(aligned_n_replicates)

    devauc_result = run_posterior_predictive(
        run_dir=devauc_run_dir,
        sigma_table_path=str(devauc_candidate["path"]),
        output_root_dir=output_root_dir,
        n_replicates=n_replicates,
        burn_in=burn_in,
        random_seed=random_seed,
        candidate_pool_size=candidate_pool_size,
        worker_processes=worker_processes,
        posterior_draw_tail_cap=effective_tail_cap,
    )
    sersic_result = run_posterior_predictive(
        run_dir=sersic_run_dir,
        sigma_table_path=str(sersic_candidate["path"]),
        output_root_dir=output_root_dir,
        n_replicates=n_replicates,
        burn_in=burn_in,
        random_seed=random_seed + 1,
        candidate_pool_size=candidate_pool_size,
        worker_processes=worker_processes,
        posterior_draw_tail_cap=effective_tail_cap,
    )

    return PosteriorPredictiveMonitorResult(
        status="completed",
        external_dir=Path(external_dir).expanduser().resolve(),
        not_before=resolved_not_before.isoformat(),
        devauc_table_path=devauc_candidate["path"],
        sersic_table_path=sersic_candidate["path"],
        devauc_table_mtime=devauc_candidate["mtime"].isoformat(),
        sersic_table_mtime=sersic_candidate["mtime"].isoformat(),
        devauc_result=devauc_result,
        sersic_result=sersic_result,
        metadata={
            "devauc_table": devauc_candidate["axis_summary"],
            "sersic_table": sersic_candidate["axis_summary"],
            "poll_interval_seconds": float(poll_interval_seconds),
            "timeout_seconds": None if timeout_seconds is None else float(timeout_seconds),
            "requested_n_replicates": None if n_replicates is None else int(n_replicates),
            "aligned_n_replicates": int(aligned_n_replicates),
        },
    )


def _resolve_burn_in(requested_burn_in: str | int, warmup: int) -> int:
    """Normalize the CLI/API burn-in value into a concrete integer."""

    if isinstance(requested_burn_in, str):
        if requested_burn_in != "auto":
            raise ValueError("Burn-in must be an integer or the literal string 'auto'.")
        return int(warmup)
    return int(requested_burn_in)


def _safe_weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Return a weighted mean or `nan` if the weight support is empty.

    Weighted trend curves are only meaningful when at least one candidate
    contributes non-zero probability mass. Returning `nan` in empty regions is
    safer than silently substituting a different statistic, because it makes
    numerical pathologies visible in both the saved arrays and the plots.
    """

    finite_mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite_mask):
        return float("nan")
    return float(np.average(values[finite_mask], weights=weights[finite_mask]))


def _reduce_population_to_mass_bins(
    log_mstar: np.ndarray,
    values: np.ndarray,
    mass_bin_edges: np.ndarray,
    detectable_weights: np.ndarray,
    selected_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compress one parent-population realization into external-style mass-bin trends.

    Why this reducer exists:
    - the earlier implementation evaluated conditional means at fixed stellar
      mass and representative size, which does not match the external workflow
      the user wants to mirror
    - the external workflow first draws a full parent population and only then
      aggregates within stellar-mass bins
    - the three figure categories are therefore defined by three different
      within-bin averages over the same parent-population realization

    Parameters
    ----------
    log_mstar:
        Sampled stellar masses for one posterior draw.
    values:
        The quantity being reduced (`m5`, `gamma`, or `sigma_ap`).
    mass_bin_edges:
        Closed-open bin edges. The final bin includes its upper edge so the
        right-most boundary does not silently discard exact-edge samples.
    detectable_weights / selected_weights:
        Per-galaxy lensing weights used for the two selected populations.

    Returns
    -------
    dict[str, np.ndarray]
        Arrays for the three trend categories plus diagnostic counts / weight
        sums that make sparse or numerically degenerate bins inspectable.
    """

    log_mstar = np.asarray(log_mstar, dtype=float)
    values = np.asarray(values, dtype=float)
    mass_bin_edges = np.asarray(mass_bin_edges, dtype=float)
    detectable_weights = np.asarray(detectable_weights, dtype=float)
    selected_weights = np.asarray(selected_weights, dtype=float)

    n_mass_bins = mass_bin_edges.size - 1
    parent = np.full(n_mass_bins, np.nan, dtype=float)
    detectable = np.full(n_mass_bins, np.nan, dtype=float)
    selected = np.full(n_mass_bins, np.nan, dtype=float)
    parent_bin_counts = np.zeros(n_mass_bins, dtype=int)
    detectable_weight_sums = np.zeros(n_mass_bins, dtype=float)
    selected_weight_sums = np.zeros(n_mass_bins, dtype=float)

    finite_parent_support = np.isfinite(log_mstar) & np.isfinite(values)
    finite_detectable_support = finite_parent_support & np.isfinite(detectable_weights) & (detectable_weights > 0.0)
    finite_selected_support = finite_parent_support & np.isfinite(selected_weights) & (selected_weights > 0.0)

    for bin_index in range(n_mass_bins):
        lower_edge = mass_bin_edges[bin_index]
        upper_edge = mass_bin_edges[bin_index + 1]
        in_bin = log_mstar >= lower_edge
        if bin_index == n_mass_bins - 1:
            in_bin &= log_mstar <= upper_edge
        else:
            in_bin &= log_mstar < upper_edge

        parent_mask = finite_parent_support & in_bin
        parent_bin_counts[bin_index] = int(np.count_nonzero(parent_mask))
        if np.any(parent_mask):
            parent[bin_index] = float(np.mean(values[parent_mask]))

        detectable_mask = finite_detectable_support & in_bin
        detectable_weight_sums[bin_index] = float(np.sum(detectable_weights[detectable_mask]))
        detectable[bin_index] = _safe_weighted_average(values[detectable_mask], detectable_weights[detectable_mask])

        selected_mask = finite_selected_support & in_bin
        selected_weight_sums[bin_index] = float(np.sum(selected_weights[selected_mask]))
        selected[bin_index] = _safe_weighted_average(values[selected_mask], selected_weights[selected_mask])

    return {
        "parent": parent,
        "detectable": detectable,
        "selected": selected,
        "parent_bin_counts": parent_bin_counts,
        "detectable_weight_sums": detectable_weight_sums,
        "selected_weight_sums": selected_weight_sums,
    }


def _load_flattened_posterior_chain(chain_path: Path, burn_in: int) -> np.ndarray:
    """
    Load and flatten the post-burn-in posterior chain.

    Separating this step from PPC draw selection keeps the code explicit about
    which policy is responsible for:
    - discarding the requested warmup segment
    - flattening the walker/time grid
    - optionally tail-capping or sub-sampling the resulting chain
    """

    backend = emcee.backends.HDFBackend(str(chain_path))
    chain = backend.get_chain()
    if burn_in >= chain.shape[0]:
        raise ValueError(
            f"Burn-in {burn_in} removes all samples from chain with {chain.shape[0]} stored steps."
        )
    return chain[burn_in:].reshape(-1, chain.shape[-1])


def _select_posterior_draws(
    flattened_chain: np.ndarray,
    n_replicates: int | None,
    rng: np.random.Generator,
    tail_cap: int = DEFAULT_CANONICAL_POSTERIOR_DRAW_CAP,
) -> tuple[np.ndarray, str]:
    """
    Choose the posterior draws used by one PPC run.

    The default canonical mode uses the tail of the flattened chain rather than
    randomly re-sampling it. This keeps the result size bounded while still
    honoring the user's request that PPC should operate on the stored posterior
    chain itself when no explicit sub-sample size is requested.
    """

    if n_replicates is None:
        used_draw_count = min(int(flattened_chain.shape[0]), int(tail_cap))
        return flattened_chain[-used_draw_count:], "tail_capped_full_chain"

    draw_indices = rng.integers(0, flattened_chain.shape[0], size=int(n_replicates))
    return flattened_chain[draw_indices], "sampled_subset"


def _load_posterior_draws(
    chain_path: Path,
    burn_in: int,
    rng: np.random.Generator,
    n_replicates: int | None,
    tail_cap: int = DEFAULT_CANONICAL_POSTERIOR_DRAW_CAP,
) -> tuple[np.ndarray, str]:
    """
    Load the flattened posterior chain and apply the agreed PPC draw policy.

    The caller receives both the selected draws and the mode string so summary
    files can explain whether a run used the canonical tail-capped chain or an
    explicitly requested posterior subset.
    """

    flattened_chain = _load_flattened_posterior_chain(chain_path=chain_path, burn_in=burn_in)
    return _select_posterior_draws(
        flattened_chain=flattened_chain,
        n_replicates=n_replicates,
        rng=rng,
        tail_cap=tail_cap,
    )


def _resolve_candidate_pool_size(candidate_pool_size: int | None, base_normals_count: int) -> int:
    """
    Resolve the effective candidate-pool size for one PPC run.

    Why this helper exists:
    - the PPC candidate pool has a different meaning from the normalization MC
      sample count, so the policy should stay explicit and testable
    - the canonical cap is now `100000`, but synthetic tests and smaller runs
      should still clamp to the available random-basis bank
    - the pool must never shrink below the requested replicated-sample sizes
    """

    resolved = int(
        candidate_pool_size
        if candidate_pool_size is not None
        else min(int(base_normals_count), DEFAULT_MAX_CANDIDATE_POOL_SIZE)
    )
    return max(resolved, THETA_SAMPLE_SIZE, SIGMA_SAMPLE_SIZE)


def _vectorized_skewnorm_sample(z0: np.ndarray, z1: np.ndarray, loc: float, scale: float, alpha: float) -> np.ndarray:
    """Vectorized skew-normal sampling using the same construction as the kernel."""

    delta = alpha / math.sqrt(1.0 + alpha * alpha)
    return loc + scale * (delta * np.abs(z0) + math.sqrt(1.0 - delta * delta) * z1)


def _vectorized_truncnorm_sample(
    z_u: np.ndarray,
    loc: np.ndarray,
    scale: float,
    low: float,
    high: float,
) -> np.ndarray:
    """
    Vectorized truncated-normal sampling that mirrors the kernel approximation.

    The `z_u` inputs are first mapped through the standard-normal CDF so that
    the reused basis values become uniform draws in `[0, 1]`.
    """

    a = (low - loc) / scale
    b = (high - loc) / scale
    pa = ndtr(a)
    pb = ndtr(b)
    u = np.clip(ndtr(z_u), 1.0e-12, 1.0 - 1.0e-12)
    q = pa + (pb - pa) * u
    sampled = loc + scale * ndtri(np.clip(q, 1.0e-12, 1.0 - 1.0e-12))
    return np.clip(sampled, low, high)


def _vectorized_mu_r(mstar: np.ndarray, n_value: np.ndarray, profile: ProfileSpec) -> np.ndarray:
    """Vectorized mean-size relation shared by PPC candidate generation."""

    out = profile.mu_r0 + profile.beta_r * (mstar - 11.4)
    if profile.nu_r is not None:
        out += profile.nu_r * (np.log10(np.maximum(n_value, 1.0e-12)) - math.log10(4.0))
    return out


def _vectorized_theta_ein_arcsec(
    zd: np.ndarray,
    zs: np.ndarray,
    log_enclosed_mass: np.ndarray,
    gamma: np.ndarray,
    z_grid: np.ndarray,
    chi_kpc_grid: np.ndarray,
    mass_radius_kpc: float,
) -> np.ndarray:
    """
    Vectorized Einstein-radius calculation copied from the scalar primitive.

    This helper intentionally mirrors the inference code's geometry formula so
    the PPC pipeline remains statistically aligned with the fitted model.
    """

    c_km_s = 299792.458
    g_kpc_kms2_msun = 4.30091e-6

    theta_ein = np.zeros_like(gamma, dtype=float)
    valid = (zd > 0.0) & (zs > zd) & (gamma > 1.0)
    if not np.any(valid):
        return theta_ein

    chi_l = np.interp(zd[valid], z_grid, chi_kpc_grid, left=float(chi_kpc_grid[0]), right=float(chi_kpc_grid[-1]))
    chi_s = np.interp(zs[valid], z_grid, chi_kpc_grid, left=float(chi_kpc_grid[0]), right=float(chi_kpc_grid[-1]))

    dl = chi_l / (1.0 + zd[valid])
    ds = chi_s / (1.0 + zs[valid])
    dls = (chi_s - chi_l) / (1.0 + zs[valid])
    geometry_ok = (chi_s > chi_l) & (dl > 0.0) & (ds > 0.0) & (dls > 0.0)
    if not np.any(geometry_ok):
        return theta_ein

    inner_indices = np.flatnonzero(valid)[geometry_ok]
    sigma_crit = (c_km_s * c_km_s) / (4.0 * math.pi * g_kpc_kms2_msun) * (ds[geometry_ok] / (dl[geometry_ok] * dls[geometry_ok]))
    base = (10.0 ** log_enclosed_mass[inner_indices]) / (
        math.pi * sigma_crit * (mass_radius_kpc ** (3.0 - gamma[inner_indices]))
    )
    physical_ok = base > 0.0
    if np.any(physical_ok):
        chosen = inner_indices[physical_ok]
        r_ein = base[physical_ok] ** (1.0 / (gamma[chosen] - 1.0))
        theta_ein[chosen] = r_ein / dl[geometry_ok][physical_ok] * 206265.0
    return theta_ein


def _discovery_probability(theta_ein: np.ndarray, theta0: float, loga: float) -> np.ndarray:
    """Vectorized version of the strong-lens discovery sigmoid."""

    a = 10.0 ** loga
    x = -a * (theta_ein - theta0)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(x))


def _expected_theta_dimension_for_gamma_mode(gamma_mode_code: int) -> int:
    """
    Return the sampled theta dimension expected by the active gamma mode.

    PPC consumes posterior draws after inference has finished, so it cannot
    rely on the sampler to catch a mismatched chain shape. The helper keeps
    that validation close to the chain consumer and makes the mode-specific
    dimensional contract explicit.
    """

    if gamma_mode_code == GAMMA_MODE_DEPENDENT_CODE:
        return 12
    if gamma_mode_code == GAMMA_MODE_INDEPENDENT_CODE:
        return 10
    raise ValueError(f"Unsupported gamma mode code '{gamma_mode_code}' in posterior predictive workflow.")


def _unpack_population_theta(theta: np.ndarray, gamma_mode_code: int) -> dict[str, float]:
    """
    Decode one posterior draw into named scalar parameters for PPC generation.

    The gamma mode now changes the actual sampled vector length. Converting the
    raw `theta` row into a named mapping up front prevents later code from
    scattering fragile index arithmetic across the population generator.
    """

    theta_array = np.asarray(theta, dtype=float)
    expected_dimension = _expected_theta_dimension_for_gamma_mode(gamma_mode_code)
    if theta_array.shape != (expected_dimension,):
        raise ValueError(
            "Posterior draw has the wrong dimension for posterior predictive "
            f"generation: expected {expected_dimension}, got {theta_array.shape}."
        )

    if gamma_mode_code == GAMMA_MODE_DEPENDENT_CODE:
        return {
            "mu5_0": float(theta_array[0]),
            "beta5": float(theta_array[1]),
            "xi5": float(theta_array[2]),
            "sigma5": float(theta_array[3]),
            "mu_gamma_0": float(theta_array[4]),
            "beta_gamma": float(theta_array[5]),
            "xi_gamma": float(theta_array[6]),
            "sigma_gamma": float(theta_array[7]),
            "mu_zs": float(theta_array[8]),
            "sigma_zs": float(theta_array[9]),
            "theta0": float(theta_array[10]),
            "loga": float(theta_array[11]),
        }

    return {
        "mu5_0": float(theta_array[0]),
        "beta5": float(theta_array[1]),
        "xi5": float(theta_array[2]),
        "sigma5": float(theta_array[3]),
        "mu_gamma_0": float(theta_array[4]),
        # In the independent mode the gamma slopes are removed from the sampled
        # vector entirely, so PPC reconstructs the scientific contract
        # explicitly instead of assuming hidden placeholder slots still exist.
        "beta_gamma": 0.0,
        "xi_gamma": 0.0,
        "sigma_gamma": float(theta_array[5]),
        "mu_zs": float(theta_array[6]),
        "sigma_zs": float(theta_array[7]),
        "theta0": float(theta_array[8]),
        "loga": float(theta_array[9]),
    }


def _gamma_population_mean(
    mu_gamma_0: float,
    beta_gamma: float,
    xi_gamma: float,
    mstar_shift11p4: np.ndarray,
    delta_r: np.ndarray,
    gamma_mode_code: int,
) -> np.ndarray:
    """
    Evaluate the mode-aware population mean of `gamma` for one PPC draw.

    This helper is shared by histogram PPC and trend generation so both
    workflows follow exactly the same rule for the new independent mode.
    """

    if gamma_mode_code == GAMMA_MODE_DEPENDENT_CODE:
        return mu_gamma_0 + beta_gamma * mstar_shift11p4 + xi_gamma * delta_r
    if gamma_mode_code == GAMMA_MODE_INDEPENDENT_CODE:
        return np.full_like(mstar_shift11p4, mu_gamma_0, dtype=float)
    raise ValueError(f"Unsupported gamma mode code '{gamma_mode_code}' in posterior predictive workflow.")


def _draw_candidate_population(
    theta: np.ndarray,
    profile: ProfileSpec,
    context,
    rng: np.random.Generator,
    candidate_pool_size: int,
) -> dict[str, np.ndarray]:
    """
    Generate one sampled parent population for a single posterior draw.

    This helper now serves two downstream workflows:
    - histogram PPC, which resamples explicit replicated lenses from the
      selected population
    - Fig. 8-like trend evaluation, which aggregates the parent population into
      stellar-mass bins without first drawing explicit lens catalogs

    The candidate pool is built from the same fixed random-basis bank used by
    normalization so the latent sampling law stays aligned with the fitted
    model. The requested `candidate_pool_size` therefore controls how many
    parent-population galaxies we materialize for the downstream workflow, not
    the original normalization Monte Carlo sample count itself.
    """

    basis = context.base_normals
    if candidate_pool_size == basis.shape[0]:
        normals = basis
    else:
        replace = basis.shape[0] < candidate_pool_size
        sampled_indices = rng.choice(basis.shape[0], size=candidate_pool_size, replace=replace)
        normals = basis[sampled_indices]

    theta_components = _unpack_population_theta(theta=theta, gamma_mode_code=context.gamma_mode_code)
    mu5_0 = theta_components["mu5_0"]
    beta5 = theta_components["beta5"]
    xi5 = theta_components["xi5"]
    sigma5 = theta_components["sigma5"]
    mu_gamma_0 = theta_components["mu_gamma_0"]
    beta_gamma = theta_components["beta_gamma"]
    xi_gamma = theta_components["xi_gamma"]
    sigma_gamma = theta_components["sigma_gamma"]
    mu_zs = theta_components["mu_zs"]
    sigma_zs = theta_components["sigma_zs"]
    theta0 = theta_components["theta0"]
    loga = theta_components["loga"]

    mstar = _vectorized_skewnorm_sample(
        normals[:, 2],
        normals[:, 3],
        profile.mass_function_loc,
        profile.mass_function_scale,
        profile.mass_function_alpha,
    )
    mstar_shift11p4 = mstar - 11.4
    zd = context.mu_d + context.sigma_d * normals[:, 0]
    zs = mu_zs + sigma_zs * normals[:, 1]

    if profile.fixed_n is None:
        logn = profile.mu_n0 + profile.beta_n * mstar_shift11p4 + profile.sigma_n * normals[:, 4]
        n_value = np.power(10.0, logn)
        mu_r = _vectorized_mu_r(mstar, n_value, profile)
        re_log_kpc = mu_r + profile.sigma_r * normals[:, 5]
        delta_r = re_log_kpc - mu_r
        log_enclosed_mass = mu5_0 + beta5 * mstar_shift11p4 + xi5 * delta_r + sigma5 * normals[:, 6]
    else:
        n_value = np.full(candidate_pool_size, profile.fixed_n, dtype=float)
        mu_r = _vectorized_mu_r(mstar, n_value, profile)
        re_log_kpc = mu_r + profile.sigma_r * normals[:, 4]
        delta_r = re_log_kpc - mu_r
        log_enclosed_mass = mu5_0 + beta5 * mstar_shift11p4 + xi5 * delta_r + sigma5 * normals[:, 5]

    mu_gamma = _gamma_population_mean(
        mu_gamma_0=mu_gamma_0,
        beta_gamma=beta_gamma,
        xi_gamma=xi_gamma,
        mstar_shift11p4=mstar_shift11p4,
        delta_r=delta_r,
        gamma_mode_code=context.gamma_mode_code,
    )

    gamma = _vectorized_truncnorm_sample(
        normals[:, 7],
        loc=mu_gamma,
        scale=sigma_gamma,
        low=context.gamma_trunc_low,
        high=context.gamma_trunc_high,
    )
    re_kpc = np.power(10.0, re_log_kpc)

    theta_ein = _vectorized_theta_ein_arcsec(
        zd=zd,
        zs=zs,
        log_enclosed_mass=log_enclosed_mass,
        gamma=gamma,
        z_grid=context.z_grid,
        chi_kpc_grid=context.chi_kpc_grid,
        mass_radius_kpc=float(context.mass_radius_kpc),
    )
    cs_over_theta = np.interp(
        gamma,
        context.cs_gamma_grid,
        context.cs_over_theta_grid,
        left=float(context.cs_over_theta_grid[0]),
        right=float(context.cs_over_theta_grid[-1]),
    )
    discovery_probability = _discovery_probability(theta_ein, theta0=theta0, loga=loga)
    area = math.pi * np.square(cs_over_theta * theta_ein)

    valid_geometry = (
        np.isfinite(gamma)
        & np.isfinite(log_enclosed_mass)
        & np.isfinite(re_kpc)
        & (theta_ein > 0.0)
        & (zs > zd)
        & (zd > 0.0)
        & (zs > 0.0)
        & (area > 0.0)
    )
    detectable_weights = np.where(valid_geometry, area, 0.0)
    selected_weights = detectable_weights * np.where(
        valid_geometry & np.isfinite(discovery_probability) & (discovery_probability > 0.0),
        discovery_probability,
        0.0,
    )
    return {
        "log_mstar": mstar,
        "theta_ein": theta_ein,
        "gamma": gamma,
        "zd": zd,
        "zs": zs,
        "mass": log_enclosed_mass,
        "log_re_kpc": re_log_kpc,
        "re_kpc": re_kpc,
        "n": n_value,
        "detectable_weights": detectable_weights,
        "selected_weights": selected_weights,
        # Keep the legacy key so the histogram PPC code path remains unchanged.
        "weights": selected_weights,
    }


def _draw_trend_parent_population(
    theta: np.ndarray,
    profile: ProfileSpec,
    context,
    sigma_table: SigmaUnitTable,
    rng: np.random.Generator,
    n_parent_sample: int,
) -> dict[str, np.ndarray]:
    """
    Materialize one full parent-population realization for Fig. 8-like trends.

    Why this helper is separate from the histogram PPC path:
    - histogram PPC needs only the selected-lens weights plus latent values
    - the trend figure needs the full parent population, its stellar masses,
      its structural scatter, and the physical aperture-dispersion prediction
      for every sampled galaxy before any mass-bin averaging is performed

    The returned payload is intentionally verbose. The trend reducer is a pure
    binning step, so exposing all latent arrays here makes the scientific data
    flow transparent and easy to test.
    """

    population = _draw_candidate_population(
        theta=theta,
        profile=profile,
        context=context,
        rng=rng,
        candidate_pool_size=n_parent_sample,
    )
    sigma_unit = sigma_table.evaluate(
        gamma=population["gamma"],
        zd=population["zd"],
        log_re_kpc=population["log_re_kpc"],
        n_values=None if profile.fixed_n is not None else population["n"],
    )
    sigma_ap = np.sqrt(np.maximum(sigma_unit * np.power(10.0, population["mass"]), 1.0e-30))

    return {
        "log_mstar": population["log_mstar"],
        "zd": population["zd"],
        "zs": population["zs"],
        "n": population["n"],
        "log_re_kpc": population["log_re_kpc"],
        "re_kpc": population["re_kpc"],
        "mass": population["mass"],
        "gamma": population["gamma"],
        "theta_ein": population["theta_ein"],
        "sigma_ap": sigma_ap,
        "detectable_weights": population["detectable_weights"],
        "selected_weights": population["selected_weights"],
    }


def _allocate_trend_arrays(
    n_draws: int,
    n_mass_bins: int,
    mass_definition: MassDefinition,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Allocate the full set of trend buffers for one run or one worker chunk.

    Why this helper exists:
    - the trend workflow now supports both serial and process-pool execution
    - both execution modes must populate the exact same array contract
    - centralizing allocation prevents the two paths from silently diverging in
      shape or dtype as the trend result payload evolves
    """

    quantity_names = _trend_quantity_names(mass_definition)
    trend_draws = {
        quantity_name: {
            category_name: np.full((n_draws, n_mass_bins), np.nan, dtype=float)
            for category_name in TREND_CATEGORY_NAMES
        }
        for quantity_name in quantity_names
    }
    parent_bin_counts_draws = np.zeros((n_draws, n_mass_bins), dtype=int)
    detectable_weight_sums_draws = np.zeros((n_draws, n_mass_bins), dtype=float)
    selected_weight_sums_draws = np.zeros((n_draws, n_mass_bins), dtype=float)
    return trend_draws, parent_bin_counts_draws, detectable_weight_sums_draws, selected_weight_sums_draws


def _simulate_trend_chunk(
    posterior_draws: np.ndarray,
    start_index: int,
    profile: ProfileSpec,
    context,
    mass_definition: MassDefinition,
    sigma_table_path: str,
    mass_bin_edges: np.ndarray,
    n_parent_sample: int,
    random_seed: int,
) -> dict[str, Any]:
    """
    Simulate one contiguous chunk of posterior draws for the trend workflow.

    This mirrors the PPC chunk worker contract:
    - the worker is fully self-contained and can run inline or in a spawned
      subprocess without depending on mutable shared state
    - per-draw randomness is keyed to the global posterior draw index so serial
      and parallel execution produce byte-identical scientific outputs
    """

    apply_thread_limits(1)
    sigma_table = SigmaUnitTable.from_path(sigma_table_path)
    chunk_draw_count = int(posterior_draws.shape[0])
    n_mass_bins = int(mass_bin_edges.size - 1)
    quantity_names = _trend_quantity_names(mass_definition)
    trend_draws, parent_bin_counts_draws, detectable_weight_sums_draws, selected_weight_sums_draws = (
        _allocate_trend_arrays(chunk_draw_count, n_mass_bins, mass_definition)
    )

    for local_index, theta in enumerate(posterior_draws):
        global_index = start_index + local_index
        rng = np.random.default_rng(_build_seed_sequence(random_seed, global_index))
        parent_population = _draw_trend_parent_population(
            theta=theta,
            profile=profile,
            context=context,
            sigma_table=sigma_table,
            rng=rng,
            n_parent_sample=n_parent_sample,
        )

        for quantity_name in quantity_names:
            if quantity_name == mass_definition.label:
                values = parent_population["mass"]
            else:
                values = parent_population[quantity_name]
            reduced = _reduce_population_to_mass_bins(
                log_mstar=parent_population["log_mstar"],
                values=values,
                mass_bin_edges=mass_bin_edges,
                detectable_weights=parent_population["detectable_weights"],
                selected_weights=parent_population["selected_weights"],
            )
            for category_name in TREND_CATEGORY_NAMES:
                trend_draws[quantity_name][category_name][local_index] = reduced[category_name]

            if quantity_name == quantity_names[0]:
                parent_bin_counts_draws[local_index] = reduced["parent_bin_counts"]
                detectable_weight_sums_draws[local_index] = reduced["detectable_weight_sums"]
                selected_weight_sums_draws[local_index] = reduced["selected_weight_sums"]

    return {
        "start_index": int(start_index),
        "draw_count": chunk_draw_count,
        "trend_draws": trend_draws,
        "parent_bin_counts_draws": parent_bin_counts_draws,
        "detectable_weight_sums_draws": detectable_weight_sums_draws,
        "selected_weight_sums_draws": selected_weight_sums_draws,
    }


def _merge_trend_chunk_results(
    chunk_results: list[dict[str, Any]],
    n_draws: int,
    n_mass_bins: int,
    mass_definition: MassDefinition,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    """Merge chunk-level trend arrays back into full-run buffers in draw order."""

    quantity_names = _trend_quantity_names(mass_definition)
    trend_draws, parent_bin_counts_draws, detectable_weight_sums_draws, selected_weight_sums_draws = (
        _allocate_trend_arrays(n_draws, n_mass_bins, mass_definition)
    )
    for chunk_result in chunk_results:
        start = int(chunk_result["start_index"])
        stop = start + int(chunk_result["draw_count"])
        for quantity_name in quantity_names:
            for category_name in TREND_CATEGORY_NAMES:
                trend_draws[quantity_name][category_name][start:stop] = chunk_result["trend_draws"][quantity_name][
                    category_name
                ]
        parent_bin_counts_draws[start:stop] = chunk_result["parent_bin_counts_draws"]
        detectable_weight_sums_draws[start:stop] = chunk_result["detectable_weight_sums_draws"]
        selected_weight_sums_draws[start:stop] = chunk_result["selected_weight_sums_draws"]
    return trend_draws, parent_bin_counts_draws, detectable_weight_sums_draws, selected_weight_sums_draws


def _run_trend_draws(
    posterior_draws: np.ndarray,
    profile: ProfileSpec,
    context,
    mass_definition: MassDefinition,
    sigma_table_path: str,
    mass_bin_edges: np.ndarray,
    n_parent_sample: int,
    random_seed: int,
    parallelism: ResolvedParallelism,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute trend draws either serially or with a process pool.

    The unit of work is a contiguous posterior-draw chunk so that the trend
    workflow inherits the same scaling and deterministic chunk semantics as the
    canonical PPC path. This is necessary once trend defaults switch from 256
    sampled draws to the tail-capped full chain.
    """

    n_draws = int(posterior_draws.shape[0])
    n_mass_bins = int(mass_bin_edges.size - 1)
    chunk_count = max(1, parallelism.worker_processes) if parallelism.strategy == "process_pool" else 1
    slices = _chunk_slices(n_draws, chunk_count)

    if parallelism.strategy != "process_pool":
        return _merge_trend_chunk_results(
            [
                _simulate_trend_chunk(
                    posterior_draws=posterior_draws[work_slice],
                    start_index=work_slice.start,
                    profile=profile,
                    context=context,
                    mass_definition=mass_definition,
                    sigma_table_path=sigma_table_path,
                    mass_bin_edges=mass_bin_edges,
                    n_parent_sample=n_parent_sample,
                    random_seed=random_seed,
                )
                for work_slice in slices
            ],
            n_draws=n_draws,
            n_mass_bins=n_mass_bins,
            mass_definition=mass_definition,
        )

    spawn_context = multiprocessing.get_context("spawn")
    chunk_results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=parallelism.worker_processes,
        mp_context=spawn_context,
    ) as executor:
        futures = [
            executor.submit(
                _simulate_trend_chunk,
                posterior_draws[work_slice],
                work_slice.start,
                profile,
                context,
                mass_definition,
                sigma_table_path,
                mass_bin_edges,
                n_parent_sample,
                random_seed,
            )
            for work_slice in slices
        ]
        for future in futures:
            chunk_results.append(future.result())

    return _merge_trend_chunk_results(
        chunk_results,
        n_draws=n_draws,
        n_mass_bins=n_mass_bins,
        mass_definition=mass_definition,
    )


def _draw_replicated_lenses(
    candidates: dict[str, np.ndarray],
    sample_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw one replicated lens set from the weighted candidate population."""

    positive_indices = np.flatnonzero(candidates["weights"] > 0.0)
    if positive_indices.size == 0:
        raise ValueError("Candidate population contains no positive selection weights.")

    weights = candidates["weights"][positive_indices]
    probabilities = weights / weights.sum()
    replace = positive_indices.size < sample_size
    chosen = rng.choice(positive_indices, size=sample_size, replace=replace, p=probabilities)
    return {
        "theta_ein": candidates["theta_ein"][chosen],
        "gamma": candidates["gamma"][chosen],
        "zd": candidates["zd"][chosen],
        "zs": candidates["zs"][chosen],
        "mass": candidates["mass"][chosen],
        "re_kpc": candidates["re_kpc"][chosen],
        "n": candidates["n"][chosen],
    }


def _resolve_ppc_parallelism(runtime_config, requested_worker_processes: int | None, n_draws: int) -> ResolvedParallelism:
    """
    Resolve the PPC worker count independently from the sampler's walker logic.

    PPC parallelism is posterior-draw based, not walker-based, so reusing the
    sampler's resolver directly would encode the wrong unit of work. This
    helper mirrors the same CPU-budget policy while explicitly targeting
    process-based chunk execution.
    """

    cpu_count = max(1, int(os.cpu_count() or 1))
    reserve_cores = max(0, int(runtime_config.runtime.reserve_cores))
    auto_budget = max(1, cpu_count - reserve_cores)
    if requested_worker_processes is None:
        requested = auto_budget if int(runtime_config.runtime.num_threads) <= 0 else min(
            int(runtime_config.runtime.num_threads),
            auto_budget,
        )
    else:
        requested = max(1, min(int(requested_worker_processes), auto_budget))

    worker_processes = min(max(1, int(requested)), max(1, int(n_draws)))
    strategy = "process_pool" if worker_processes > 1 else "off"
    return ResolvedParallelism(
        strategy=strategy,
        cpu_count=cpu_count,
        reserve_cores=reserve_cores,
        compute_budget=auto_budget,
        worker_processes=worker_processes if strategy == "process_pool" else 0,
        kernel_threads_per_process=1,
    )


def _build_seed_sequence(base_seed: int, draw_index: int) -> np.random.SeedSequence:
    """Derive a deterministic per-draw seed that is independent of chunking."""

    return np.random.SeedSequence([int(base_seed), int(draw_index)])


def _allocate_replicated_arrays(
    n_draws: int,
    mass_definition: MassDefinition,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Allocate the PPC latent arrays and replicate-level statistic vectors.

    Keeping allocation in one helper makes the shape contract explicit and
    avoids diverging serial vs parallel buffer layouts.
    """

    mass_label = mass_definition.label
    theta_latent = {
        "theta_ein": np.zeros((n_draws, THETA_SAMPLE_SIZE), dtype=float),
        "gamma": np.zeros((n_draws, THETA_SAMPLE_SIZE), dtype=float),
        "zd": np.zeros((n_draws, THETA_SAMPLE_SIZE), dtype=float),
        "zs": np.zeros((n_draws, THETA_SAMPLE_SIZE), dtype=float),
        mass_label: np.zeros((n_draws, THETA_SAMPLE_SIZE), dtype=float),
        "re_kpc": np.zeros((n_draws, THETA_SAMPLE_SIZE), dtype=float),
        "n": np.zeros((n_draws, THETA_SAMPLE_SIZE), dtype=float),
    }
    sigma_latent = {
        "sigma": np.zeros((n_draws, SIGMA_SAMPLE_SIZE), dtype=float),
        "gamma": np.zeros((n_draws, SIGMA_SAMPLE_SIZE), dtype=float),
        "zd": np.zeros((n_draws, SIGMA_SAMPLE_SIZE), dtype=float),
        "zs": np.zeros((n_draws, SIGMA_SAMPLE_SIZE), dtype=float),
        mass_label: np.zeros((n_draws, SIGMA_SAMPLE_SIZE), dtype=float),
        "re_kpc": np.zeros((n_draws, SIGMA_SAMPLE_SIZE), dtype=float),
        "n": np.zeros((n_draws, SIGMA_SAMPLE_SIZE), dtype=float),
        "theta_ein": np.zeros((n_draws, SIGMA_SAMPLE_SIZE), dtype=float),
    }
    theta_replicated_stats = {name: np.zeros(n_draws, dtype=float) for name in SUMMARY_STAT_NAMES}
    sigma_replicated_stats = {name: np.zeros(n_draws, dtype=float) for name in SUMMARY_STAT_NAMES}
    return theta_latent, sigma_latent, theta_replicated_stats, sigma_replicated_stats


def _simulate_ppc_chunk(
    posterior_draws: np.ndarray,
    start_index: int,
    profile: ProfileSpec,
    context,
    mass_definition: MassDefinition,
    sigma_table_path: str,
    candidate_pool_size: int,
    random_seed: int,
) -> dict[str, Any]:
    """
    Simulate one contiguous chunk of posterior draws for PPC.

    The worker contract is intentionally self-contained so the same function can
    run inline in the parent process or inside a spawned worker process. This
    keeps the serial and parallel code paths scientifically identical.
    """

    apply_thread_limits(1)
    sigma_table = SigmaUnitTable.from_path(sigma_table_path)
    chunk_draw_count = int(posterior_draws.shape[0])
    theta_latent, sigma_latent, theta_replicated_stats, sigma_replicated_stats = _allocate_replicated_arrays(
        chunk_draw_count,
        mass_definition,
    )
    mass_label = mass_definition.label

    for local_index, theta in enumerate(posterior_draws):
        global_index = start_index + local_index
        rng = np.random.default_rng(_build_seed_sequence(random_seed, global_index))
        candidates = _draw_candidate_population(
            theta=theta,
            profile=profile,
            context=context,
            rng=rng,
            candidate_pool_size=candidate_pool_size,
        )
        theta_sample = _draw_replicated_lenses(candidates, sample_size=THETA_SAMPLE_SIZE, rng=rng)
        sigma_sample = _draw_replicated_lenses(candidates, sample_size=SIGMA_SAMPLE_SIZE, rng=rng)

        theta_latent["theta_ein"][local_index] = theta_sample["theta_ein"]
        theta_latent["gamma"][local_index] = theta_sample["gamma"]
        theta_latent["zd"][local_index] = theta_sample["zd"]
        theta_latent["zs"][local_index] = theta_sample["zs"]
        theta_latent[mass_label][local_index] = theta_sample["mass"]
        theta_latent["re_kpc"][local_index] = theta_sample["re_kpc"]
        theta_latent["n"][local_index] = theta_sample["n"]

        theta_stats = _summary_statistics(theta_sample["theta_ein"])
        for stat_name, stat_value in theta_stats.items():
            theta_replicated_stats[stat_name][local_index] = stat_value

        sigma_unit = sigma_table.evaluate(
            gamma=sigma_sample["gamma"],
            zd=sigma_sample["zd"],
            log_re_kpc=np.log10(np.maximum(sigma_sample["re_kpc"], 1.0e-12)),
            n_values=None if profile.fixed_n is not None else sigma_sample["n"],
        )
        sigma_model = np.sqrt(np.maximum(sigma_unit * (10.0 ** sigma_sample["mass"]), 1.0e-30))
        sigma_rep = rng.normal(loc=sigma_model, scale=SIGMA_RELATIVE_NOISE * sigma_model)

        sigma_latent["sigma"][local_index] = sigma_rep
        sigma_latent["gamma"][local_index] = sigma_sample["gamma"]
        sigma_latent["zd"][local_index] = sigma_sample["zd"]
        sigma_latent["zs"][local_index] = sigma_sample["zs"]
        sigma_latent[mass_label][local_index] = sigma_sample["mass"]
        sigma_latent["re_kpc"][local_index] = sigma_sample["re_kpc"]
        sigma_latent["n"][local_index] = sigma_sample["n"]
        sigma_latent["theta_ein"][local_index] = sigma_sample["theta_ein"]

        sigma_stats = _summary_statistics(sigma_rep)
        for stat_name, stat_value in sigma_stats.items():
            sigma_replicated_stats[stat_name][local_index] = stat_value

    return {
        "start_index": int(start_index),
        "draw_count": chunk_draw_count,
        "theta_latent": theta_latent,
        "sigma_latent": sigma_latent,
        "theta_replicated_stats": theta_replicated_stats,
        "sigma_replicated_stats": sigma_replicated_stats,
    }


def _chunk_slices(n_items: int, n_chunks: int) -> list[slice]:
    """Split a 1D workload into contiguous slices with near-equal lengths."""

    if n_items < 1:
        return []
    boundaries = np.linspace(0, n_items, num=n_chunks + 1, dtype=int)
    return [slice(int(boundaries[i]), int(boundaries[i + 1])) for i in range(n_chunks) if boundaries[i] < boundaries[i + 1]]


def _merge_ppc_chunk_results(
    chunk_results: list[dict[str, Any]],
    n_draws: int,
    mass_definition: MassDefinition,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Merge chunk-level PPC arrays back into full-run buffers in draw order."""

    theta_latent, sigma_latent, theta_replicated_stats, sigma_replicated_stats = _allocate_replicated_arrays(
        n_draws,
        mass_definition,
    )
    for chunk_result in chunk_results:
        start = int(chunk_result["start_index"])
        stop = start + int(chunk_result["draw_count"])
        for key, values in chunk_result["theta_latent"].items():
            theta_latent[key][start:stop] = values
        for key, values in chunk_result["sigma_latent"].items():
            sigma_latent[key][start:stop] = values
        for key, values in chunk_result["theta_replicated_stats"].items():
            theta_replicated_stats[key][start:stop] = values
        for key, values in chunk_result["sigma_replicated_stats"].items():
            sigma_replicated_stats[key][start:stop] = values
    return theta_latent, sigma_latent, theta_replicated_stats, sigma_replicated_stats


def _run_ppc_replicates(
    posterior_draws: np.ndarray,
    profile: ProfileSpec,
    context,
    mass_definition: MassDefinition,
    sigma_table_path: str,
    candidate_pool_size: int,
    random_seed: int,
    parallelism: ResolvedParallelism,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Execute the PPC replicate simulation either serially or with a process pool.

    The unit of parallel work is a contiguous block of posterior draws. This
    keeps IPC overhead bounded while still giving each worker enough work to
    amortize startup costs on the large canonical runs.
    """

    n_draws = int(posterior_draws.shape[0])
    chunk_count = max(1, parallelism.worker_processes) if parallelism.strategy == "process_pool" else 1
    slices = _chunk_slices(n_draws, chunk_count)

    if parallelism.strategy != "process_pool":
        return _merge_ppc_chunk_results(
            [
                _simulate_ppc_chunk(
                    posterior_draws=posterior_draws[work_slice],
                    start_index=work_slice.start,
                    profile=profile,
                    context=context,
                    mass_definition=mass_definition,
                    sigma_table_path=sigma_table_path,
                    candidate_pool_size=candidate_pool_size,
                    random_seed=random_seed,
                )
                for work_slice in slices
            ],
            n_draws=n_draws,
            mass_definition=mass_definition,
        )

    spawn_context = multiprocessing.get_context("spawn")
    chunk_results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=parallelism.worker_processes,
        mp_context=spawn_context,
    ) as executor:
        futures = [
            executor.submit(
                _simulate_ppc_chunk,
                posterior_draws[work_slice],
                work_slice.start,
                profile,
                context,
                mass_definition,
                sigma_table_path,
                candidate_pool_size,
                random_seed,
            )
            for work_slice in slices
        ]
        for future in futures:
            chunk_results.append(future.result())

    return _merge_ppc_chunk_results(chunk_results, n_draws=n_draws, mass_definition=mass_definition)


def _observed_theta_ein_values(observations: list[ObservationRecord]) -> np.ndarray:
    """Return the observed Einstein-radius sample used in PPC comparisons."""

    return np.asarray([observation.einstein_radius_arcsec for observation in observations], dtype=float)


def _aggregate_lens_sigma(observation: ObservationRecord) -> float | None:
    """
    Convert one observation record into a single lens-level sigma value.

    The user explicitly requested that replicated sigma samples ignore
    `num_sigma` and produce one value per sigma lens. The observed side must
    therefore be reduced to the same lens-level contract before we compare
    statistics.
    """

    if observation.num_sigma <= 0:
        return None
    sigma_values = np.asarray(observation.sigma_observed, dtype=float)
    sigma_errors = np.asarray(observation.sigma_error, dtype=float)
    if sigma_values.size == 1:
        return float(sigma_values[0])
    inverse_variance = 1.0 / np.square(np.maximum(sigma_errors, 1.0e-12))
    return float(np.sum(sigma_values * inverse_variance) / np.sum(inverse_variance))


def _observed_sigma_values(observations: list[ObservationRecord]) -> np.ndarray:
    """Return the observed 7-lens sigma sample used in PPC comparisons."""

    values = [value for observation in observations if (value := _aggregate_lens_sigma(observation)) is not None]
    return np.asarray(values, dtype=float)


def _summary_statistics(values: np.ndarray) -> dict[str, float]:
    """Compute the four summary statistics used throughout the PPC workflow."""

    return {
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "p10": float(np.percentile(values, 10.0)),
        "p90": float(np.percentile(values, 90.0)),
    }


def _summarize_observed_against_replicates(observed: dict[str, float], replicated: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """
    Compare observed summary statistics against their replicated distributions.

    The returned payload is intentionally verbose so the summary JSON remains
    self-contained and does not require notebook-side recomputation to recover
    posterior predictive percentiles.
    """

    summary: dict[str, dict[str, float]] = {}
    for stat_name, observed_value in observed.items():
        replicated_values = np.asarray(replicated[stat_name], dtype=float)
        left_percentile = float(np.mean(replicated_values <= observed_value) * 100.0)
        right_percentile = float(np.mean(replicated_values >= observed_value) * 100.0)
        summary[stat_name] = {
            "observed": float(observed_value),
            "replicated_mean": float(np.mean(replicated_values)),
            "replicated_std": float(np.std(replicated_values, ddof=1)),
            "left_percentile": left_percentile,
            "right_percentile": right_percentile,
            "two_sided_extreme_probability": float(2.0 * min(left_percentile, right_percentile) / 100.0),
        }
    return summary


def _compute_histogram_x_limits(values: np.ndarray, observed: float) -> tuple[float, float]:
    """
    Choose a stable x-axis window that favors interpretability over full range.

    Why this helper exists:
    - posterior predictive distributions can contain long tails that make the
      main bulk unreadable if plotted at full extent
    - the user explicitly wants the observed reference line to sit near the
      center of the visible panel
    - we therefore anchor the window on the observed value and size it using a
      robust central interval of the replicated distribution, not the raw min
      and max
    """

    replicated_values = np.asarray(values, dtype=float)
    if replicated_values.size == 0:
        return observed - 0.5, observed + 0.5

    central_low = float(np.percentile(replicated_values, 5.0))
    central_high = float(np.percentile(replicated_values, 95.0))
    robust_half_span = max(abs(observed - central_low), abs(central_high - observed))

    if not np.isfinite(robust_half_span) or robust_half_span <= 0.0:
        robust_scale = float(np.std(replicated_values, ddof=1)) if replicated_values.size > 1 else 0.0
        robust_half_span = max(robust_scale, abs(observed) * 0.1, 1.0e-3)

    padded_half_span = 1.15 * robust_half_span
    return observed - padded_half_span, observed + padded_half_span


def _resolve_histogram_ranges(
    values: np.ndarray,
    observed: float,
    quantity_name: str,
    stat_name: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Split PPC plotting into histogram range vs. display x-limits.

    Why this helper exists:
    - the user wants different plotting rules for only a subset of panels
    - histogram binning and visible axis limits are not the same concern
    - for the two standard-deviation panels, negative histogram bins are
      misleading, but a slightly negative display limit still improves the
      visual balance of the panel

    The function therefore returns two windows:
    - `hist_range`: the numerical support used by `ax.hist(...)`
    - `display_xlim`: the final visible axis range used by `ax.set_xlim(...)`
    """

    display_x_min, display_x_max = _compute_histogram_x_limits(values=values, observed=observed)
    hist_x_min = display_x_min
    hist_x_max = display_x_max

    if stat_name == "std":
        # The two std panels now prioritize readable non-negative support over
        # the earlier "keep the observed marker near the center" heuristic.
        # Standard deviations are physically non-negative, so the histogram
        # always starts at zero and the visible axis keeps only a very small
        # negative padding for breathing room.
        hist_x_min = 0.0

        if quantity_name == "theta_ein":
            # The user requested a fixed theta_ein-std support so that this
            # panel is stable across runs and does not get stretched by tails.
            hist_x_max = 3.0
        else:
            # Sigma std should use a one-sided robust upper envelope instead of
            # a window centered on the observed line. This keeps the main body
            # readable even when the observed statistic sits near the low end.
            replicated_values = np.asarray(values, dtype=float)
            if replicated_values.size == 0:
                upper_anchor = float(observed)
            else:
                upper_anchor = max(float(observed), float(np.percentile(replicated_values, SIGMA_STD_UPPER_PERCENTILE)))
            hist_x_max = SIGMA_STD_UPPER_PADDING_FACTOR * upper_anchor

        if not np.isfinite(hist_x_max) or hist_x_max <= 0.0:
            hist_x_max = max(float(observed), 1.0e-6)

        display_x_min = -STD_PANEL_LEFT_PADDING_FRACTION * hist_x_max
        display_x_max = hist_x_max

    if not np.isfinite(hist_x_min):
        hist_x_min = 0.0 if stat_name == "std" else observed - 0.5
    if not np.isfinite(hist_x_max):
        hist_x_max = observed + 0.5
    if not np.isfinite(display_x_min):
        display_x_min = hist_x_min
    if not np.isfinite(display_x_max):
        display_x_max = hist_x_max

    minimum_positive_width = 1.0e-6
    if hist_x_max <= hist_x_min:
        hist_x_max = hist_x_min + minimum_positive_width
    if display_x_max <= display_x_min:
        display_x_max = display_x_min + minimum_positive_width

    return (hist_x_min, hist_x_max), (display_x_min, display_x_max)


def _summarize_trend_draws(draws: np.ndarray) -> dict[str, np.ndarray]:
    """
    Convert raw posterior trend draws into 16/50/84 percentile bands.

    Using percentiles keeps the figure aligned with the paper's visual
    language and is robust to mildly skewed posterior curve distributions.
    """

    draws = np.asarray(draws, dtype=float)
    summaries = {
        "p16": np.full(draws.shape[1], np.nan, dtype=float),
        "p50": np.full(draws.shape[1], np.nan, dtype=float),
        "p84": np.full(draws.shape[1], np.nan, dtype=float),
    }
    for column_index in range(draws.shape[1]):
        finite_column = draws[np.isfinite(draws[:, column_index]), column_index]
        if finite_column.size == 0:
            continue
        summaries["p16"][column_index] = float(np.percentile(finite_column, 16.0))
        summaries["p50"][column_index] = float(np.percentile(finite_column, 50.0))
        summaries["p84"][column_index] = float(np.percentile(finite_column, 84.0))
    return summaries


def _write_trend_panel(
    ax,
    mass_grid: np.ndarray,
    parent_summary: dict[str, np.ndarray],
    detectable_summary: dict[str, np.ndarray],
    selected_summary: dict[str, np.ndarray],
    y_label: str,
) -> None:
    """
    Render one panel of the Fig. 8-like trend figure.

    The styling intentionally assigns one visual channel per population so the
    three model categories remain distinguishable when their envelopes overlap:
    - parent population: magenta uncertainty band spanning `p16-p84`
    - detectable lenses: black solid boundary lines at `p16` and `p84`
    - SLACS-like selected: blue dashed boundary lines at `p16` and `p84`
    """

    ax.fill_between(
        mass_grid,
        parent_summary["p16"],
        parent_summary["p84"],
        color="#d81b60",
        alpha=0.28,
        label="Parent population",
    )
    ax.plot(
        mass_grid,
        detectable_summary["p16"],
        color="#111111",
        linewidth=2.0,
        linestyle="-",
        label="Detectable lenses",
    )
    ax.plot(
        mass_grid,
        detectable_summary["p84"],
        color="#111111",
        linewidth=2.0,
        linestyle="-",
    )
    ax.plot(
        mass_grid,
        selected_summary["p16"],
        color="#1565c0",
        linewidth=2.0,
        linestyle="--",
        label="SLACS-like selected",
    )
    ax.plot(
        mass_grid,
        selected_summary["p84"],
        color="#1565c0",
        linewidth=2.0,
        linestyle="--",
    )
    ax.set_ylabel(y_label, fontsize=10)
    ax.tick_params(labelsize=8)


def _write_fig8_like_figure(
    figure_path: Path,
    mass_grid: np.ndarray,
    summary_payload: dict[str, dict[str, dict[str, np.ndarray]]],
    mass_definition: MassDefinition,
) -> None:
    """Render the three-panel Fig. 8-like trend figure."""

    figure, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    panel_specs = (
        (mass_definition.label, mass_definition.label),
        ("gamma", "gamma"),
        ("sigma_ap", "sigma_ap [km/s]"),
    )

    for axis, (quantity_name, y_label) in zip(axes, panel_specs, strict=True):
        _write_trend_panel(
            axis,
            mass_grid=mass_grid,
            parent_summary=summary_payload[quantity_name]["parent"],
            detectable_summary=summary_payload[quantity_name]["detectable"],
            selected_summary=summary_payload[quantity_name]["selected"],
            y_label=y_label,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[:3], labels[:3], loc="upper left", fontsize=8, frameon=False)
    axes[-1].set_xlabel(r"log $M_*/M_\odot$", fontsize=10)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def _write_histogram_panel(
    ax,
    values: np.ndarray,
    observed: float,
    title: str,
    left_percentile: float,
    right_percentile: float,
    quantity_name: str,
    stat_name: str,
) -> None:
    """
    Draw one PPC histogram panel with the observed value and tail labels.

    The annotations follow the paper-style PPC presentation the user asked for:
    the replicated histogram provides the reference distribution, the dashed
    line marks the observed statistic, and the two corners report the left and
    right posterior-predictive tail percentages.
    """

    hist_range, display_xlim = _resolve_histogram_ranges(
        values=values,
        observed=observed,
        quantity_name=quantity_name,
        stat_name=stat_name,
    )
    hist_x_min, hist_x_max = hist_range
    display_x_min, display_x_max = display_xlim
    plotted_values = np.asarray(values, dtype=float)
    plotted_values = plotted_values[(plotted_values >= hist_x_min) & (plotted_values <= hist_x_max)]
    if plotted_values.size == 0:
        # Keep the panel renderable even if every replicated draw lies outside
        # the chosen window; this can happen in pathological mismatch cases.
        fallback_value = min(max(observed, hist_x_min), hist_x_max)
        plotted_values = np.asarray([fallback_value], dtype=float)

    ax.hist(
        plotted_values,
        bins=DEFAULT_PPC_HISTOGRAM_BIN_COUNT,
        range=(hist_x_min, hist_x_max),
        color="#d7c3a6",
        edgecolor="#4d3a24",
        linewidth=0.8,
    )
    ax.axvline(observed, color="#8b1e3f", linestyle="--", linewidth=1.6)
    ax.set_xlim(display_x_min, display_x_max)
    ax.set_title(title, fontsize=10)
    ax.text(
        0.03,
        0.95,
        f"L {left_percentile:.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#5d4037",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )
    ax.text(
        0.97,
        0.95,
        f"R {right_percentile:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#5d4037",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )
    ax.tick_params(labelsize=8)


def _write_overview_figure(
    figure_path: Path,
    profile_name: str,
    theta_replicated_stats: dict[str, np.ndarray],
    theta_summary: dict[str, dict[str, float]],
    sigma_replicated_stats: dict[str, np.ndarray],
    sigma_summary: dict[str, dict[str, float]],
) -> None:
    """Render the 8-panel posterior predictive overview figure."""

    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    for axis, stat_name, label in zip(axes[0], SUMMARY_STAT_NAMES, ("median", "std", "p10", "p90"), strict=True):
        _write_histogram_panel(
            axis,
            theta_replicated_stats[stat_name],
            theta_summary[stat_name]["observed"],
            rf"$\theta_{{\mathrm{{ein}}}}$ {label}",
            theta_summary[stat_name]["left_percentile"],
            theta_summary[stat_name]["right_percentile"],
            quantity_name="theta_ein",
            stat_name=stat_name,
        )
    for axis, stat_name, label in zip(axes[1], SUMMARY_STAT_NAMES, ("median", "std", "p10", "p90"), strict=True):
        _write_histogram_panel(
            axis,
            sigma_replicated_stats[stat_name],
            sigma_summary[stat_name]["observed"],
            rf"$\sigma$ {label}",
            sigma_summary[stat_name]["left_percentile"],
            sigma_summary[stat_name]["right_percentile"],
            quantity_name="sigma",
            stat_name=stat_name,
        )
    figure.suptitle(f"Posterior Predictive Check: {profile_name}", fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)


def _materialize_result_dir(output_root_dir: Path, profile_name: str, run_id: str) -> Path:
    """Create the deterministic result directory for one PPC run."""

    result_dir = output_root_dir.expanduser().resolve() / profile_name / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def run_posterior_trends(
    run_dir: str,
    sigma_table_path: str,
    output_root_dir: str,
    n_posterior_draws: int | None = DEFAULT_TREND_POSTERIOR_DRAWS,
    burn_in: str | int = "auto",
    random_seed: int = DEFAULT_RANDOM_SEED + 1,
    n_parent_sample: int = DEFAULT_TREND_PARENT_SAMPLE_SIZE,
    n_mass_bins: int = DEFAULT_TREND_MASS_BIN_COUNT,
    mass_bin_min: float = DEFAULT_TREND_MASS_BIN_MIN,
    mass_bin_max: float = DEFAULT_TREND_MASS_BIN_MAX,
    worker_processes: int | None = None,
    posterior_draw_tail_cap: int = DEFAULT_CANONICAL_POSTERIOR_DRAW_CAP,
) -> PosteriorTrendResult:
    """
    Generate the Fig. 8-like posterior trend figure for one completed run.

    This workflow intentionally mirrors the external binning pattern the user
    asked to reproduce:
    - draw a full parent-population realization for each posterior sample
    - reduce that realization into stellar-mass bins
    - summarize the posterior distribution of those binned curves

    The selection kernel itself remains the current local one. The canonical
    upgrade implemented here is about posterior usage and execution policy:
    - by default the trend figure now consumes the tail-capped full posterior
      chain rather than an arbitrary sampled subset of 256 draws
    - the large canonical workload shares the same process-pool strategy and
      deterministic per-draw seeding used by histogram PPC
    """

    if n_parent_sample < 1:
        raise ValueError("Trend evaluation requires at least one parent-population sample per posterior draw.")
    if n_mass_bins < 1:
        raise ValueError("Trend evaluation requires at least one stellar-mass bin.")
    if mass_bin_max <= mass_bin_min:
        raise ValueError("Trend evaluation requires `mass_bin_max` to be greater than `mass_bin_min`.")

    resolved_run_dir = Path(run_dir).expanduser().resolve()
    config_snapshot_path = resolved_run_dir / "config_snapshot.yaml"
    chain_path = resolved_run_dir / "chain.h5"
    runtime_config = load_runtime_config(config_snapshot_path)
    mass_definition = runtime_config.mass_definition
    trend_quantity_names = _trend_quantity_names(mass_definition)
    burn_in_steps = _resolve_burn_in(burn_in, runtime_config.sampling.warmup)
    selection_rng = np.random.default_rng(random_seed)
    posterior_draws, posterior_draw_mode = _load_posterior_draws(
        chain_path=chain_path,
        burn_in=burn_in_steps,
        rng=selection_rng,
        n_replicates=n_posterior_draws,
        tail_cap=posterior_draw_tail_cap,
    )
    n_posterior_draws_used = int(posterior_draws.shape[0])

    compiled_context, profile_spec, _, _, _, _ = build_compiled_context(runtime_config)
    sigma_table = SigmaUnitTable.from_path(sigma_table_path)
    _assert_sigma_table_matches_run(
        sigma_table=sigma_table,
        profile_name=profile_spec.name,
        mass_definition=mass_definition,
    )

    mass_bin_edges = np.linspace(mass_bin_min, mass_bin_max, n_mass_bins + 1, dtype=float)
    mass_bin_centers = 0.5 * (mass_bin_edges[:-1] + mass_bin_edges[1:])

    parallelism = _resolve_ppc_parallelism(
        runtime_config=runtime_config,
        requested_worker_processes=worker_processes,
        n_draws=n_posterior_draws_used,
    )
    trend_draws, parent_bin_counts_draws, detectable_weight_sums_draws, selected_weight_sums_draws = _run_trend_draws(
        posterior_draws=posterior_draws,
        profile=profile_spec,
        context=compiled_context,
        mass_definition=mass_definition,
        sigma_table_path=str(Path(sigma_table_path).expanduser().resolve()),
        mass_bin_edges=mass_bin_edges,
        n_parent_sample=n_parent_sample,
        random_seed=random_seed,
        parallelism=parallelism,
    )

    trend_summary = {
        quantity_name: {
            category_name: _summarize_trend_draws(trend_draws[quantity_name][category_name])
            for category_name in TREND_CATEGORY_NAMES
        }
        for quantity_name in trend_quantity_names
    }

    result_dir = _materialize_result_dir(Path(output_root_dir), runtime_config.profile.name, resolved_run_dir.name)
    serializable_summary = {
        quantity_name: {
            category_name: {
                key: value.tolist()
                for key, value in trend_summary[quantity_name][category_name].items()
            }
            for category_name in TREND_CATEGORY_NAMES
        }
        for quantity_name in trend_quantity_names
    }
    summary_payload = {
        "run_id": resolved_run_dir.name,
        "profile_name": runtime_config.profile.name,
        "gamma_mode": runtime_config.gamma_model.mode,
        "parameter_order": list(runtime_config.parameter_schema.public_parameter_names),
        "input_run_dir": str(resolved_run_dir),
        "result_dir": str(result_dir),
        "burn_in_applied": burn_in_steps,
        "requested_n_posterior_draws": None if n_posterior_draws is None else int(n_posterior_draws),
        "n_posterior_draws": n_posterior_draws_used,
        "n_posterior_draws_used": n_posterior_draws_used,
        "posterior_draw_mode": posterior_draw_mode,
        "posterior_draw_tail_cap": int(posterior_draw_tail_cap),
        "n_parent_sample": int(n_parent_sample),
        "n_mass_bins": int(n_mass_bins),
        "mass_bin_min": float(mass_bin_min),
        "mass_bin_max": float(mass_bin_max),
        "mass_bin_edges": mass_bin_edges.tolist(),
        "mass_bin_centers": mass_bin_centers.tolist(),
        "generator_mode": "sampled_population_binned",
        "mass_definition": mass_definition_metadata(mass_definition),
        "parallel_strategy": parallelism.strategy,
        "worker_processes": int(parallelism.worker_processes),
        "parallelism": parallelism.to_dict(),
        "quantities": {name: {"label": name} for name in trend_quantity_names},
        "categories": {
            "parent": {"label": "Parent population"},
            "detectable": {"label": "Detectable lenses"},
            "selected": {"label": "SLACS-like selected"},
        },
        "bands": serializable_summary,
    }
    (result_dir / "fig8_like_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    np_save_payload: dict[str, np.ndarray] = {
        "mass_bin_edges": mass_bin_edges,
        "mass_bin_centers": mass_bin_centers,
        "parent_bin_counts_draws": parent_bin_counts_draws,
        "detectable_weight_sums_draws": detectable_weight_sums_draws,
        "selected_weight_sums_draws": selected_weight_sums_draws,
    }
    for quantity_name in trend_quantity_names:
        for category_name in TREND_CATEGORY_NAMES:
            np_save_payload[f"{category_name}_{quantity_name}_draws"] = trend_draws[quantity_name][category_name]
    np.savez(result_dir / "fig8_like_curves.npz", **np_save_payload)

    _write_fig8_like_figure(
        figure_path=result_dir / "fig8_like.png",
        mass_grid=mass_bin_centers,
        summary_payload=trend_summary,
        mass_definition=mass_definition,
    )

    return PosteriorTrendResult(
        run_id=resolved_run_dir.name,
        profile_name=runtime_config.profile.name,
        input_run_dir=resolved_run_dir,
        result_dir=result_dir,
        status="completed",
        burn_in_applied=burn_in_steps,
        n_posterior_draws=n_posterior_draws_used,
        n_mass_bins=int(n_mass_bins),
        sigma_table_path=Path(sigma_table_path).expanduser().resolve(),
        metadata={
            "requested_n_posterior_draws": None if n_posterior_draws is None else int(n_posterior_draws),
            "n_posterior_draws_used": n_posterior_draws_used,
            "posterior_draw_mode": posterior_draw_mode,
            "posterior_draw_tail_cap": int(posterior_draw_tail_cap),
            "gamma_mode": runtime_config.gamma_model.mode,
            "parameter_order": list(runtime_config.parameter_schema.public_parameter_names),
            "n_parent_sample": int(n_parent_sample),
            "mass_bin_min": float(mass_bin_min),
            "mass_bin_max": float(mass_bin_max),
            "n_mass_bins": int(n_mass_bins),
            "generator_mode": "sampled_population_binned",
            "mass_definition": mass_definition_metadata(mass_definition),
            "parallel_strategy": parallelism.strategy,
            "worker_processes": int(parallelism.worker_processes),
            "parallelism": parallelism.to_dict(),
        },
    )


def run_posterior_predictive(
    run_dir: str,
    sigma_table_path: str,
    output_root_dir: str,
    n_replicates: int | None = DEFAULT_N_REPLICATES,
    burn_in: str | int = "auto",
    random_seed: int = DEFAULT_RANDOM_SEED,
    candidate_pool_size: int | None = None,
    worker_processes: int | None = None,
    posterior_draw_tail_cap: int = DEFAULT_CANONICAL_POSTERIOR_DRAW_CAP,
) -> PosteriorPredictiveResult:
    """
    Run the posterior predictive test for one completed inference run.

    Parameters
    ----------
    run_dir:
        Completed inference run directory containing `config_snapshot.yaml` and
        `chain.h5`.
    sigma_table_path:
        Path to the Jeans interpolation table consumed by the sigma PPC step.
    output_root_dir:
        Root directory under which PPC artifacts will be written as
        `<output_root>/<profile>/<run_id>/...`.
    n_replicates:
        Optional number of posterior draws / replicated samples to generate.
        When omitted, the canonical PPC policy uses the tail of the flattened
        post-burn-in chain, capped at `DEFAULT_CANONICAL_POSTERIOR_DRAW_CAP`.
    burn_in:
        Integer number of MCMC steps to discard, or `"auto"` to reuse the
        original run configuration's `warmup` value.
    random_seed:
        Seed controlling posterior-draw sampling, candidate-pool subsampling,
        weighted replicated-lens draws, and sigma noise injection.
    candidate_pool_size:
        Number of candidate lenses to materialize per posterior draw before
        applying weighted sampling. This is intentionally distinct from the
        normalization kernel's Monte Carlo sample count.
    worker_processes:
        Optional process count for posterior-draw chunk parallelism. When
        omitted, PPC resolves a process budget from the machine CPU count and
        the stored runtime `reserve_cores`.
    posterior_draw_tail_cap:
        Maximum number of tail posterior draws used when `n_replicates` is not
        explicitly provided. The monitor uses this to align both profiles.
    """

    resolved_run_dir = Path(run_dir).expanduser().resolve()
    config_snapshot_path = resolved_run_dir / "config_snapshot.yaml"
    chain_path = resolved_run_dir / "chain.h5"
    runtime_config = load_runtime_config(config_snapshot_path)
    mass_definition = runtime_config.mass_definition
    mass_label = mass_definition.label
    burn_in_steps = _resolve_burn_in(burn_in, runtime_config.sampling.warmup)
    selection_rng = np.random.default_rng(random_seed)
    posterior_draws, posterior_draw_mode = _load_posterior_draws(
        chain_path,
        burn_in=burn_in_steps,
        rng=selection_rng,
        n_replicates=n_replicates,
        tail_cap=posterior_draw_tail_cap,
    )
    n_posterior_draws_used = int(posterior_draws.shape[0])

    compiled_context, profile_spec, _, _, _, observations = build_compiled_context(runtime_config)
    sigma_table = SigmaUnitTable.from_path(sigma_table_path)
    _assert_sigma_table_matches_run(
        sigma_table=sigma_table,
        profile_name=profile_spec.name,
        mass_definition=mass_definition,
    )

    effective_candidate_pool_size = _resolve_candidate_pool_size(
        candidate_pool_size=candidate_pool_size,
        base_normals_count=int(compiled_context.base_normals.shape[0]),
    )
    parallelism = _resolve_ppc_parallelism(
        runtime_config=runtime_config,
        requested_worker_processes=worker_processes,
        n_draws=n_posterior_draws_used,
    )
    theta_latent, sigma_latent, theta_replicated_stats, sigma_replicated_stats = _run_ppc_replicates(
        posterior_draws=posterior_draws,
        profile=profile_spec,
        context=compiled_context,
        mass_definition=mass_definition,
        sigma_table_path=str(Path(sigma_table_path).expanduser().resolve()),
        candidate_pool_size=effective_candidate_pool_size,
        random_seed=random_seed,
        parallelism=parallelism,
    )

    theta_observed = _summary_statistics(_observed_theta_ein_values(observations))
    sigma_observed = _summary_statistics(_observed_sigma_values(observations))
    theta_summary = _summarize_observed_against_replicates(theta_observed, theta_replicated_stats)
    sigma_summary = _summarize_observed_against_replicates(sigma_observed, sigma_replicated_stats)

    result_dir = _materialize_result_dir(Path(output_root_dir), runtime_config.profile.name, resolved_run_dir.name)
    summary_payload = {
        "run_id": resolved_run_dir.name,
        "profile_name": runtime_config.profile.name,
        "gamma_mode": runtime_config.gamma_model.mode,
        "parameter_order": list(runtime_config.parameter_schema.public_parameter_names),
        "input_run_dir": str(resolved_run_dir),
        "result_dir": str(result_dir),
        "burn_in_applied": burn_in_steps,
        "requested_n_replicates": None if n_replicates is None else int(n_replicates),
        "n_posterior_draws_used": n_posterior_draws_used,
        "posterior_draw_mode": posterior_draw_mode,
        "mass_definition": mass_definition_metadata(mass_definition),
        "sample_sizes": {"theta_ein": THETA_SAMPLE_SIZE, "sigma": SIGMA_SAMPLE_SIZE},
        "statistics": {"theta_ein": theta_summary, "sigma": sigma_summary},
        "parallelism": parallelism.to_dict(),
    }
    (result_dir / "ppc_summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    manifest_payload = {
        "run_id": resolved_run_dir.name,
        "profile_name": runtime_config.profile.name,
        "gamma_mode": runtime_config.gamma_model.mode,
        "parameter_order": list(runtime_config.parameter_schema.public_parameter_names),
        "config_snapshot_path": str(config_snapshot_path),
        "chain_path": str(chain_path),
        "sigma_table_path": str(Path(sigma_table_path).expanduser().resolve()),
        "candidate_pool_size": effective_candidate_pool_size,
        "normalization_samples": int(runtime_config.integration.normalization_samples),
        "burn_in_applied": burn_in_steps,
        "random_seed": int(random_seed),
        "requested_n_replicates": None if n_replicates is None else int(n_replicates),
        "n_posterior_draws_used": n_posterior_draws_used,
        "posterior_draw_mode": posterior_draw_mode,
        "mass_definition": mass_definition_metadata(mass_definition),
        "parallelism": parallelism.to_dict(),
    }
    (result_dir / "run_manifest.json").write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

    np.savez(
        result_dir / "replicated_statistics.npz",
        theta_sample_theta_ein=theta_latent["theta_ein"],
        theta_sample_gamma=theta_latent["gamma"],
        theta_sample_zd=theta_latent["zd"],
        theta_sample_zs=theta_latent["zs"],
        **{f"theta_sample_{mass_label}": theta_latent[mass_label]},
        theta_sample_re_kpc=theta_latent["re_kpc"],
        theta_sample_n=theta_latent["n"],
        sigma_sample_sigma=sigma_latent["sigma"],
        sigma_sample_theta_ein=sigma_latent["theta_ein"],
        sigma_sample_gamma=sigma_latent["gamma"],
        sigma_sample_zd=sigma_latent["zd"],
        sigma_sample_zs=sigma_latent["zs"],
        **{f"sigma_sample_{mass_label}": sigma_latent[mass_label]},
        sigma_sample_re_kpc=sigma_latent["re_kpc"],
        sigma_sample_n=sigma_latent["n"],
        theta_stat_median=theta_replicated_stats["median"],
        theta_stat_std=theta_replicated_stats["std"],
        theta_stat_p10=theta_replicated_stats["p10"],
        theta_stat_p90=theta_replicated_stats["p90"],
        sigma_stat_median=sigma_replicated_stats["median"],
        sigma_stat_std=sigma_replicated_stats["std"],
        sigma_stat_p10=sigma_replicated_stats["p10"],
        sigma_stat_p90=sigma_replicated_stats["p90"],
    )

    _write_overview_figure(
        result_dir / "ppc_overview.png",
        profile_name=runtime_config.profile.name,
        theta_replicated_stats=theta_replicated_stats,
        theta_summary=theta_summary,
        sigma_replicated_stats=sigma_replicated_stats,
        sigma_summary=sigma_summary,
    )

    result = PosteriorPredictiveResult(
        run_id=resolved_run_dir.name,
        profile_name=runtime_config.profile.name,
        input_run_dir=resolved_run_dir,
        result_dir=result_dir,
        status="completed",
        burn_in_applied=burn_in_steps,
        n_replicates=n_posterior_draws_used,
        sample_sizes={"theta_ein": THETA_SAMPLE_SIZE, "sigma": SIGMA_SAMPLE_SIZE},
        sigma_table_path=Path(sigma_table_path).expanduser().resolve(),
        metadata={
            "requested_n_replicates": None if n_replicates is None else int(n_replicates),
            "n_posterior_draws_used": n_posterior_draws_used,
            "posterior_draw_mode": posterior_draw_mode,
            "candidate_pool_size": effective_candidate_pool_size,
            "normalization_samples": runtime_config.integration.normalization_samples,
            "gamma_mode": runtime_config.gamma_model.mode,
            "parameter_order": list(runtime_config.parameter_schema.public_parameter_names),
            "mass_definition": mass_definition_metadata(mass_definition),
            "parallelism": parallelism.to_dict(),
            "statistics": summary_payload["statistics"],
        },
    )
    return result
