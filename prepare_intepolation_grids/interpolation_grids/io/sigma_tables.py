"""HDF5 builders for PPT-facing sigma-unit interpolation tables.

The historical raw-observation pipeline updates per-galaxy `s2_grid` datasets.
Posterior predictive checks need a different artifact: one large interpolation
table per tracer profile that can be queried for arbitrary replicated lenses.

This module owns that artifact format so the builder logic, schema decisions,
and write-path behavior stay isolated from the older raw-file update flow.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from interpolation_grids.config import (
    EXTERNAL_DATA_DIRECTORY,
    MASS_DEFINITION_LABELS,
    SIGMA_UNIT_DEVAUC_LOG_RE_KPC_AXIS,
    SIGMA_UNIT_GAMMA_AXIS,
    SIGMA_UNIT_PROFILE_FILENAMES,
    SIGMA_UNIT_QUANTITY_NAME,
    SIGMA_UNIT_SCHEMA_VERSION,
    SIGMA_UNIT_SERSIC_LOG_RE_KPC_AXIS,
    SIGMA_UNIT_SERSIC_N_AXIS,
    SIGMA_UNIT_UNITS,
    SIGMA_UNIT_ZD_AXIS,
    SUPPORTED_MASS_RADII_KPC,
    sigma_unit_units_for_radius,
)
from interpolation_grids.models import SigmaUnitTable
from interpolation_grids.physics.jeans import compute_sigma_unit_grid


def _normalize_profile_names(profiles: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Validate and normalize requested profile names while preserving order."""

    if profiles is None:
        return ("devauc", "sersic")

    normalized_profiles: list[str] = []
    for profile_name in profiles:
        normalized_name = profile_name.strip().lower()
        if normalized_name not in {"devauc", "sersic"}:
            raise ValueError(f"Unsupported sigma-unit table profile: {profile_name}")
        if normalized_name not in normalized_profiles:
            normalized_profiles.append(normalized_name)
    return tuple(normalized_profiles)


def _resolve_worker_count(workers: int | None) -> int:
    """Turn an optional worker request into a safe positive process count."""

    if workers is None:
        return max(1, os.cpu_count() or 1)
    return max(1, int(workers))


def _build_multiprocessing_context() -> mp.context.BaseContext | None:
    """Prefer `fork` on macOS because it is materially more robust here.

    The production table build launches long-running CPU-bound workers. On this
    machine, the default `spawn` startup path is more fragile and adds extra
    process bootstrap cost. Using `fork` keeps the implementation small while
    matching the plan's requirement to squeeze as much performance as possible
    out of the local workstation.
    """

    try:
        return mp.get_context("fork")
    except ValueError:
        return None


def _recommended_chunksize(task_count: int, workers: int) -> int:
    """Choose a coarse task chunksize to reduce process-pool overhead."""

    return max(1, task_count // max(1, workers * 8))


def _normalize_mass_radii(mass_radii_kpc: tuple[float, ...] | list[float] | None) -> tuple[float, ...]:
    """Validate and normalize requested enclosed-mass radii while preserving order."""

    if mass_radii_kpc is None:
        return tuple(float(value) for value in SUPPORTED_MASS_RADII_KPC)

    normalized_radii: list[float] = []
    for radius_kpc in mass_radii_kpc:
        normalized_radius = float(radius_kpc)
        if normalized_radius not in SUPPORTED_MASS_RADII_KPC:
            raise ValueError(f"Unsupported sigma-unit mass radius: {radius_kpc}")
        if normalized_radius not in normalized_radii:
            normalized_radii.append(normalized_radius)
    return tuple(normalized_radii)


def _devauc_task(task: tuple[int, int, float, float, tuple[float, ...]]) -> tuple[int, int, np.ndarray]:
    """Worker payload for one `(zd, log_re_kpc)` devauc coordinate."""

    zd_index, log_re_index, zd, log_re_kpc, gamma_axis = task
    values = compute_sigma_unit_grid(
        profile_name="devauc",
        gamma_grid=np.asarray(gamma_axis, dtype=float),
        zd=zd,
        re_kpc=10.0**log_re_kpc,
    )
    return zd_index, log_re_index, values


def _sersic_task(task: tuple[int, int, int, float, float, float, tuple[float, ...]]) -> tuple[int, int, int, np.ndarray]:
    """Worker payload for one `(zd, log_re_kpc, n)` sersic coordinate."""

    zd_index, log_re_index, n_index, zd, log_re_kpc, n_value, gamma_axis = task
    values = compute_sigma_unit_grid(
        profile_name="sersic",
        gamma_grid=np.asarray(gamma_axis, dtype=float),
        zd=zd,
        re_kpc=10.0**log_re_kpc,
        n_value=n_value,
    )
    return zd_index, log_re_index, n_index, values


def _build_devauc_values(gamma_axis: np.ndarray, zd_axis: np.ndarray, log_re_axis: np.ndarray, workers: int) -> np.ndarray:
    """Compute the full devauc table, optionally saturating the local CPU."""

    values = np.empty((gamma_axis.size, zd_axis.size, log_re_axis.size), dtype=float)
    tasks = [
        (zd_index, log_re_index, float(zd), float(log_re_kpc), tuple(float(g) for g in gamma_axis))
        for zd_index, zd in enumerate(zd_axis)
        for log_re_index, log_re_kpc in enumerate(log_re_axis)
    ]

    if workers == 1:
        for task in tasks:
            zd_index, log_re_index, gamma_values = _devauc_task(task)
            values[:, zd_index, log_re_index] = gamma_values
        return values

    executor_kwargs = {"max_workers": workers}
    mp_context = _build_multiprocessing_context()
    if mp_context is not None:
        executor_kwargs["mp_context"] = mp_context

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        for zd_index, log_re_index, gamma_values in executor.map(
            _devauc_task,
            tasks,
            chunksize=_recommended_chunksize(len(tasks), workers),
        ):
            values[:, zd_index, log_re_index] = gamma_values
    return values


def _build_sersic_values(
    gamma_axis: np.ndarray,
    zd_axis: np.ndarray,
    log_re_axis: np.ndarray,
    n_axis: np.ndarray,
    workers: int,
) -> np.ndarray:
    """Compute the full sersic table, optionally saturating the local CPU."""

    values = np.empty((gamma_axis.size, zd_axis.size, log_re_axis.size, n_axis.size), dtype=float)
    tasks = [
        (
            zd_index,
            log_re_index,
            n_index,
            float(zd),
            float(log_re_kpc),
            float(n_value),
            tuple(float(g) for g in gamma_axis),
        )
        for zd_index, zd in enumerate(zd_axis)
        for log_re_index, log_re_kpc in enumerate(log_re_axis)
        for n_index, n_value in enumerate(n_axis)
    ]

    if workers == 1:
        for task in tasks:
            zd_index, log_re_index, n_index, gamma_values = _sersic_task(task)
            values[:, zd_index, log_re_index, n_index] = gamma_values
        return values

    executor_kwargs = {"max_workers": workers}
    mp_context = _build_multiprocessing_context()
    if mp_context is not None:
        executor_kwargs["mp_context"] = mp_context

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        for zd_index, log_re_index, n_index, gamma_values in executor.map(
            _sersic_task,
            tasks,
            chunksize=_recommended_chunksize(len(tasks), workers),
        ):
            values[:, zd_index, log_re_index, n_index] = gamma_values
    return values


def build_sigma_unit_table(
    profile_name: str,
    mass_radius_kpc: float = 5.0,
    gamma_axis: np.ndarray | None = None,
    zd_axis: np.ndarray | None = None,
    log_re_kpc_axis: np.ndarray | None = None,
    n_axis: np.ndarray | None = None,
    workers: int | None = None,
) -> SigmaUnitTable:
    """Build one in-memory sigma-unit table by direct Jeans evaluation.

    Parameters are override-friendly so tests can exercise the builder on small
    grids while the production CLI uses the locked default axes from
    `config.py`.
    """

    normalized_profile = profile_name.strip().lower()
    gamma_axis = np.asarray(SIGMA_UNIT_GAMMA_AXIS if gamma_axis is None else gamma_axis, dtype=float)
    zd_axis = np.asarray(SIGMA_UNIT_ZD_AXIS if zd_axis is None else zd_axis, dtype=float)
    workers = _resolve_worker_count(workers)

    if normalized_profile == "devauc":
        log_re_axis = np.asarray(
            SIGMA_UNIT_DEVAUC_LOG_RE_KPC_AXIS if log_re_kpc_axis is None else log_re_kpc_axis,
            dtype=float,
        )
        values = _build_devauc_values(
            gamma_axis=gamma_axis,
            zd_axis=zd_axis,
            log_re_axis=log_re_axis,
            workers=workers,
        )
        return SigmaUnitTable(
            profile_name=normalized_profile,
            mass_definition_label=MASS_DEFINITION_LABELS[float(mass_radius_kpc)],
            mass_radius_kpc=float(mass_radius_kpc),
            gamma_axis=gamma_axis,
            zd_axis=zd_axis,
            log_re_kpc_axis=log_re_axis,
            values=values * np.power(5.0 / float(mass_radius_kpc), 3.0 - gamma_axis)[:, None, None],
        )

    if normalized_profile == "sersic":
        log_re_axis = np.asarray(
            SIGMA_UNIT_SERSIC_LOG_RE_KPC_AXIS if log_re_kpc_axis is None else log_re_kpc_axis,
            dtype=float,
        )
        n_axis = np.asarray(SIGMA_UNIT_SERSIC_N_AXIS if n_axis is None else n_axis, dtype=float)
        values = _build_sersic_values(
            gamma_axis=gamma_axis,
            zd_axis=zd_axis,
            log_re_axis=log_re_axis,
            n_axis=n_axis,
            workers=workers,
        )
        return SigmaUnitTable(
            profile_name=normalized_profile,
            mass_definition_label=MASS_DEFINITION_LABELS[float(mass_radius_kpc)],
            mass_radius_kpc=float(mass_radius_kpc),
            gamma_axis=gamma_axis,
            zd_axis=zd_axis,
            log_re_kpc_axis=log_re_axis,
            n_axis=n_axis,
            values=values * np.power(5.0 / float(mass_radius_kpc), 3.0 - gamma_axis)[:, None, None, None],
        )

    raise ValueError(f"Unsupported sigma-unit table profile: {profile_name}")


def write_sigma_unit_table_hdf5(table: SigmaUnitTable, output_path: Path | str) -> Path:
    """Write one sigma-unit table to the explicit HDF5 schema used by PPT."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix=f"{output_path.stem}.",
        suffix=output_path.suffix,
        dir=output_path.parent,
        delete=False,
    ) as temp_file:
        working_path = Path(temp_file.name)

    try:
        with h5py.File(working_path, "w") as handle:
            handle.attrs["schema_version"] = SIGMA_UNIT_SCHEMA_VERSION
            handle.attrs["quantity_name"] = SIGMA_UNIT_QUANTITY_NAME
            handle.attrs["mass_definition_label"] = table.mass_definition_label
            handle.attrs["mass_radius_kpc"] = table.mass_radius_kpc
            handle.attrs["units"] = sigma_unit_units_for_radius(table.mass_radius_kpc)
            handle.create_dataset("profile_name", data=np.bytes_(table.profile_name))
            handle.create_dataset("gamma_axis", data=np.asarray(table.gamma_axis, dtype=float))
            handle.create_dataset("zd_axis", data=np.asarray(table.zd_axis, dtype=float))
            handle.create_dataset("log_re_kpc_axis", data=np.asarray(table.log_re_kpc_axis, dtype=float))
            if table.n_axis is not None:
                handle.create_dataset("n_axis", data=np.asarray(table.n_axis, dtype=float))
            # `S_unit` is physically non-negative. We clip tiny negative values
            # caused by numerical noise so downstream PPC loading can remain
            # strict about rejecting non-physical tables.
            handle.create_dataset("s_unit_grid", data=np.maximum(np.asarray(table.values, dtype=float), 0.0))
        working_path.replace(output_path)
    finally:
        working_path.unlink(missing_ok=True)

    return output_path


def build_default_sigma_unit_hdf5_tables(
    output_directory: Path | str = EXTERNAL_DATA_DIRECTORY,
    gamma_axis: np.ndarray | None = None,
    zd_axis: np.ndarray | None = None,
    devauc_log_re_kpc_axis: np.ndarray | None = None,
    sersic_log_re_kpc_axis: np.ndarray | None = None,
    sersic_n_axis: np.ndarray | None = None,
    profiles: tuple[str, ...] | list[str] | None = None,
    mass_radii_kpc: tuple[float, ...] | list[float] | None = None,
    workers: int | None = None,
) -> dict[str, Path]:
    """Build and write the standard devauc and sersic sigma-unit HDF5 tables."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    normalized_profiles = _normalize_profile_names(profiles)
    normalized_mass_radii = _normalize_mass_radii(mass_radii_kpc)
    output_paths: dict[str, Path] = {}

    for normalized_profile in normalized_profiles:
        for mass_radius_kpc in normalized_mass_radii:
            table = build_sigma_unit_table(
                profile_name=normalized_profile,
                mass_radius_kpc=mass_radius_kpc,
                gamma_axis=gamma_axis,
                zd_axis=zd_axis,
                log_re_kpc_axis=(
                    devauc_log_re_kpc_axis if normalized_profile == "devauc" else sersic_log_re_kpc_axis
                ),
                n_axis=None if normalized_profile == "devauc" else sersic_n_axis,
                workers=workers,
            )
            output_key = f"{normalized_profile}_{MASS_DEFINITION_LABELS[float(mass_radius_kpc)]}"
            output_paths[output_key] = write_sigma_unit_table_hdf5(
                table=table,
                output_path=output_directory / SIGMA_UNIT_PROFILE_FILENAMES[(normalized_profile, float(mass_radius_kpc))],
            )

    return output_paths
