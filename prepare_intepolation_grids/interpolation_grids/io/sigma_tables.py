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
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from interpolation_grids.config import (
    OBSERVATION_FLAVOR_APERTURE_POLICIES,
    EXTERNAL_DATA_DIRECTORY,
    MASS_DEFINITION_LABELS,
    SIGMA_UNIT_BUNDLE_FILENAMES,
    SIGMA_UNIT_BUNDLE_SCHEMA_VERSION,
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
    SUPPORTED_OBSERVATION_FLAVORS,
    sigma_unit_units_for_radius,
)
from interpolation_grids.models import AperturePolicy, SigmaUnitTable
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


def _normalize_observation_flavors(
    observation_flavors: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    """Validate and normalize requested observation flavors while preserving order."""

    if observation_flavors is None:
        return tuple(SUPPORTED_OBSERVATION_FLAVORS)

    normalized_flavors: list[str] = []
    for observation_flavor in observation_flavors:
        normalized_flavor = observation_flavor.strip().lower()
        if normalized_flavor not in SUPPORTED_OBSERVATION_FLAVORS:
            raise ValueError(f"Unsupported observation flavor: {observation_flavor}")
        if normalized_flavor not in normalized_flavors:
            normalized_flavors.append(normalized_flavor)
    return tuple(normalized_flavors)


def _aperture_policy_for_observation_flavor(observation_flavor: str) -> AperturePolicy:
    """Return the canonical aperture policy for one named observation flavor."""

    normalized_flavor = observation_flavor.strip().lower()
    try:
        return OBSERVATION_FLAVOR_APERTURE_POLICIES[normalized_flavor]
    except KeyError as exc:
        raise ValueError(f"Unsupported observation flavor: {observation_flavor}") from exc


def _decode_hdf5_scalar(raw_value: object) -> str:
    """Decode one scalar HDF5 payload into a plain Python string.

    Older sigma-unit tables store `profile_name` as a byte dataset while newer
    metadata may already be plain strings. The repack path should not care
    about that serialization detail, so this helper normalizes both cases.
    """

    if isinstance(raw_value, bytes):
        return raw_value.decode("utf-8")
    return str(raw_value)


def _read_optional_hdf5_float_attribute(handle: h5py.Group | h5py.File, attr_name: str) -> float | None:
    """Return one optional HDF5 float attribute or `None` when it is absent."""

    if attr_name not in handle.attrs:
        return None
    return float(handle.attrs[attr_name])


def _devauc_task(
    task: tuple[int, int, float, float, tuple[float, ...], AperturePolicy | None]
) -> tuple[int, int, np.ndarray]:
    """Worker payload for one `(zd, log_re_kpc)` devauc coordinate."""

    zd_index, log_re_index, zd, log_re_kpc, gamma_axis, aperture_policy = task
    values = compute_sigma_unit_grid(
        profile_name="devauc",
        gamma_grid=np.asarray(gamma_axis, dtype=float),
        zd=zd,
        re_kpc=10.0**log_re_kpc,
        aperture_policy=aperture_policy,
    )
    return zd_index, log_re_index, values


def _sersic_task(
    task: tuple[int, int, int, float, float, float, tuple[float, ...], AperturePolicy | None]
) -> tuple[int, int, int, np.ndarray]:
    """Worker payload for one `(zd, log_re_kpc, n)` sersic coordinate."""

    zd_index, log_re_index, n_index, zd, log_re_kpc, n_value, gamma_axis, aperture_policy = task
    values = compute_sigma_unit_grid(
        profile_name="sersic",
        gamma_grid=np.asarray(gamma_axis, dtype=float),
        zd=zd,
        re_kpc=10.0**log_re_kpc,
        n_value=n_value,
        aperture_policy=aperture_policy,
    )
    return zd_index, log_re_index, n_index, values


def _build_devauc_values(
    gamma_axis: np.ndarray,
    zd_axis: np.ndarray,
    log_re_axis: np.ndarray,
    workers: int,
    aperture_policy: AperturePolicy | None = None,
) -> np.ndarray:
    """Compute the full devauc table, optionally saturating the local CPU."""

    values = np.empty((gamma_axis.size, zd_axis.size, log_re_axis.size), dtype=float)
    tasks = [
        (
            zd_index,
            log_re_index,
            float(zd),
            float(log_re_kpc),
            tuple(float(g) for g in gamma_axis),
            aperture_policy,
        )
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
    aperture_policy: AperturePolicy | None = None,
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
            aperture_policy,
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
    observation_flavor: str = "slit",
    aperture_policy: AperturePolicy | None = None,
) -> SigmaUnitTable:
    """Build one in-memory sigma-unit table by direct Jeans evaluation.

    Parameters are override-friendly so tests can exercise the builder on small
    grids while the production CLI uses the locked default axes from
    `config.py`.
    """

    normalized_profile = profile_name.strip().lower()
    normalized_observation_flavor = observation_flavor.strip().lower()
    gamma_axis = np.asarray(SIGMA_UNIT_GAMMA_AXIS if gamma_axis is None else gamma_axis, dtype=float)
    zd_axis = np.asarray(SIGMA_UNIT_ZD_AXIS if zd_axis is None else zd_axis, dtype=float)
    workers = _resolve_worker_count(workers)
    resolved_aperture_policy = aperture_policy or _aperture_policy_for_observation_flavor(normalized_observation_flavor)

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
            aperture_policy=resolved_aperture_policy,
        )
        return SigmaUnitTable(
            profile_name=normalized_profile,
            mass_definition_label=MASS_DEFINITION_LABELS[float(mass_radius_kpc)],
            mass_radius_kpc=float(mass_radius_kpc),
            gamma_axis=gamma_axis,
            zd_axis=zd_axis,
            log_re_kpc_axis=log_re_axis,
            values=values * np.power(5.0 / float(mass_radius_kpc), 3.0 - gamma_axis)[:, None, None],
            observation_flavor=normalized_observation_flavor,
            aperture_shape=resolved_aperture_policy.shape,
            aperture_width_arcsec=resolved_aperture_policy.width_arcsec,
            aperture_height_arcsec=resolved_aperture_policy.height_arcsec,
            aperture_radius_arcsec=resolved_aperture_policy.radius_arcsec,
            seeing_fwhm_arcsec=resolved_aperture_policy.seeing_fwhm_arcsec,
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
            aperture_policy=resolved_aperture_policy,
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
            observation_flavor=normalized_observation_flavor,
            aperture_shape=resolved_aperture_policy.shape,
            aperture_width_arcsec=resolved_aperture_policy.width_arcsec,
            aperture_height_arcsec=resolved_aperture_policy.height_arcsec,
            aperture_radius_arcsec=resolved_aperture_policy.radius_arcsec,
            seeing_fwhm_arcsec=resolved_aperture_policy.seeing_fwhm_arcsec,
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
            handle.attrs["observation_flavor"] = table.observation_flavor
            handle.attrs["aperture_shape"] = table.aperture_shape
            handle.attrs["seeing_fwhm_arcsec"] = table.seeing_fwhm_arcsec
            if table.aperture_width_arcsec is not None:
                handle.attrs["aperture_width_arcsec"] = table.aperture_width_arcsec
            if table.aperture_height_arcsec is not None:
                handle.attrs["aperture_height_arcsec"] = table.aperture_height_arcsec
            if table.aperture_radius_arcsec is not None:
                handle.attrs["aperture_radius_arcsec"] = table.aperture_radius_arcsec
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


def _write_table_datasets(group_handle: h5py.Group, table: SigmaUnitTable) -> None:
    """Write one sigma-unit table payload into an existing HDF5 group."""

    group_handle.attrs["mass_definition_label"] = table.mass_definition_label
    group_handle.attrs["mass_radius_kpc"] = table.mass_radius_kpc
    group_handle.attrs["units"] = sigma_unit_units_for_radius(table.mass_radius_kpc)
    group_handle.attrs["observation_flavor"] = table.observation_flavor
    group_handle.attrs["aperture_shape"] = table.aperture_shape
    group_handle.attrs["seeing_fwhm_arcsec"] = table.seeing_fwhm_arcsec
    if table.aperture_width_arcsec is not None:
        group_handle.attrs["aperture_width_arcsec"] = table.aperture_width_arcsec
    if table.aperture_height_arcsec is not None:
        group_handle.attrs["aperture_height_arcsec"] = table.aperture_height_arcsec
    if table.aperture_radius_arcsec is not None:
        group_handle.attrs["aperture_radius_arcsec"] = table.aperture_radius_arcsec
    group_handle.create_dataset("gamma_axis", data=np.asarray(table.gamma_axis, dtype=float))
    group_handle.create_dataset("zd_axis", data=np.asarray(table.zd_axis, dtype=float))
    group_handle.create_dataset("log_re_kpc_axis", data=np.asarray(table.log_re_kpc_axis, dtype=float))
    if table.n_axis is not None:
        group_handle.create_dataset("n_axis", data=np.asarray(table.n_axis, dtype=float))
    group_handle.create_dataset("s_unit_grid", data=np.maximum(np.asarray(table.values, dtype=float), 0.0))


def write_sigma_unit_bundle_hdf5(
    tables: dict[tuple[str, str], SigmaUnitTable],
    output_path: Path | str,
) -> Path:
    """Write or refresh one per-profile sigma bundle while preserving untouched leaves."""

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
        if output_path.exists():
            shutil.copy2(output_path, working_path)

        with h5py.File(working_path, "a") as handle:
            if not tables:
                raise ValueError("Sigma bundle writer requires at least one table leaf.")

            profile_names = {table.profile_name for table in tables.values()}
            if len(profile_names) != 1:
                raise ValueError("Sigma bundle writer expects all leaves to share the same profile name.")
            profile_name = next(iter(profile_names))

            handle.attrs["schema_version"] = SIGMA_UNIT_BUNDLE_SCHEMA_VERSION
            handle.attrs["quantity_name"] = SIGMA_UNIT_QUANTITY_NAME
            if "profile_name" in handle:
                del handle["profile_name"]
            handle.create_dataset("profile_name", data=np.bytes_(profile_name))
            for observation_flavor in SUPPORTED_OBSERVATION_FLAVORS:
                handle.require_group(observation_flavor)

            for (observation_flavor, mass_definition_label), table in tables.items():
                flavor_group = handle.require_group(observation_flavor)
                if mass_definition_label in flavor_group:
                    del flavor_group[mass_definition_label]
                leaf_group = flavor_group.create_group(mass_definition_label)
                _write_table_datasets(leaf_group, table)

        working_path.replace(output_path)
    finally:
        working_path.unlink(missing_ok=True)

    return output_path


def read_legacy_sigma_unit_table_hdf5(
    input_path: Path | str,
    *,
    observation_flavor: str = "slit",
    aperture_policy: AperturePolicy | None = None,
) -> SigmaUnitTable:
    """Load one legacy flat sigma-unit HDF5 file into the in-memory table model.

    Why this helper exists:
    - the project historically stored one file per `(profile, mass definition)`
    - the new bundle schema groups those leaves under one per-profile file
    - for the staged migration requested by the user, we need to repack the
      old slit tables without recomputing any Jeans values

    The legacy files do not carry a complete aperture contract, so this loader
    stamps the explicit observation-flavor metadata supplied by the caller.
    That keeps the new bundle self-describing even though the source files are
    older and less explicit.
    """

    input_path = Path(input_path)
    normalized_observation_flavor = observation_flavor.strip().lower()
    resolved_aperture_policy = aperture_policy or _aperture_policy_for_observation_flavor(normalized_observation_flavor)

    with h5py.File(input_path, "r") as handle:
        if "profile_name" not in handle:
            raise ValueError(f"Legacy sigma table is missing 'profile_name': {input_path}")
        if "gamma_axis" not in handle or "zd_axis" not in handle or "log_re_kpc_axis" not in handle:
            raise ValueError(f"Legacy sigma table axes are incomplete: {input_path}")
        if "s_unit_grid" not in handle:
            raise ValueError(f"Legacy sigma table is missing 's_unit_grid': {input_path}")
        if "mass_definition_label" not in handle.attrs or "mass_radius_kpc" not in handle.attrs:
            raise ValueError(f"Legacy sigma table mass-definition metadata are incomplete: {input_path}")

        profile_name = _decode_hdf5_scalar(handle["profile_name"][()])
        mass_definition_label = str(handle.attrs["mass_definition_label"])
        mass_radius_kpc = float(handle.attrs["mass_radius_kpc"])
        n_axis = np.asarray(handle["n_axis"][:], dtype=float) if "n_axis" in handle else None

        return SigmaUnitTable(
            profile_name=profile_name,
            mass_definition_label=mass_definition_label,
            mass_radius_kpc=mass_radius_kpc,
            gamma_axis=np.asarray(handle["gamma_axis"][:], dtype=float),
            zd_axis=np.asarray(handle["zd_axis"][:], dtype=float),
            log_re_kpc_axis=np.asarray(handle["log_re_kpc_axis"][:], dtype=float),
            n_axis=n_axis,
            values=np.asarray(handle["s_unit_grid"][:], dtype=float),
            observation_flavor=normalized_observation_flavor,
            aperture_shape=resolved_aperture_policy.shape,
            aperture_width_arcsec=resolved_aperture_policy.width_arcsec,
            aperture_height_arcsec=resolved_aperture_policy.height_arcsec,
            aperture_radius_arcsec=resolved_aperture_policy.radius_arcsec,
            seeing_fwhm_arcsec=resolved_aperture_policy.seeing_fwhm_arcsec,
        )


def repack_legacy_sigma_unit_hdf5_tables_into_bundles(
    input_directory: Path | str = EXTERNAL_DATA_DIRECTORY,
    output_directory: Path | str = EXTERNAL_DATA_DIRECTORY,
    profiles: tuple[str, ...] | list[str] | None = None,
    mass_radii_kpc: tuple[float, ...] | list[float] | None = None,
) -> dict[str, Path]:
    """Repack legacy flat sigma-unit tables into the new per-profile bundle files.

    This migration path intentionally does not evaluate any new Jeans solves.
    It only reads the existing legacy files, stamps their known slit-aperture
    metadata, and writes new bundle files with the migrated `/slit/<mass>`
    leaves. The `/boss` group is created but left empty so the output schema is
    ready for a later dedicated BOSS build without pretending that those leaves
    already exist.
    """

    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    normalized_profiles = _normalize_profile_names(profiles)
    normalized_mass_radii = _normalize_mass_radii(mass_radii_kpc)
    output_paths: dict[str, Path] = {}
    slit_aperture_policy = _aperture_policy_for_observation_flavor("slit")

    for normalized_profile in normalized_profiles:
        tables_for_profile: dict[tuple[str, str], SigmaUnitTable] = {}
        for mass_radius_kpc in normalized_mass_radii:
            legacy_filename = SIGMA_UNIT_PROFILE_FILENAMES[(normalized_profile, float(mass_radius_kpc))]
            legacy_path = input_directory / legacy_filename
            if not legacy_path.exists():
                raise FileNotFoundError(
                    f"Expected legacy sigma table '{legacy_filename}' for profile '{normalized_profile}' "
                    f"and radius {mass_radius_kpc:g} kpc under {input_directory}."
                )
            table = read_legacy_sigma_unit_table_hdf5(
                legacy_path,
                observation_flavor="slit",
                aperture_policy=slit_aperture_policy,
            )
            tables_for_profile[("slit", table.mass_definition_label)] = table

        output_paths[normalized_profile] = write_sigma_unit_bundle_hdf5(
            tables=tables_for_profile,
            output_path=output_directory / SIGMA_UNIT_BUNDLE_FILENAMES[normalized_profile],
        )

    return output_paths


def build_default_sigma_unit_hdf5_tables(
    output_directory: Path | str = EXTERNAL_DATA_DIRECTORY,
    gamma_axis: np.ndarray | None = None,
    zd_axis: np.ndarray | None = None,
    devauc_log_re_kpc_axis: np.ndarray | None = None,
    sersic_log_re_kpc_axis: np.ndarray | None = None,
    sersic_n_axis: np.ndarray | None = None,
    profiles: tuple[str, ...] | list[str] | None = None,
    observation_flavors: tuple[str, ...] | list[str] | None = None,
    mass_radii_kpc: tuple[float, ...] | list[float] | None = None,
    workers: int | None = None,
) -> dict[str, Path]:
    """Build and write the canonical per-profile sigma bundle HDF5 files."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    normalized_profiles = _normalize_profile_names(profiles)
    normalized_observation_flavors = _normalize_observation_flavors(observation_flavors)
    normalized_mass_radii = _normalize_mass_radii(mass_radii_kpc)
    output_paths: dict[str, Path] = {}

    for normalized_profile in normalized_profiles:
        tables_for_profile: dict[tuple[str, str], SigmaUnitTable] = {}
        for normalized_observation_flavor in normalized_observation_flavors:
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
                    observation_flavor=normalized_observation_flavor,
                    aperture_policy=_aperture_policy_for_observation_flavor(normalized_observation_flavor),
                )
                tables_for_profile[(normalized_observation_flavor, table.mass_definition_label)] = table

        output_paths[normalized_profile] = write_sigma_unit_bundle_hdf5(
            tables=tables_for_profile,
            output_path=output_directory / SIGMA_UNIT_BUNDLE_FILENAMES[normalized_profile],
        )

    return output_paths
