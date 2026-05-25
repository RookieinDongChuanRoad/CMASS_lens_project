"""Write canonical inference datasets from current project data products.

The writer is a data-preparation boundary.  It accepts the existing CMASS-style
observation HDF5 and cross-section HDF5 products, normalizes them into the
canonical schema blocks, and writes one inference-ready HDF5 file.  It does not
teach Bayesian inference how to read the result and it does not replace a full
standalone schema validator.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from statistical_sl.data_preparation.config import H_UNITS_V1
from statistical_sl.core.canonical_schema import (
    BLOCK_LENSES,
    BLOCK_LENSING_CROSS_SECTION,
    BLOCK_LENSING_MASS_GRIDS,
    BLOCK_METADATA,
    BLOCK_VELOCITY_DISPERSION_GRIDS,
    CANONICAL_SCHEMA_VERSION,
    CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
    CAPABILITY_LENSING_MASS_GRIDS_V1,
    CAPABILITY_LENS_OBSERVATIONS_V1,
    CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1,
    CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
    CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1,
    DEFAULT_BOUNDARY_POLICY,
)

DEFAULT_THETA_E_AXIS = np.linspace(0.0, 5.0, 256, dtype=float)
MAX_SIGMA_OBSERVATIONS = 2
OBSERVED_APERTURE_SIGMA_DEFINITION = "observed_aperture"


def _decode_scalar_string(value: Any) -> str:
    """Normalize HDF5 string scalars into Python strings."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_scalar_string(value.item())
    return str(value)


def _read_required_attr(group: h5py.Group, aliases: tuple[str, ...]) -> Any:
    """Read the first present attr alias and fail with all attempted names."""

    for alias in aliases:
        if alias in group.attrs:
            return group.attrs[alias]
    raise KeyError(f"{group.name} is missing required attrs: {aliases}")


def _read_optional_attr(group: h5py.Group, aliases: tuple[str, ...], default: Any) -> Any:
    """Read an optional attr alias with one explicit fallback value."""

    for alias in aliases:
        if alias in group.attrs:
            return group.attrs[alias]
    return default


def _optional_contract_string(group: h5py.Group, name: str) -> str | None:
    """Read one optional observation-contract string attr from a raw lens group."""

    if name not in group.attrs:
        return None
    value = _decode_scalar_string(group.attrs[name]).strip().lower()
    return value or None


def _optional_contract_float(group: h5py.Group, name: str) -> float | None:
    """Read one optional observation-contract float attr from a raw lens group."""

    if name not in group.attrs:
        return None
    value = group.attrs[name]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    numeric_value = float(value)
    return None if np.isnan(numeric_value) else numeric_value


def _observation_contract_from_group(group: h5py.Group) -> dict[str, str | float | None] | None:
    """
    Resolve explicit aperture metadata from one raw observation group.

    Missing metadata returns ``None`` so legacy raw files remain readable.  Once
    a group declares `aperture_shape`, the contract must be complete; otherwise
    the canonical writer would produce an HDF5 file that looks self-describing
    but cannot be validated by downstream PPC.
    """

    aperture_shape = _optional_contract_string(group, "aperture_shape")
    if aperture_shape is None:
        return None

    seeing_fwhm_arcsec = _optional_contract_float(group, "seeing_fwhm_arcsec")
    if seeing_fwhm_arcsec is None:
        raise ValueError(f"{group.name} declares aperture_shape but is missing seeing_fwhm_arcsec.")

    aperture_width_arcsec = _optional_contract_float(group, "aperture_width_arcsec")
    aperture_height_arcsec = _optional_contract_float(group, "aperture_height_arcsec")
    aperture_radius_arcsec = _optional_contract_float(group, "aperture_radius_arcsec")
    observation_flavor = _optional_contract_string(group, "observation_flavor")
    sigma_definition = _optional_contract_string(group, "sigma_definition") or OBSERVED_APERTURE_SIGMA_DEFINITION

    if aperture_shape == "rectangular":
        if aperture_width_arcsec is None or aperture_height_arcsec is None:
            raise ValueError(f"{group.name} rectangular aperture metadata requires width and height.")
        if aperture_radius_arcsec is not None:
            raise ValueError(f"{group.name} rectangular aperture metadata must not define a radius.")
        return {
            "observation_flavor": observation_flavor or "slit",
            "sigma_definition": sigma_definition,
            "aperture_shape": aperture_shape,
            "aperture_width_arcsec": aperture_width_arcsec,
            "aperture_height_arcsec": aperture_height_arcsec,
            "aperture_radius_arcsec": None,
            "seeing_fwhm_arcsec": seeing_fwhm_arcsec,
        }

    if aperture_shape == "circular":
        if aperture_radius_arcsec is None:
            raise ValueError(f"{group.name} circular aperture metadata requires aperture_radius_arcsec.")
        if aperture_width_arcsec is not None or aperture_height_arcsec is not None:
            raise ValueError(f"{group.name} circular aperture metadata must not define width or height.")
        return {
            "observation_flavor": observation_flavor or "boss",
            "sigma_definition": sigma_definition,
            "aperture_shape": aperture_shape,
            "aperture_width_arcsec": None,
            "aperture_height_arcsec": None,
            "aperture_radius_arcsec": aperture_radius_arcsec,
            "seeing_fwhm_arcsec": seeing_fwhm_arcsec,
        }

    raise ValueError(f"{group.name} has unsupported aperture_shape={aperture_shape!r}.")


def _contract_key(contract: dict[str, str | float | None]) -> tuple[object, ...]:
    """Return a comparable key for one observation contract dictionary."""

    return (
        contract["observation_flavor"],
        contract["sigma_definition"],
        contract["aperture_shape"],
        contract["aperture_width_arcsec"],
        contract["aperture_height_arcsec"],
        contract["aperture_radius_arcsec"],
        contract["seeing_fwhm_arcsec"],
    )


def _resolve_observation_contract(
    observation_handle: h5py.File,
    lens_ids: tuple[str, ...],
) -> dict[str, str | float | None] | None:
    """
    Resolve a file-level observation contract from per-lens raw metadata.

    PPC validates one sigma table against the whole run.  Mixed aperture
    contracts inside one canonical dataset would make that validation
    ambiguous, so the writer rejects inconsistent explicit metadata early.
    """

    contracts = [
        _observation_contract_from_group(observation_handle[lens_id])
        for lens_id in lens_ids
    ]
    explicit_contracts = [contract for contract in contracts if contract is not None]
    if not explicit_contracts:
        return None
    if len(explicit_contracts) != len(contracts):
        raise ValueError("Observation contract metadata must be present for every lens group or none of them.")

    first_contract = explicit_contracts[0]
    first_key = _contract_key(first_contract)
    for lens_id, contract in zip(lens_ids, explicit_contracts, strict=True):
        if _contract_key(contract) != first_key:
            raise ValueError(f"Observation contract metadata is inconsistent at lens group '{lens_id}'.")
    return first_contract


def _read_sigma_slots(group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-width sigma observation and error arrays."""

    sigma_obs = np.zeros(MAX_SIGMA_OBSERVATIONS, dtype=float)
    sigma_err = np.ones(MAX_SIGMA_OBSERVATIONS, dtype=float)
    raw_sigma = np.atleast_1d(group.attrs.get("sigma", np.asarray([], dtype=float))).astype(float)
    raw_sigma_err = np.atleast_1d(group.attrs.get("sigma_err", np.asarray([], dtype=float))).astype(float)
    count = min(MAX_SIGMA_OBSERVATIONS, raw_sigma.shape[0], raw_sigma_err.shape[0])
    if count:
        sigma_obs[:count] = raw_sigma[:count]
        sigma_err[:count] = raw_sigma_err[:count]
    return sigma_obs, sigma_err


def _mass_group_for_lens(
    lens_group: h5py.Group,
    *,
    mass_definition_label: str,
) -> h5py.Group:
    """Return the selected mass-definition subgroup for one lens."""

    if "mass_definitions" not in lens_group:
        raise KeyError(f"{lens_group.name} is missing mass_definitions.")
    mass_root = lens_group["mass_definitions"]
    if mass_definition_label not in mass_root:
        raise KeyError(f"{lens_group.name} is missing mass_definitions/{mass_definition_label}.")
    return mass_root[mass_definition_label]


def _read_legacy_cross_section_input(cross_section_path: Path, handle: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """Read the legacy one-dimensional CMASS cross-section grid."""

    if "compressed_grids" not in handle:
        raise KeyError(f"{cross_section_path} is missing compressed_grids.")
    compressed = handle["compressed_grids"]
    gamma_name = "gamma_grid" if "gamma_grid" in compressed else "gamma_grids"
    cs_name = "cs_over_theta_ein" if "cs_over_theta_ein" in compressed else "cs_over_theta_ein_grid"
    return (
        np.asarray(compressed[gamma_name][()], dtype=float),
        np.asarray(compressed[cs_name][()], dtype=float),
    )


def _read_cross_section_product(
    cross_section_path: Path,
    *,
    theta_e_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Read either supported cross-section product as canonical theta_E x gamma arrays.

    Legacy CMASS files only store ``cs_over_theta_ein(gamma)``.  They therefore
    need the caller-provided canonical theta axis to build an area grid.
    Sonnenfeld finite-fibre files already store a two-dimensional area grid, so
    the writer preserves the file's own ``tein_grid`` and ignores the optional
    legacy conversion axis.
    """

    with h5py.File(cross_section_path, "r") as handle:
        if {"tein_grid", "gamma_grid", "mufibre3_cs_grid"}.issubset(handle.keys()):
            theta_axis = np.asarray(handle["tein_grid"][()], dtype=float)
            gamma_axis = np.asarray(handle["gamma_grid"][()], dtype=float)
            cross_section_grid = np.asarray(handle["mufibre3_cs_grid"][()], dtype=float)
            expected_shape = (theta_axis.size, gamma_axis.size)
            if cross_section_grid.shape != expected_shape:
                raise ValueError(
                    "mufibre3_cs_grid must have shape "
                    f"{expected_shape}, got {cross_section_grid.shape}."
                )
            return theta_axis, gamma_axis, cross_section_grid, "mufibre3_cs_grid"

        gamma_axis, cs_over_theta = _read_legacy_cross_section_input(cross_section_path, handle)
        cross_section_grid = np.pi * (theta_e_axis[:, None] * cs_over_theta[None, :]) ** 2
        return theta_e_axis, gamma_axis, cross_section_grid, "separable_cs_over_theta_ein"


def _write_string_array(group: h5py.Group, dataset_name: str, values: tuple[str, ...]) -> None:
    """Write a variable-length UTF-8 string dataset."""

    string_dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(dataset_name, data=np.asarray(values, dtype=object), dtype=string_dtype)


def _write_lens_observations(
    output: h5py.File,
    *,
    observation_handle: h5py.File,
    lens_ids: tuple[str, ...],
    profile_name: str,
) -> None:
    """Write the `/lenses` block from normalized observation attrs."""

    lens_group = output.create_group(BLOCK_LENSES)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    lens_group.create_dataset("lens_id", data=np.asarray(lens_ids, dtype=object), dtype=string_dtype)
    n_lens = len(lens_ids)
    z_d = np.zeros(n_lens, dtype=float)
    z_s = np.zeros(n_lens, dtype=float)
    log_mstar_obs = np.zeros(n_lens, dtype=float)
    log_mstar_err = np.zeros(n_lens, dtype=float)
    log_re_obs = np.zeros(n_lens, dtype=float)
    n_obs = np.zeros(n_lens, dtype=float)
    theta_e_obs = np.zeros(n_lens, dtype=float)
    num_sigma = np.zeros(n_lens, dtype=np.int64)
    sigma_obs = np.zeros((n_lens, MAX_SIGMA_OBSERVATIONS), dtype=float)
    sigma_err = np.ones((n_lens, MAX_SIGMA_OBSERVATIONS), dtype=float)

    stellar_mass_aliases = (
        ("logmchab_deV_h2", "logmchab_h2", "logmchab_deV", "logmchab")
        if profile_name == "devauc"
        else ("logmchab_h2", "logmchab", "logmchab_deV_h2", "logmchab_deV")
    )
    size_aliases = (
        (
            "log10_reff_deV_hinv_kpc",
            "log10_reff_deV_kpc",
            "log10_re_hinv_kpc",
            "log10_re_kpc",
            "re_arcsec",
        )
        if profile_name == "devauc"
        else (
            "log10_re_hinv_kpc",
            "log10_re_kpc",
            "log10_reff_deV_hinv_kpc",
            "log10_reff_deV_kpc",
            "re_arcsec",
        )
    )

    for index, lens_id in enumerate(lens_ids):
        source_group = observation_handle[lens_id]
        z_d[index] = float(_read_required_attr(source_group, ("zd",)))
        z_s[index] = float(_read_required_attr(source_group, ("zs",)))
        log_mstar_obs[index] = float(_read_required_attr(source_group, stellar_mass_aliases))
        log_mstar_err[index] = float(
            _read_optional_attr(source_group, ("logmchab_err", "logmchab_deV_err"), 0.0)
        )
        log_re_obs[index] = float(_read_required_attr(source_group, size_aliases))
        n_obs[index] = float(_read_optional_attr(source_group, ("nser",), 4.0))
        theta_e_obs[index] = float(_read_required_attr(source_group, ("rein_arcsec",)))
        num_sigma[index] = int(_read_optional_attr(source_group, ("num_sigma",), 0))
        sigma_obs[index], sigma_err[index] = _read_sigma_slots(source_group)

    lens_group.create_dataset("z_d", data=z_d)
    lens_group.create_dataset("z_s", data=z_s)
    lens_group.create_dataset("log_mstar_obs", data=log_mstar_obs)
    lens_group.create_dataset("log_mstar_err", data=log_mstar_err)
    lens_group.create_dataset("log_re_obs", data=log_re_obs)
    lens_group.create_dataset("n_obs", data=n_obs)
    lens_group.create_dataset("theta_e_obs", data=theta_e_obs)
    lens_group.create_dataset("num_sigma", data=num_sigma)
    lens_group.create_dataset("sigma_obs", data=sigma_obs)
    lens_group.create_dataset("sigma_err", data=sigma_err)


def _write_lensing_mass_grids(
    output: h5py.File,
    *,
    observation_handle: h5py.File,
    lens_ids: tuple[str, ...],
    mass_definition_label: str,
) -> None:
    """Write mass, Jacobian, and per-lens sigma-unit grids."""

    grid_group = output.create_group(BLOCK_LENSING_MASS_GRIDS)
    first_lens_group = observation_handle[lens_ids[0]]
    gamma_grid = np.asarray(first_lens_group["gamma_grid"][()], dtype=float)
    n_lens = len(lens_ids)
    n_gamma = gamma_grid.shape[0]
    per_lens_gamma_grid = np.zeros((n_lens, n_gamma), dtype=float)
    mass_grid = np.zeros((n_lens, n_gamma), dtype=float)
    derivative_grid = np.zeros((n_lens, n_gamma), dtype=float)
    s2_grid = np.zeros((n_lens, n_gamma), dtype=float)
    has_s2 = np.zeros(n_lens, dtype=np.int64)

    for index, lens_id in enumerate(lens_ids):
        source_group = observation_handle[lens_id]
        lens_gamma = np.asarray(source_group["gamma_grid"][()], dtype=float)
        if lens_gamma.shape != gamma_grid.shape or not np.allclose(lens_gamma, gamma_grid):
            raise ValueError("Canonical writer currently requires all lenses to share the same gamma_grid.")
        mass_group = _mass_group_for_lens(source_group, mass_definition_label=mass_definition_label)
        per_lens_gamma_grid[index] = lens_gamma
        mass_grid[index] = np.asarray(mass_group["mass_grid"][()], dtype=float)
        derivative_grid[index] = np.asarray(mass_group["dmass_dthetaein_grid"][()], dtype=float)
        if "s2_grid" in mass_group:
            s2_grid[index] = np.asarray(mass_group["s2_grid"][()], dtype=float)
            has_s2[index] = 1

        num_sigma = int(_read_optional_attr(source_group, ("num_sigma",), 0))
        if num_sigma > 0 and has_s2[index] == 0:
            raise ValueError(f"{lens_id} has num_sigma={num_sigma} but no s2_grid.")

    grid_group.create_dataset("gamma_grid", data=per_lens_gamma_grid)
    grid_group.create_dataset("log_enclosed_mass_grid", data=mass_grid)
    grid_group.create_dataset("dmass_dthetaein_grid", data=derivative_grid)
    grid_group.create_dataset("s2_grid", data=s2_grid)
    grid_group.create_dataset("has_s2", data=has_s2)

    velocity_group = output.create_group(BLOCK_VELOCITY_DISPERSION_GRIDS)
    per_lens_group = velocity_group.create_group("per_lens_s2")
    per_lens_group.create_dataset("s2_grid", data=s2_grid)
    per_lens_group.create_dataset("has_s2", data=has_s2)
    per_lens_group.attrs["source"] = f"/{BLOCK_LENSING_MASS_GRIDS}/s2_grid"


def _write_lensing_cross_section(
    output: h5py.File,
    *,
    cross_section_path: Path,
    theta_e_axis: np.ndarray,
) -> None:
    """Write a unified two-dimensional theta_E x gamma cross-section grid."""

    theta_axis, gamma_axis, cross_section_grid, source = _read_cross_section_product(
        cross_section_path,
        theta_e_axis=theta_e_axis,
    )
    group = output.create_group(BLOCK_LENSING_CROSS_SECTION)
    group.create_dataset("theta_e_axis", data=theta_axis)
    group.create_dataset("gamma_axis", data=gamma_axis)
    group.create_dataset("cross_section_grid", data=cross_section_grid)
    group.attrs["boundary_policy"] = DEFAULT_BOUNDARY_POLICY
    group.attrs["source"] = source


def _copy_optional_sigma_bundle(
    output: h5py.File,
    *,
    sigma_bundle_path: Path | None,
    mass_definition_label: str,
) -> tuple[str, ...]:
    """
    Copy optional sigma-bundle leaves into canonical velocity-dispersion blocks.

    The canonical writer keeps this intentionally conservative: only leaves
    matching the active mass definition are copied, and missing optional groups
    are simply skipped.
    """

    if sigma_bundle_path is None:
        return ()

    capabilities: list[str] = []
    velocity_group = output[BLOCK_VELOCITY_DISPERSION_GRIDS]
    with h5py.File(sigma_bundle_path, "r") as source:
        if "within_re" in source and mass_definition_label in source["within_re"]:
            source.copy(source["within_re"][mass_definition_label], velocity_group, name="fp_within_re")
            capabilities.append(CAPABILITY_VELOCITY_DISPERSION_FP_WITHIN_RE_V1)
    return tuple(capabilities)


def _copy_hdf5_group_payload(
    *,
    source_group: h5py.Group,
    target_parent: h5py.Group,
    target_name: str,
) -> h5py.Group:
    """Copy datasets and attrs from one HDF5 group into a new sibling group."""

    if target_name in target_parent:
        del target_parent[target_name]
    target_group = target_parent.create_group(target_name)
    for attr_name, attr_value in source_group.attrs.items():
        target_group.attrs[attr_name] = attr_value
    for item_name in source_group.keys():
        source_group.copy(source_group[item_name], target_group, name=item_name)
    return target_group


def _validate_population_sigma_unit_group(
    group: h5py.Group,
    *,
    population_sigma_path: Path,
    mass_definition_label: str,
) -> None:
    """Validate the population sigma-unit leaf before copying it into canonical output."""

    required_datasets = ("gamma_axis", "zd_axis", "log_re_kpc_axis", "s_unit_grid")
    missing = [dataset_name for dataset_name in required_datasets if dataset_name not in group]
    if missing:
        raise ValueError(f"{population_sigma_path} is missing population sigma datasets: {missing}.")
    if "mass_definition_label" not in group.attrs:
        raise ValueError(f"{population_sigma_path} is missing mass_definition_label attr.")
    source_mass_label = _decode_scalar_string(group.attrs["mass_definition_label"])
    if source_mass_label != mass_definition_label:
        raise ValueError(
            f"{population_sigma_path} uses mass_definition_label={source_mass_label}; "
            f"expected {mass_definition_label}."
        )

    gamma_axis = np.asarray(group["gamma_axis"][()], dtype=float)
    zd_axis = np.asarray(group["zd_axis"][()], dtype=float)
    log_re_axis = np.asarray(group["log_re_kpc_axis"][()], dtype=float)
    values = np.asarray(group["s_unit_grid"][()], dtype=float)
    expected_shape = (gamma_axis.size, zd_axis.size, log_re_axis.size)
    if values.shape[:3] != expected_shape:
        raise ValueError(
            f"{population_sigma_path} s_unit_grid leading shape must be {expected_shape}; got {values.shape}."
        )


def _copy_optional_population_sigma_unit(
    output: h5py.File,
    *,
    population_sigma_path: Path | None,
    mass_definition_label: str,
) -> tuple[str, ...]:
    """Copy an optional flat population sigma-unit table into the canonical file."""

    if population_sigma_path is None:
        return ()

    velocity_group = output[BLOCK_VELOCITY_DISPERSION_GRIDS]
    with h5py.File(population_sigma_path, "r") as source:
        _validate_population_sigma_unit_group(
            source,
            population_sigma_path=population_sigma_path,
            mass_definition_label=mass_definition_label,
        )
        _copy_hdf5_group_payload(
            source_group=source,
            target_parent=velocity_group,
            target_name="population_sigma_unit",
        )
    return (CAPABILITY_VELOCITY_DISPERSION_POPULATION_SIGMA_UNIT_V1,)


def _write_metadata(
    output: h5py.File,
    *,
    unit_convention: str,
    h_ref: float,
    profile_name: str,
    mass_definition_label: str,
    capabilities: tuple[str, ...],
    observation_contract: dict[str, str | float | None] | None = None,
) -> None:
    """Write schema metadata and capability labels."""

    metadata = output.create_group(BLOCK_METADATA)
    metadata.attrs["schema_version"] = CANONICAL_SCHEMA_VERSION
    metadata.attrs["unit_convention"] = str(unit_convention)
    metadata.attrs["h_ref"] = float(h_ref)
    metadata.attrs["profile_name"] = str(profile_name)
    metadata.attrs["mass_definition_label"] = str(mass_definition_label)
    if observation_contract is not None:
        for key, value in observation_contract.items():
            if value is not None:
                metadata.attrs[key] = value
    _write_string_array(metadata, "capabilities", tuple(dict.fromkeys(capabilities)))


def write_canonical_inference_dataset(
    *,
    observation_path: str | Path,
    cross_section_path: str | Path,
    output_path: str | Path,
    profile_name: str,
    mass_definition_label: str,
    unit_convention: str = H_UNITS_V1,
    h_ref: float = 0.7,
    theta_e_axis: np.ndarray | None = None,
    sigma_bundle_path: str | Path | None = None,
    population_sigma_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Write one canonical inference dataset and return the output path.

    The function writes through a temporary file in the destination directory so
    a failed consistency check cannot leave a partial canonical dataset at the
    requested path.
    """

    output_path = Path(output_path).expanduser().resolve()
    observation_path = Path(observation_path).expanduser().resolve()
    cross_section_path = Path(cross_section_path).expanduser().resolve()
    resolved_sigma_bundle_path = (
        Path(sigma_bundle_path).expanduser().resolve()
        if sigma_bundle_path is not None
        else None
    )
    resolved_population_sigma_path = (
        Path(population_sigma_path).expanduser().resolve()
        if population_sigma_path is not None
        else None
    )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass overwrite=True to replace it.")

    theta_e_values = DEFAULT_THETA_E_AXIS if theta_e_axis is None else np.asarray(theta_e_axis, dtype=float)
    if theta_e_values.ndim != 1 or theta_e_values.size == 0:
        raise ValueError("theta_e_axis must be a non-empty one-dimensional array.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        with h5py.File(observation_path, "r") as observations, h5py.File(temporary_path, "w") as output:
            lens_ids = tuple(sorted(observations.keys()))
            if not lens_ids:
                raise ValueError(f"{observation_path} contains no lens groups.")
            observation_contract = _resolve_observation_contract(observations, lens_ids)

            _write_lens_observations(
                output,
                observation_handle=observations,
                lens_ids=lens_ids,
                profile_name=profile_name,
            )
            _write_lensing_mass_grids(
                output,
                observation_handle=observations,
                lens_ids=lens_ids,
                mass_definition_label=mass_definition_label,
            )
            _write_lensing_cross_section(
                output,
                cross_section_path=cross_section_path,
                theta_e_axis=theta_e_values,
            )
            capabilities = (
                CAPABILITY_LENS_OBSERVATIONS_V1,
                CAPABILITY_LENSING_MASS_GRIDS_V1,
                CAPABILITY_LENSING_CROSS_SECTION_THETA_GAMMA_V1,
                CAPABILITY_VELOCITY_DISPERSION_PER_LENS_S2_V1,
            ) + _copy_optional_sigma_bundle(
                output,
                sigma_bundle_path=resolved_sigma_bundle_path,
                mass_definition_label=mass_definition_label,
            ) + _copy_optional_population_sigma_unit(
                output,
                population_sigma_path=resolved_population_sigma_path,
                mass_definition_label=mass_definition_label,
            )
            _write_metadata(
                output,
                unit_convention=unit_convention,
                h_ref=h_ref,
                profile_name=profile_name,
                mass_definition_label=mass_definition_label,
                capabilities=capabilities,
                observation_contract=observation_contract,
            )

        temporary_path.replace(output_path)
        return output_path
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


__all__ = ["write_canonical_inference_dataset"]
