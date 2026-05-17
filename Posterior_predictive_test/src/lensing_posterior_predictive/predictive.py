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
import shutil
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm.auto import tqdm

from cmass_lens_inference.compiled_context import build_compiled_context
from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.cosmology import FlatLambdaCDM
from cmass_lens_inference.profiles import build_profile_spec
from cmass_lens_inference.mass_definition import (
    H_UNITS_V1,
    LEGACY_FIXED_KPC,
    MassDefinition,
    get_mass_definition,
    mass_definition_metadata,
    sigma_bundle_filename,
)
from cmass_lens_inference.types import (
    ObservationRecord,
    ProfileSpec,
    RuntimeConfig,
)
from cmass_lens_inference.parallel import apply_thread_limits
from .interfaces import DiagnosticsExecution
from .registry import get_predictive_definition
from .legacy import load_legacy_ppc_runtime_config
from .types import (
    Fig8ObservationAnnotationResult,
    PosteriorDiagnosticsResult,
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
DEFAULT_PPC_OUTPUT_ROOT_DIR = Path("/Users/liurongfu/Work/CMASS_lens_project/outputs")
DEFAULT_MONITOR_NOT_BEFORE = datetime(2026, 3, 9, 15, 27, 7, tzinfo=timezone(timedelta(hours=8)))
_MAX_ALLOWED_NEGATIVE_FRACTION = 0.05
_MAX_ALLOWED_NEGATIVE_ABSOLUTE_VALUE = 1.0e-4
SIGMA_UNIT_BUNDLE_SCHEMA_VERSION = "sigma_unit_bundle_hdf5_v2"
SLIT_OBSERVATION_FLAVOR = "slit"
BOSS_OBSERVATION_FLAVOR = "boss"
DEFAULT_SLIT_APERTURE_WIDTH_ARCSEC = 1.6
DEFAULT_SLIT_APERTURE_HEIGHT_ARCSEC = 0.9
DEFAULT_BOSS_APERTURE_RADIUS_ARCSEC = 1.0
DEFAULT_SLIT_SEEING_FWHM_ARCSEC = 0.9
DEFAULT_BOSS_SEEING_FWHM_ARCSEC = 1.5
OBSERVED_APERTURE_SIGMA_DEFINITION = "observed_aperture"
WITHIN_RE_SIGMA_DEFINITION = "within_re"
# Keep the historical name as a slit alias so older call sites and tests
# continue to resolve the canonical 0.9" slit contract.
DEFAULT_SEEING_FWHM_ARCSEC = DEFAULT_SLIT_SEEING_FWHM_ARCSEC
DEFAULT_TREND_POSTERIOR_DRAWS: int | None = None
DEFAULT_TREND_PARENT_SAMPLE_SIZE = 10000
DEFAULT_DIAGNOSTICS_PARENT_SAMPLE_SIZE = DEFAULT_TREND_PARENT_SAMPLE_SIZE
DEFAULT_TREND_MASS_BIN_COUNT = 19
DEFAULT_TREND_MASS_BIN_MIN = 10.15
DEFAULT_TREND_MASS_BIN_MAX = 12.05
TREND_CATEGORY_NAMES = ("parent", "detectable", "selected")
LOG10_2PI = math.log10(2.0 * math.pi)
GAMMA_MODE_DEPENDENT = "dependent"
GAMMA_MODE_INDEPENDENT = "independent"
GAMMA_MODE_SIGMA_STAR_DEPENDENT = "sigma_star_dependent"
_GAMMA_MODE_ALIASES = {
    GAMMA_MODE_DEPENDENT: GAMMA_MODE_DEPENDENT,
    GAMMA_MODE_INDEPENDENT: GAMMA_MODE_INDEPENDENT,
    GAMMA_MODE_SIGMA_STAR_DEPENDENT: GAMMA_MODE_SIGMA_STAR_DEPENDENT,
    "sigma-star-dependent": GAMMA_MODE_SIGMA_STAR_DEPENDENT,
    "sigma_star": GAMMA_MODE_SIGMA_STAR_DEPENDENT,
    "sigma-star": GAMMA_MODE_SIGMA_STAR_DEPENDENT,
}
_GAMMA_MODE_TITLE_LABELS = {
    GAMMA_MODE_DEPENDENT: "dependent",
    GAMMA_MODE_INDEPENDENT: "independent",
    GAMMA_MODE_SIGMA_STAR_DEPENDENT: "Sigma_* dependent",
}


@dataclass(frozen=True)
class ObservedTrendSeries:
    """
    One quantity's observed points for the Fig. 8-style overlay.

    The redraw command needs a small, explicit payload so the plotting code can
    stay agnostic about HDF5 details. Each series therefore carries:
    - `x`: stellar-mass positions for all observed points
    - `y`: observed central values
    - `yerr_lower` / `yerr_upper`: lower and upper vertical uncertainties
    """

    x: np.ndarray
    y: np.ndarray
    yerr_lower: np.ndarray
    yerr_upper: np.ndarray


@dataclass(frozen=True)
class ObservedTrendBand:
    """
    Posterior-summarized observed overlay for model-dependent x-axes.

    Some trend overlays cannot be represented as one fixed x-position per lens.
    `delta_r = logre_kpc - mu_r` is the current example: the scientific
    contract says the overlay should be derived through the same x-axis
    definition used by the model workflow, then summarized into percentile
    bands across posterior draws. This dataclass keeps that contract explicit.
    """

    x: np.ndarray
    p16: np.ndarray
    p50: np.ndarray
    p84: np.ndarray


@dataclass(frozen=True)
class TrendAxisSpec:
    """
    Explicit x-axis contract for one trend artifact.

    Keeping the axis metadata in a typed object avoids scattering naming,
    labeling, and bin-edge policy across the plotting and serialization code.
    """

    name: str
    label: str
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    observed_overlay_mode: str
    figure_label: str | None = None


@dataclass(frozen=True)
class ObservedGammaMeasurements:
    """
    Observed gamma sample aligned with per-lens structural measurements.

    The standalone gamma trend figures need both:
    - plotting-ready gamma central values and uncertainties
    - structural coordinates (`logM*`, `logre_kpc`, `n`) so model-dependent
      x-axes such as `delta_r` can be recomputed consistently
    """

    lens_ids: tuple[str, ...]
    log_mstar: np.ndarray
    log_re_kpc: np.ndarray
    log_sigma_star: np.ndarray
    n_value: np.ndarray
    gamma_mid: np.ndarray
    gamma_yerr_lower: np.ndarray
    gamma_yerr_upper: np.ndarray


ObservedTrendOverlay = ObservedTrendSeries | ObservedTrendBand


@dataclass(frozen=True)
class ObservationContract:
    """
    Explicit aperture and seeing contract read from one raw observation file.

    The PPC and trend workflows must validate that the selected sigma bundle
    matches the raw observations exactly. Carrying the resolved contract as a
    typed object keeps that comparison explicit and avoids re-parsing HDF5
    attrs at every call site.
    """

    observation_flavor: str
    sigma_definition: str
    aperture_shape: str
    aperture_width_arcsec: float | None
    aperture_height_arcsec: float | None
    aperture_radius_arcsec: float | None
    seeing_fwhm_arcsec: float


def _trend_quantity_names(mass_definition: MassDefinition) -> tuple[str, str, str]:
    """Return the public quantity names for one trend run."""

    return (mass_definition.label, "gamma", "sigma_ap")


def _stellar_mass_axis_label(mass_definition: MassDefinition) -> str:
    """Return the plot label for the active stellar-mass coordinate."""

    if mass_definition.unit_convention == H_UNITS_V1:
        return r"log $M_*/(h^{-2} M_\odot)$"
    return r"log $M_*/M_\odot$"


def _effective_radius_axis_label(mass_definition: MassDefinition) -> str:
    """Return the plot label for the active size coordinate."""

    if mass_definition.unit_convention == H_UNITS_V1:
        return r"log $r_e$ [$h^{-1}$ kpc]"
    return r"log $r_e$ [kpc]"


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
    zd_axis: np.ndarray | None
    log_re_kpc_axis: np.ndarray
    values: np.ndarray
    n_axis: np.ndarray | None = None
    sigma_definition: str = OBSERVED_APERTURE_SIGMA_DEFINITION
    bundle_group_name: str = SLIT_OBSERVATION_FLAVOR
    observation_flavor: str | None = SLIT_OBSERVATION_FLAVOR
    aperture_shape: str = "rectangular"
    aperture_width_arcsec: float | None = DEFAULT_SLIT_APERTURE_WIDTH_ARCSEC
    aperture_height_arcsec: float | None = DEFAULT_SLIT_APERTURE_HEIGHT_ARCSEC
    aperture_radius_arcsec: float | None = None
    seeing_fwhm_arcsec: float | None = DEFAULT_SEEING_FWHM_ARCSEC
    bundle_leaf_path: str = "/"
    unit_convention: str = LEGACY_FIXED_KPC
    h_ref: float | None = None

    @classmethod
    def from_path(
        cls,
        table_path: str | Path,
        *,
        mass_definition: MassDefinition | None = None,
        observation_flavor: str | None = None,
        bundle_group: str | None = None,
    ) -> "SigmaUnitTable":
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
            return cls._from_hdf5(
                path,
                mass_definition=mass_definition,
                observation_flavor=observation_flavor,
                bundle_group=bundle_group,
            )
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
            raw_unit_convention = (
                payload["unit_convention"].item()
                if "unit_convention" in payload.files
                else default_mass_definition.unit_convention
            )
            raw_h_ref = payload["h_ref"].item() if "h_ref" in payload.files else None
            n_axis = payload["n_axis"] if "n_axis" in payload.files else None
            return cls(
                profile_name=str(profile_name),
                mass_definition_label=str(raw_mass_label),
                mass_radius_kpc=float(raw_mass_radius),
                units=str(raw_units),
                unit_convention=str(raw_unit_convention),
                h_ref=None if raw_h_ref is None else float(raw_h_ref),
                gamma_axis=np.asarray(payload["gamma_axis"], dtype=float),
                zd_axis=np.asarray(payload["zd_axis"], dtype=float),
                log_re_kpc_axis=np.asarray(payload["log_re_kpc_axis"], dtype=float),
                values=_validate_sigma_unit_grid(np.asarray(payload["s_unit_grid"], dtype=float), source_path=path),
                n_axis=None if n_axis is None else np.asarray(n_axis, dtype=float),
                sigma_definition=OBSERVED_APERTURE_SIGMA_DEFINITION,
                bundle_group_name=SLIT_OBSERVATION_FLAVOR,
                observation_flavor=SLIT_OBSERVATION_FLAVOR,
                aperture_shape="rectangular",
                aperture_width_arcsec=DEFAULT_SLIT_APERTURE_WIDTH_ARCSEC,
                aperture_height_arcsec=DEFAULT_SLIT_APERTURE_HEIGHT_ARCSEC,
                aperture_radius_arcsec=None,
                seeing_fwhm_arcsec=DEFAULT_SEEING_FWHM_ARCSEC,
                bundle_leaf_path="/",
            )

    @classmethod
    def _from_hdf5(
        cls,
        path: Path,
        *,
        mass_definition: MassDefinition | None = None,
        observation_flavor: str | None = None,
        bundle_group: str | None = None,
    ) -> "SigmaUnitTable":
        """Load either the legacy single-table schema or the new bundle schema."""

        with h5py.File(path, "r") as handle:
            schema_version = _decode_hdf5_string(handle.attrs.get("schema_version", ""))
            if schema_version == SIGMA_UNIT_BUNDLE_SCHEMA_VERSION:
                if mass_definition is None or (observation_flavor is None and bundle_group is None):
                    raise ValueError(
                        f"Sigma bundle '{path}' requires `mass_definition` plus either `bundle_group` or `observation_flavor` "
                        "to select one leaf."
                    )
                return cls._from_hdf5_bundle(
                    path,
                    handle,
                    mass_definition=mass_definition,
                    observation_flavor=observation_flavor,
                    bundle_group=bundle_group,
                )
            return cls._from_hdf5_single_table(path, handle)

    @classmethod
    def _from_hdf5_single_table(cls, path: Path, handle: h5py.File) -> "SigmaUnitTable":
        """Load the legacy single-table HDF5 schema."""

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
        raw_unit_convention = handle.attrs.get("unit_convention", default_mass_definition.unit_convention)
        raw_h_ref = _optional_hdf5_float(handle.attrs.get("h_ref"))
        n_axis = np.asarray(handle["n_axis"], dtype=float) if "n_axis" in handle else None
        raw_observation_flavor = handle.attrs.get("observation_flavor", SLIT_OBSERVATION_FLAVOR)
        observation_flavor = _decode_hdf5_string(raw_observation_flavor).strip().lower()
        if observation_flavor == BOSS_OBSERVATION_FLAVOR:
            aperture_shape = _decode_hdf5_string(handle.attrs.get("aperture_shape", "circular"))
            aperture_radius_arcsec = _optional_hdf5_float(handle.attrs.get("aperture_radius_arcsec", DEFAULT_BOSS_APERTURE_RADIUS_ARCSEC))
            aperture_width_arcsec = None
            aperture_height_arcsec = None
        else:
            observation_flavor = SLIT_OBSERVATION_FLAVOR
            aperture_shape = _decode_hdf5_string(handle.attrs.get("aperture_shape", "rectangular"))
            aperture_width_arcsec = _optional_hdf5_float(
                handle.attrs.get("aperture_width_arcsec", DEFAULT_SLIT_APERTURE_WIDTH_ARCSEC)
            )
            aperture_height_arcsec = _optional_hdf5_float(
                handle.attrs.get("aperture_height_arcsec", DEFAULT_SLIT_APERTURE_HEIGHT_ARCSEC)
            )
            aperture_radius_arcsec = _optional_hdf5_float(handle.attrs.get("aperture_radius_arcsec"))
        seeing_fwhm_arcsec = _optional_hdf5_float(handle.attrs.get("seeing_fwhm_arcsec", DEFAULT_SEEING_FWHM_ARCSEC))
        return cls(
            profile_name=profile_name,
            mass_definition_label=_decode_hdf5_string(raw_mass_label),
            mass_radius_kpc=float(raw_mass_radius),
            units=_decode_hdf5_string(raw_units),
            unit_convention=_decode_hdf5_string(raw_unit_convention),
            h_ref=raw_h_ref,
            gamma_axis=np.asarray(handle["gamma_axis"], dtype=float),
            zd_axis=np.asarray(handle["zd_axis"], dtype=float),
            log_re_kpc_axis=np.asarray(handle["log_re_kpc_axis"], dtype=float),
            values=_validate_sigma_unit_grid(np.asarray(handle["s_unit_grid"], dtype=float), source_path=path),
            n_axis=n_axis,
            sigma_definition=OBSERVED_APERTURE_SIGMA_DEFINITION,
            bundle_group_name=observation_flavor,
            observation_flavor=observation_flavor,
            aperture_shape=aperture_shape,
            aperture_width_arcsec=aperture_width_arcsec,
            aperture_height_arcsec=aperture_height_arcsec,
            aperture_radius_arcsec=aperture_radius_arcsec,
            seeing_fwhm_arcsec=float(seeing_fwhm_arcsec or DEFAULT_SEEING_FWHM_ARCSEC),
            bundle_leaf_path="/",
        )

    @classmethod
    def _from_hdf5_bundle(
        cls,
        path: Path,
        handle: h5py.File,
        *,
        mass_definition: MassDefinition,
        observation_flavor: str | None,
        bundle_group: str | None,
    ) -> "SigmaUnitTable":
        """Load one selected leaf from the v2 per-profile bundle schema."""

        selected_bundle_group = bundle_group.strip().lower() if bundle_group is not None else None
        if selected_bundle_group is None:
            if observation_flavor is None:
                raise ValueError(
                    f"Sigma bundle '{path}' requires either `bundle_group` or `observation_flavor`."
                )
            selected_bundle_group = observation_flavor.strip().lower()
        if selected_bundle_group not in handle:
            raise ValueError(
                f"Sigma bundle '{path}' does not contain the bundle group '{selected_bundle_group}'."
            )
        bundle_root_group = handle[selected_bundle_group]
        if mass_definition.label not in bundle_root_group:
            raise ValueError(
                f"Sigma bundle '{path}' does not contain the mass-definition leaf "
                f"'{selected_bundle_group}/{mass_definition.label}'."
            )

        leaf = bundle_root_group[mass_definition.label]
        dataset_names = set(leaf.keys())
        required_dataset_names = {"gamma_axis", "log_re_kpc_axis", "s_unit_grid"}
        if selected_bundle_group != WITHIN_RE_SIGMA_DEFINITION:
            required_dataset_names.add("zd_axis")
        missing = sorted(required_dataset_names.difference(dataset_names))
        if missing:
            raise ValueError(
                f"Sigma bundle leaf '{path}:{selected_bundle_group}/{mass_definition.label}' is missing datasets: {missing}."
            )

        raw_profile_name = handle["profile_name"][()]
        profile_name = _decode_hdf5_string(raw_profile_name)
        raw_mass_label = leaf.attrs.get("mass_definition_label", mass_definition.label)
        raw_mass_radius = leaf.attrs.get("mass_radius_kpc", float(mass_definition.radius_kpc))
        raw_units = leaf.attrs.get("units", mass_definition.sigma_unit_units)
        raw_unit_convention = leaf.attrs.get(
            "unit_convention",
            handle.attrs.get("unit_convention", mass_definition.unit_convention),
        )
        raw_h_ref = _optional_hdf5_float(leaf.attrs.get("h_ref", handle.attrs.get("h_ref")))
        n_axis = np.asarray(leaf["n_axis"], dtype=float) if "n_axis" in leaf else None
        sigma_definition = _decode_hdf5_string(
            leaf.attrs.get(
                "sigma_definition",
                WITHIN_RE_SIGMA_DEFINITION
                if selected_bundle_group == WITHIN_RE_SIGMA_DEFINITION
                else OBSERVED_APERTURE_SIGMA_DEFINITION,
            )
        ).strip().lower()
        if selected_bundle_group == WITHIN_RE_SIGMA_DEFINITION:
            return cls(
                profile_name=profile_name,
                mass_definition_label=_decode_hdf5_string(raw_mass_label),
                mass_radius_kpc=float(raw_mass_radius),
                units=_decode_hdf5_string(raw_units),
                unit_convention=_decode_hdf5_string(raw_unit_convention),
                h_ref=raw_h_ref,
                gamma_axis=np.asarray(leaf["gamma_axis"], dtype=float),
                zd_axis=None,
                log_re_kpc_axis=np.asarray(leaf["log_re_kpc_axis"], dtype=float),
                values=_validate_sigma_unit_grid(np.asarray(leaf["s_unit_grid"], dtype=float), source_path=path),
                n_axis=n_axis,
                sigma_definition=sigma_definition,
                bundle_group_name=selected_bundle_group,
                observation_flavor=None,
                aperture_shape=_decode_hdf5_string(leaf.attrs.get("aperture_shape", "")),
                aperture_width_arcsec=None,
                aperture_height_arcsec=None,
                aperture_radius_arcsec=None,
                seeing_fwhm_arcsec=None,
                bundle_leaf_path=f"/{selected_bundle_group}/{mass_definition.label}",
            )

        normalized_observation_flavor = selected_bundle_group
        return cls(
            profile_name=profile_name,
            mass_definition_label=_decode_hdf5_string(raw_mass_label),
            mass_radius_kpc=float(raw_mass_radius),
            units=_decode_hdf5_string(raw_units),
            unit_convention=_decode_hdf5_string(raw_unit_convention),
            h_ref=raw_h_ref,
            gamma_axis=np.asarray(leaf["gamma_axis"], dtype=float),
            zd_axis=np.asarray(leaf["zd_axis"], dtype=float),
            log_re_kpc_axis=np.asarray(leaf["log_re_kpc_axis"], dtype=float),
            values=_validate_sigma_unit_grid(np.asarray(leaf["s_unit_grid"], dtype=float), source_path=path),
            n_axis=n_axis,
            sigma_definition=sigma_definition,
            bundle_group_name=selected_bundle_group,
            observation_flavor=normalized_observation_flavor,
            aperture_shape=_decode_hdf5_string(leaf.attrs.get("aperture_shape", "")),
            aperture_width_arcsec=_optional_hdf5_float(leaf.attrs.get("aperture_width_arcsec")),
            aperture_height_arcsec=_optional_hdf5_float(leaf.attrs.get("aperture_height_arcsec")),
            aperture_radius_arcsec=_optional_hdf5_float(leaf.attrs.get("aperture_radius_arcsec")),
            seeing_fwhm_arcsec=float(_optional_hdf5_float(leaf.attrs.get("seeing_fwhm_arcsec")) or np.nan),
            bundle_leaf_path=f"/{normalized_observation_flavor}/{mass_definition.label}",
        )

def _assert_sigma_table_matches_run(
    sigma_table: SigmaUnitTable,
    profile_name: str,
    mass_definition: MassDefinition,
    observation_flavor: str,
    observation_contract: ObservationContract | None = None,
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
    if sigma_table.unit_convention != mass_definition.unit_convention:
        raise ValueError(
            f"Sigma table unit_convention '{sigma_table.unit_convention}' does not match "
            f"run unit_convention '{mass_definition.unit_convention}'."
        )
    if mass_definition.unit_convention == H_UNITS_V1:
        if sigma_table.h_ref is None:
            raise ValueError("H-unit sigma table is missing h_ref metadata.")
        # The runtime config currently derives h_ref from H0=70. The PPC layer
        # validates table metadata against the active MassDefinition convention
        # and leaves exact h_ref matching to the inference config/HDF5 loaders,
        # but it still requires h_ref to be physically meaningful.
        if (not np.isfinite(float(sigma_table.h_ref))) or float(sigma_table.h_ref) <= 0.0:
            raise ValueError(f"H-unit sigma table has invalid h_ref={sigma_table.h_ref!r}.")
    normalized_observation_flavor = observation_flavor.strip().lower()
    if sigma_table.observation_flavor != normalized_observation_flavor:
        raise ValueError(
            f"Sigma table observation flavor '{sigma_table.observation_flavor}' does not match "
            f"run observation flavor '{normalized_observation_flavor}'."
        )
    if normalized_observation_flavor == BOSS_OBSERVATION_FLAVOR:
        if sigma_table.aperture_shape != "circular" or not np.isclose(
            sigma_table.aperture_radius_arcsec if sigma_table.aperture_radius_arcsec is not None else np.nan,
            DEFAULT_BOSS_APERTURE_RADIUS_ARCSEC,
        ):
            raise ValueError("Sigma table aperture metadata does not match the BOSS circular-aperture contract.")
    else:
        if sigma_table.aperture_shape != "rectangular" or not np.isclose(
            sigma_table.aperture_width_arcsec if sigma_table.aperture_width_arcsec is not None else np.nan,
            DEFAULT_SLIT_APERTURE_WIDTH_ARCSEC,
        ) or not np.isclose(
            sigma_table.aperture_height_arcsec if sigma_table.aperture_height_arcsec is not None else np.nan,
            DEFAULT_SLIT_APERTURE_HEIGHT_ARCSEC,
        ):
            raise ValueError("Sigma table aperture metadata does not match the slit rectangular-aperture contract.")
    # The seeing contract is flavor-specific: slit bundles are built at 0.9"
    # while the BOSS bundle records the broader 1.5" production value.
    expected_seeing_fwhm_arcsec = (
        DEFAULT_BOSS_SEEING_FWHM_ARCSEC
        if normalized_observation_flavor == BOSS_OBSERVATION_FLAVOR
        else DEFAULT_SLIT_SEEING_FWHM_ARCSEC
    )
    if not np.isclose(
        sigma_table.seeing_fwhm_arcsec,
        expected_seeing_fwhm_arcsec,
    ):
        raise ValueError("Sigma table seeing metadata does not match the expected production value.")

    if observation_contract is None:
        return

    if sigma_table.observation_flavor != observation_contract.observation_flavor:
        raise ValueError("Sigma table observation flavor does not match the raw observation contract.")
    if sigma_table.sigma_definition != observation_contract.sigma_definition:
        raise ValueError("Sigma table sigma definition does not match the observation contract.")
    if sigma_table.aperture_shape != observation_contract.aperture_shape:
        raise ValueError("Sigma table aperture shape does not match the raw observation contract.")
    if not np.isclose(
        sigma_table.seeing_fwhm_arcsec,
        observation_contract.seeing_fwhm_arcsec,
    ):
        raise ValueError("Sigma table seeing metadata does not match the raw observation contract.")

    if observation_contract.observation_flavor == BOSS_OBSERVATION_FLAVOR:
        if not np.isclose(
            sigma_table.aperture_radius_arcsec if sigma_table.aperture_radius_arcsec is not None else np.nan,
            observation_contract.aperture_radius_arcsec if observation_contract.aperture_radius_arcsec is not None else np.nan,
        ):
            raise ValueError("Sigma table aperture radius does not match the raw observation contract.")
        return

    if not np.isclose(
        sigma_table.aperture_width_arcsec if sigma_table.aperture_width_arcsec is not None else np.nan,
        observation_contract.aperture_width_arcsec if observation_contract.aperture_width_arcsec is not None else np.nan,
    ) or not np.isclose(
        sigma_table.aperture_height_arcsec if sigma_table.aperture_height_arcsec is not None else np.nan,
        observation_contract.aperture_height_arcsec if observation_contract.aperture_height_arcsec is not None else np.nan,
    ):
        raise ValueError("Sigma table slit aperture dimensions do not match the raw observation contract.")


def _decode_hdf5_string(raw_value: Any) -> str:
    """Normalize the different scalar string encodings that HDF5 may return."""

    if isinstance(raw_value, bytes):
        return raw_value.decode("utf-8")
    if isinstance(raw_value, np.ndarray) and raw_value.shape == ():
        return _decode_hdf5_string(raw_value.item())
    return str(raw_value)


def _optional_hdf5_float(raw_value: Any) -> float | None:
    """Decode optional numeric HDF5 attrs into Python floats."""

    if raw_value is None:
        return None
    if isinstance(raw_value, np.ndarray) and raw_value.shape == ():
        return _optional_hdf5_float(raw_value.item())
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _load_observation_contract_from_path(observation_path: str | Path) -> ObservationContract:
    """Infer and validate the full raw-observation contract from HDF5 attrs."""

    path = Path(observation_path).expanduser().resolve()
    with h5py.File(path, "r") as handle:
        group_names = sorted(handle.keys())
        if not group_names:
            raise ValueError(f"Observation file '{path}' contains no lens groups.")

        sample_group = handle[group_names[0]]
        aperture_shape = _decode_hdf5_string(sample_group.attrs.get("aperture_shape", "")).strip().lower()
        aperture_radius_arcsec = _optional_hdf5_float(sample_group.attrs.get("aperture_radius_arcsec"))

        is_boss = aperture_shape == "circular" and aperture_radius_arcsec is not None and np.isclose(
            aperture_radius_arcsec,
            DEFAULT_BOSS_APERTURE_RADIUS_ARCSEC,
        )
        if not is_boss:
            return ObservationContract(
                observation_flavor=SLIT_OBSERVATION_FLAVOR,
                sigma_definition=OBSERVED_APERTURE_SIGMA_DEFINITION,
                aperture_shape="rectangular",
                aperture_width_arcsec=DEFAULT_SLIT_APERTURE_WIDTH_ARCSEC,
                aperture_height_arcsec=DEFAULT_SLIT_APERTURE_HEIGHT_ARCSEC,
                aperture_radius_arcsec=None,
                seeing_fwhm_arcsec=DEFAULT_SLIT_SEEING_FWHM_ARCSEC,
            )

        for group_name in group_names:
            group = handle[group_name]
            group_shape = _decode_hdf5_string(group.attrs.get("aperture_shape", "")).strip().lower()
            group_radius_arcsec = _optional_hdf5_float(group.attrs.get("aperture_radius_arcsec"))
            group_seeing_arcsec = _optional_hdf5_float(group.attrs.get("seeing_fwhm_arcsec"))
            if group_shape != "circular" or group_radius_arcsec is None or not np.isclose(
                group_radius_arcsec,
                DEFAULT_BOSS_APERTURE_RADIUS_ARCSEC,
            ):
                raise ValueError(
                    f"BOSS observation file '{path}' has inconsistent circular-aperture metadata in lens '{group_name}'."
                )
            if group_seeing_arcsec is None or not np.isclose(group_seeing_arcsec, DEFAULT_BOSS_SEEING_FWHM_ARCSEC):
                raise ValueError(
                    f"BOSS observation file '{path}' must record seeing_fwhm_arcsec={DEFAULT_BOSS_SEEING_FWHM_ARCSEC:.1f} "
                    f"for every lens group; lens '{group_name}' does not."
                )

        return ObservationContract(
            observation_flavor=BOSS_OBSERVATION_FLAVOR,
            sigma_definition=OBSERVED_APERTURE_SIGMA_DEFINITION,
            aperture_shape="circular",
            aperture_width_arcsec=None,
            aperture_height_arcsec=None,
            aperture_radius_arcsec=DEFAULT_BOSS_APERTURE_RADIUS_ARCSEC,
            seeing_fwhm_arcsec=DEFAULT_BOSS_SEEING_FWHM_ARCSEC,
        )


def _load_observation_contract_from_canonical_dataset_path(
    dataset_path: str | Path,
) -> ObservationContract | None:
    """
    Read an explicit observation contract from canonical dataset metadata.

    New canonical datasets can carry the aperture and seeing contract directly
    under `/metadata`.  This reader intentionally touches only that block so
    PPC can resolve the sigma-table flavor before deciding whether to load the
    full canonical inference schema.  Missing contract attrs return ``None`` to
    preserve compatibility with older canonical files.
    """

    path = Path(dataset_path).expanduser().resolve()
    with h5py.File(path, "r") as handle:
        if "metadata" not in handle:
            return None
        attrs = handle["metadata"].attrs
        observation_flavor = _decode_hdf5_string(attrs.get("observation_flavor", "")).strip().lower()
        aperture_shape = _decode_hdf5_string(attrs.get("aperture_shape", "")).strip().lower()
        sigma_definition = _decode_hdf5_string(
            attrs.get("sigma_definition", OBSERVED_APERTURE_SIGMA_DEFINITION)
        ).strip().lower()
        if not observation_flavor and not aperture_shape:
            return None
        if not observation_flavor or not aperture_shape:
            raise ValueError(
                f"Canonical dataset '{path}' has incomplete observation contract metadata."
            )

        seeing_fwhm_arcsec = _optional_hdf5_float(attrs.get("seeing_fwhm_arcsec"))
        if seeing_fwhm_arcsec is None:
            raise ValueError(
                f"Canonical dataset '{path}' observation contract is missing seeing_fwhm_arcsec."
            )

        aperture_width_arcsec = _optional_hdf5_float(attrs.get("aperture_width_arcsec"))
        aperture_height_arcsec = _optional_hdf5_float(attrs.get("aperture_height_arcsec"))
        aperture_radius_arcsec = _optional_hdf5_float(attrs.get("aperture_radius_arcsec"))

    if aperture_shape == "circular":
        if aperture_radius_arcsec is None:
            raise ValueError(
                f"Canonical dataset '{path}' circular observation contract is missing aperture_radius_arcsec."
            )
        return ObservationContract(
            observation_flavor=observation_flavor,
            sigma_definition=sigma_definition,
            aperture_shape=aperture_shape,
            aperture_width_arcsec=None,
            aperture_height_arcsec=None,
            aperture_radius_arcsec=aperture_radius_arcsec,
            seeing_fwhm_arcsec=seeing_fwhm_arcsec,
        )

    if aperture_shape == "rectangular":
        if aperture_width_arcsec is None or aperture_height_arcsec is None:
            raise ValueError(
                f"Canonical dataset '{path}' rectangular observation contract is missing width or height."
            )
        return ObservationContract(
            observation_flavor=observation_flavor,
            sigma_definition=sigma_definition,
            aperture_shape=aperture_shape,
            aperture_width_arcsec=aperture_width_arcsec,
            aperture_height_arcsec=aperture_height_arcsec,
            aperture_radius_arcsec=None,
            seeing_fwhm_arcsec=seeing_fwhm_arcsec,
        )

    raise ValueError(f"Canonical dataset '{path}' has unsupported aperture_shape={aperture_shape!r}.")


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


def _detect_observation_flavor_from_path(observation_path: str | Path) -> str:
    """Infer the observation flavor from the raw HDF5 aperture metadata."""

    return _load_observation_contract_from_path(observation_path).observation_flavor


def _resolve_external_sigma_table_paths(
    external_dir: str | Path,
) -> dict[str, Path]:
    """Return the monitored per-profile bundle paths."""

    resolved_dir = Path(external_dir).expanduser().resolve()
    return {
        "devauc": resolved_dir / sigma_bundle_filename("devauc"),
        "sersic": resolved_dir / sigma_bundle_filename("sersic"),
    }


def _inspect_sigma_table_candidate(
    table_path: Path,
    expected_profile: str,
    expected_mass_definition: MassDefinition,
    observation_contract: ObservationContract,
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

    table = SigmaUnitTable.from_path(
        table_path,
        mass_definition=expected_mass_definition,
        observation_flavor=observation_contract.observation_flavor,
    )
    _assert_sigma_table_matches_run(
        sigma_table=table,
        profile_name=expected_profile,
        mass_definition=expected_mass_definition,
        observation_flavor=observation_contract.observation_flavor,
        observation_contract=observation_contract,
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
        "observation_flavor": observation_contract.observation_flavor,
        "bundle_leaf_path": table.bundle_leaf_path,
    }
    return {
        "path": table_path.resolve(),
        "mtime": mtime,
        "table": table,
        "axis_summary": axis_summary,
    }


def wait_for_external_sigma_tables_and_run(
    devauc_run_dir: str,
    sersic_run_dir: str,
    output_root_dir: str | Path = DEFAULT_PPC_OUTPUT_ROOT_DIR,
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
    devauc_runtime_config = _load_ppc_runtime_config(Path(devauc_run_dir).expanduser().resolve() / "config_snapshot.yaml")
    sersic_runtime_config = _load_ppc_runtime_config(Path(sersic_run_dir).expanduser().resolve() / "config_snapshot.yaml")
    devauc_observation_contract = _infer_observation_contract_from_runtime_config(devauc_runtime_config)
    sersic_observation_contract = _infer_observation_contract_from_runtime_config(sersic_runtime_config)
    table_paths = _resolve_external_sigma_table_paths(external_dir)
    started_at = time.monotonic()
    last_error_message = "monitor has not inspected any candidate tables yet"

    while True:
        try:
            devauc_candidate = _inspect_sigma_table_candidate(
                table_path=table_paths["devauc"],
                expected_profile="devauc",
                expected_mass_definition=devauc_runtime_config.mass_definition,
                observation_contract=devauc_observation_contract,
                not_before=resolved_not_before,
            )
            sersic_candidate = _inspect_sigma_table_candidate(
                table_path=table_paths["sersic"],
                expected_profile="sersic",
                expected_mass_definition=sersic_runtime_config.mass_definition,
                observation_contract=sersic_observation_contract,
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
        devauc_available = _load_posterior_draws(
            chain_path=Path(devauc_run_dir).expanduser().resolve() / "chain.h5",
            burn_in=devauc_burn_in,
            rng=np.random.default_rng(random_seed),
            n_replicates=None,
            tail_cap=DEFAULT_CANONICAL_POSTERIOR_DRAW_CAP,
            parameter_names=devauc_runtime_config.parameter_schema.internal_parameter_names,
        )[0].shape[0]
        sersic_available = _load_posterior_draws(
            chain_path=Path(sersic_run_dir).expanduser().resolve() / "chain.h5",
            burn_in=sersic_burn_in,
            rng=np.random.default_rng(random_seed),
            n_replicates=None,
            tail_cap=DEFAULT_CANONICAL_POSTERIOR_DRAW_CAP,
            parameter_names=sersic_runtime_config.parameter_schema.internal_parameter_names,
        )[0].shape[0]
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


def _load_ppc_runtime_config(config_path: str | Path) -> RuntimeConfig:
    """
    Load a runtime config for PPC, including archived pre-registry snapshots.

    The first attempt always uses the production parser.  Only when the file is
    clearly an old PPC-compatible raw-data snapshot do we fall back to the local
    compatibility bridge above.
    """

    path = Path(config_path).expanduser().resolve()
    try:
        return load_runtime_config(path)
    except (KeyError, ValueError) as exc:
        raw_data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw_data, dict):
            raise
        legacy_markers = (
            "mass_definition" in raw_data
            and isinstance(raw_data.get("data"), dict)
            and "observation_path" in raw_data["data"]
            and "cross_section_path" in raw_data["data"]
        )
        if not legacy_markers:
            raise
        try:
            return load_legacy_ppc_runtime_config(path, raw_data)
        except Exception as legacy_exc:
            raise ValueError(
                f"Config '{path}' is not accepted by production parsing and also failed PPC legacy parsing: "
                f"{legacy_exc}"
            ) from exc


def _runtime_gamma_mode(runtime_config: RuntimeConfig) -> str:
    """Return the active gamma mode through the current generic schema bridge."""

    return runtime_config.parameter_schema.gamma_mode


def _observation_contract_from_flavor(observation_flavor: str) -> ObservationContract:
    """Return the explicit PPC aperture contract for one known observation flavor."""

    normalized = observation_flavor.strip().lower()
    if normalized == BOSS_OBSERVATION_FLAVOR:
        return ObservationContract(
            observation_flavor=BOSS_OBSERVATION_FLAVOR,
            sigma_definition=OBSERVED_APERTURE_SIGMA_DEFINITION,
            aperture_shape="circular",
            aperture_width_arcsec=None,
            aperture_height_arcsec=None,
            aperture_radius_arcsec=DEFAULT_BOSS_APERTURE_RADIUS_ARCSEC,
            seeing_fwhm_arcsec=DEFAULT_BOSS_SEEING_FWHM_ARCSEC,
        )
    return ObservationContract(
        observation_flavor=SLIT_OBSERVATION_FLAVOR,
        sigma_definition=OBSERVED_APERTURE_SIGMA_DEFINITION,
        aperture_shape="rectangular",
        aperture_width_arcsec=DEFAULT_SLIT_APERTURE_WIDTH_ARCSEC,
        aperture_height_arcsec=DEFAULT_SLIT_APERTURE_HEIGHT_ARCSEC,
        aperture_radius_arcsec=None,
        seeing_fwhm_arcsec=DEFAULT_SLIT_SEEING_FWHM_ARCSEC,
    )


def _infer_observation_contract_from_runtime_config(runtime_config: RuntimeConfig) -> ObservationContract:
    """
    Resolve the observed-aperture contract for legacy and canonical PPC inputs.

    Legacy run snapshots carry a raw observation HDF5 path whose group attrs are
    authoritative.  Current b283 inference configs are canonical-only, so PPC
    falls back to the dataset filename convention used by the prepared CMASS
    products (`*_boss_*` versus slit/default) and then validates the selected
    sigma bundle against that explicit flavor contract.
    """

    if runtime_config.data.observation_path is not None:
        return _load_observation_contract_from_path(runtime_config.data.observation_path)
    dataset_path = runtime_config.data.inference_dataset_path
    if dataset_path is None:
        raise ValueError("PPC diagnostics require either observation_path or inference_dataset_path.")
    canonical_contract = _load_observation_contract_from_canonical_dataset_path(dataset_path)
    if canonical_contract is not None:
        return canonical_contract
    filename = dataset_path.name.lower()
    return _observation_contract_from_flavor(
        BOSS_OBSERVATION_FLAVOR if "_boss_" in filename else SLIT_OBSERVATION_FLAVOR
    )


def _build_ppc_context(runtime_config: RuntimeConfig):
    """
    Build the numerical context used by PPC diagnostics for old and new configs.

    Raw-path snapshots use the historical `build_compiled_context` reader.
    Canonical-only configs dispatch through the predictive registry so the
    generic PPT workflow no longer owns model-name branches.  The active model
    definition is responsible for building whatever context its diagnostics
    need, while this function preserves the old tuple shape consumed by the
    existing artifact writers.
    """

    if runtime_config.data.observation_path is not None and runtime_config.data.cross_section_path is not None:
        return build_compiled_context(runtime_config)
    predictive_definition = get_predictive_definition(runtime_config.model.name)
    return predictive_definition.build_context(runtime_config)


def _predictive_metadata_payload(predictive_definition) -> dict[str, Any]:
    """
    Serialize model-owned predictive contract metadata into artifact payloads.

    The old artifacts exposed only `backend`, which was enough while every PPT
    run was implicitly CMASS.  Model-aware diagnostics need the model name,
    schema version, supported diagnostic family, and external input contract in
    every summary/manifest so downstream readers can reject incompatible
    payloads without guessing from filenames.
    """

    return {
        "model_name": predictive_definition.model_name,
        "predictive_backend": predictive_definition.backend,
        "predictive_schema_version": predictive_definition.artifact_schema_version,
        "supported_diagnostics": list(predictive_definition.supported_diagnostics),
        "required_external_inputs": list(predictive_definition.required_external_inputs),
    }


def _resolve_sigma_table_path_for_definition(
    predictive_definition,
    sigma_table_path: str | Path | None,
) -> Path | None:
    """
    Resolve the optional sigma-table path according to the active model contract.

    `--sigma-table` is no longer a CLI-global required argument.  CMASS still
    declares it as a required external input, so missing values fail here with a
    model-specific message.  Future models that do not require this table can
    leave the argument unset and provide their own predictive input path.
    """

    requires_sigma_table = "sigma_table" in predictive_definition.required_external_inputs
    if sigma_table_path is None:
        if requires_sigma_table:
            raise ValueError(
                f"Model '{predictive_definition.model_name}' predictive diagnostics require "
                "external input 'sigma_table'. Pass --sigma-table or provide sigma_table_path."
            )
        return None
    return Path(sigma_table_path).expanduser().resolve()


def _empty_observed_series() -> ObservedTrendSeries:
    """Return an empty observed-overlay series for raw-only quantities."""

    return ObservedTrendSeries(
        x=np.asarray([], dtype=float),
        y=np.asarray([], dtype=float),
        yerr_lower=np.asarray([], dtype=float),
        yerr_upper=np.asarray([], dtype=float),
    )


def _load_observed_trend_points_for_runtime(
    runtime_config: RuntimeConfig,
    observations: list[ObservationRecord],
    mass_definition: MassDefinition,
) -> dict[str, ObservedTrendSeries]:
    """
    Load Fig. 8 observed overlays from raw attrs or canonical observations.

    Canonical datasets do not currently store the flat-prior `m5/gamma`
    observed-summary attrs used by the raw overlay.  In that case PPC keeps the
    scientifically available sigma overlay and leaves mass/gamma overlays empty
    rather than inventing values.
    """

    if runtime_config.data.observation_path is not None:
        return _load_observed_trend_points(
            observation_path=runtime_config.data.observation_path,
            profile_name=runtime_config.profile.name,
            mass_definition=mass_definition,
        )

    sigma_x_values: list[float] = []
    sigma_y_values: list[float] = []
    sigma_errors: list[float] = []
    for observation in observations:
        if observation.num_sigma <= 0:
            continue
        for sigma_value, sigma_error in zip(observation.sigma_observed, observation.sigma_error, strict=True):
            sigma_x_values.append(float(observation.log_stellar_mass_obs))
            sigma_y_values.append(float(sigma_value))
            sigma_errors.append(float(sigma_error))

    return {
        mass_definition.label: _empty_observed_series(),
        "gamma": _empty_observed_series(),
        "sigma_ap": ObservedTrendSeries(
            x=np.asarray(sigma_x_values, dtype=float),
            y=np.asarray(sigma_y_values, dtype=float),
            yerr_lower=np.asarray(sigma_errors, dtype=float),
            yerr_upper=np.asarray(sigma_errors, dtype=float),
        ),
    }


def _load_observed_gamma_measurements_for_runtime(
    runtime_config: RuntimeConfig,
    observations: list[ObservationRecord],
    cosmology: FlatLambdaCDM,
    mass_definition: MassDefinition,
) -> ObservedGammaMeasurements:
    """
    Load structural coordinates for standalone gamma trends.

    Raw inputs use the full observed-gamma summary attrs.  Canonical inputs only
    provide structure and sigma observations, so gamma y-values are left as
    NaN; the structural x-axes still remain well-defined and useful for trend
    binning.
    """

    if runtime_config.data.observation_path is not None:
        return _load_observed_gamma_measurements(
            observation_path=runtime_config.data.observation_path,
            profile_name=runtime_config.profile.name,
            observations=observations,
            cosmology=cosmology,
            mass_definition=mass_definition,
        )

    profile_spec = build_profile_spec(runtime_config.profile.name)
    selected_observations = [observation for observation in observations if observation.num_sigma > 0]
    lens_ids = tuple(observation.lens_id for observation in selected_observations)
    log_mstar = np.asarray([observation.log_stellar_mass_obs for observation in selected_observations], dtype=float)
    log_re_kpc = np.asarray(
        [
            (
                observation.log_effective_radius_obs
                if observation.log_effective_radius_obs is not None
                else math.log10(max(observation.effective_radius_arcsec * cosmology.kpc_per_arcsec(observation.z_d), 1.0e-12))
            )
            for observation in selected_observations
        ],
        dtype=float,
    )
    n_value = np.asarray(
        [
            float(profile_spec.fixed_n if profile_spec.fixed_n is not None else observation.n_observed)
            for observation in selected_observations
        ],
        dtype=float,
    )
    return ObservedGammaMeasurements(
        lens_ids=lens_ids,
        log_mstar=log_mstar,
        log_re_kpc=log_re_kpc,
        log_sigma_star=_compute_log_sigma_star(log_mstar, log_re_kpc),
        n_value=n_value,
        gamma_mid=np.full(log_mstar.shape, np.nan, dtype=float),
        gamma_yerr_lower=np.zeros(log_mstar.shape, dtype=float),
        gamma_yerr_upper=np.zeros(log_mstar.shape, dtype=float),
    )


def _build_padded_bin_edges(
    values: np.ndarray,
    n_bins: int,
    padding_fraction: float,
    minimum_padding: float,
) -> np.ndarray:
    """
    Build deterministic bin edges from an observed support range plus padding.

    The added trend figures should not depend on ad-hoc manual limits. This
    helper turns a finite reference sample into a reproducible bin contract
    while guarding the degenerate single-valued case with a minimum padding.
    """

    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("Cannot build trend bins from an empty or non-finite value set.")

    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))
    span = max_value - min_value
    padding = max(float(span) * float(padding_fraction), float(minimum_padding))
    if span <= 0.0:
        min_value -= padding
        max_value += padding
    else:
        min_value -= padding
        max_value += padding
    return np.linspace(min_value, max_value, int(n_bins) + 1, dtype=float)


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


def _reorder_numpyro_parameter_axis(
    samples_by_chain: np.ndarray,
    stored_parameter_names: np.ndarray,
    requested_parameter_names: tuple[str, ...] | None,
    *,
    artifact_path: Path,
) -> np.ndarray:
    """
    Return NumPyro samples in the run config's parameter order.

    NumPyro artifacts are self-describing, while PPC/trend code expects plain
    theta matrices in `RuntimeConfig.parameter_schema.internal_parameter_names`
    order. Reordering at the loader boundary keeps all downstream predictive
    code unchanged and prevents silent parameter swaps.
    """

    samples = np.asarray(samples_by_chain, dtype=float)
    if samples.ndim != 3:
        raise ValueError(f"Posterior artifact '{artifact_path}' must store samples with shape (chain, draw, parameter).")
    if requested_parameter_names is None:
        return samples

    stored_names = [str(name) for name in np.asarray(stored_parameter_names).tolist()]
    missing = [name for name in requested_parameter_names if name not in stored_names]
    if missing:
        raise ValueError(
            f"Posterior artifact '{artifact_path}' is missing parameters required by the run config: {missing}."
        )
    order = [stored_names.index(name) for name in requested_parameter_names]
    return samples[:, :, order]


def _load_flattened_numpyro_samples_npz(
    samples_path: Path,
    *,
    parameter_names: tuple[str, ...] | None,
) -> np.ndarray:
    """Load flattened posterior draws from the compact NumPyro `samples.npz` artifact."""

    with np.load(samples_path) as payload:
        if "samples_by_chain" in payload:
            samples_by_chain = np.asarray(payload["samples_by_chain"], dtype=float)
        elif "flat_samples" in payload:
            flat_samples = np.asarray(payload["flat_samples"], dtype=float)
            if flat_samples.ndim != 2:
                raise ValueError(f"Posterior artifact '{samples_path}' has invalid `flat_samples` shape.")
            samples_by_chain = flat_samples[None, :, :]
        else:
            raise ValueError(f"Posterior artifact '{samples_path}' is missing `samples_by_chain` or `flat_samples`.")
        stored_parameter_names = payload.get("parameter_names", np.asarray([], dtype="U"))

    ordered = _reorder_numpyro_parameter_axis(
        samples_by_chain,
        stored_parameter_names,
        parameter_names,
        artifact_path=samples_path,
    )
    return ordered.reshape(-1, ordered.shape[-1])


def _load_flattened_numpyro_posterior_nc(
    posterior_path: Path,
    *,
    parameter_names: tuple[str, ...] | None,
) -> np.ndarray:
    """Load flattened posterior draws from the ArviZ `posterior.nc` artifact."""

    import arviz as az

    inference_data = az.from_netcdf(posterior_path)
    posterior = inference_data.posterior
    stored_parameter_names = tuple(str(name) for name in posterior.data_vars)
    ordered_parameter_names = parameter_names or stored_parameter_names
    missing = [name for name in ordered_parameter_names if name not in posterior.data_vars]
    if missing:
        raise ValueError(
            f"Posterior artifact '{posterior_path}' is missing parameters required by the run config: {missing}."
        )
    samples_by_chain = np.stack(
        [np.asarray(posterior[name].values, dtype=float) for name in ordered_parameter_names],
        axis=-1,
    )
    if samples_by_chain.ndim != 3:
        raise ValueError(f"Posterior artifact '{posterior_path}' must store scalar parameter sites by chain and draw.")
    return samples_by_chain.reshape(-1, samples_by_chain.shape[-1])


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
    parameter_names: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, str, str]:
    """
    Load posterior draws from NumPyro artifacts or legacy emcee chains.

    New inference runs write `samples.npz` and `posterior.nc` after warmup, so
    burn-in is only applied to legacy `chain.h5` inputs. The caller receives
    the selected draws, the selection mode, and the artifact name for metadata.
    """

    run_dir = chain_path.parent
    samples_path = run_dir / "samples.npz"
    posterior_path = run_dir / "posterior.nc"
    if samples_path.exists():
        flattened_draws = _load_flattened_numpyro_samples_npz(
            samples_path,
            parameter_names=parameter_names,
        )
        artifact_name = "samples.npz"
    elif posterior_path.exists():
        flattened_draws = _load_flattened_numpyro_posterior_nc(
            posterior_path,
            parameter_names=parameter_names,
        )
        artifact_name = "posterior.nc"
    elif chain_path.exists():
        flattened_draws = _load_flattened_posterior_chain(chain_path=chain_path, burn_in=burn_in)
        artifact_name = "chain.h5"
    else:
        raise FileNotFoundError(
            f"Run directory '{run_dir}' does not contain samples.npz, posterior.nc, or legacy chain.h5."
        )

    selected, mode = _select_posterior_draws(
        flattened_chain=flattened_draws,
        n_replicates=n_replicates,
        rng=rng,
        tail_cap=tail_cap,
    )
    return selected, mode, artifact_name


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


def _compute_observed_delta_r(
    log_mstar: np.ndarray,
    n_value: np.ndarray,
    log_re_kpc: np.ndarray,
    profile: ProfileSpec,
    theta: np.ndarray | None = None,
    stellar_mass_pivot: float = 11.4,
    mu_r0: float | None = None,
) -> np.ndarray:
    """Compute observed ``delta_r`` with the same size relation used by the Numba kernel."""

    del theta
    resolved_mu_r0 = profile.mu_r0 if mu_r0 is None else float(mu_r0)
    log_mstar_array = np.asarray(log_mstar, dtype=float)
    n_value_array = np.asarray(n_value, dtype=float)
    mu_r = resolved_mu_r0 + profile.beta_r * (log_mstar_array - float(stellar_mass_pivot))
    if profile.nu_r is not None:
        mu_r = mu_r + profile.nu_r * (np.log10(np.maximum(n_value_array, 1.0e-12)) - math.log10(4.0))
    return np.asarray(log_re_kpc, dtype=float) - mu_r



def _compute_log_sigma_star(log_mstar: np.ndarray, log_re_kpc: np.ndarray) -> np.ndarray:
    """
    Compute the stellar surface-density proxy used by the sigma-star gamma model.

    The scientific definition in this project is the log surface density within
    one effective radius:
    `log Sigma_* = log M_* - log10(2π) - 2 log r_e`.
    Centralizing the formula keeps the observed overlays, posterior trend
    panels, and latent-population simulation on the exact same contract.
    """

    return np.asarray(log_mstar, dtype=float) - LOG10_2PI - 2.0 * np.asarray(log_re_kpc, dtype=float)


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


def _infer_mass_definition_from_trend_npz_keys(dataset_names: set[str]) -> MassDefinition:
    """
    Infer whether an existing trend run is an `m5` or `m10` product.

    The redraw workflow intentionally trusts the already-materialized `.npz`
    artifact instead of directory names. This keeps the annotator aligned with
    the actual plotted quantity even if a run is renamed or copied elsewhere.
    """

    if any(name.endswith("_m10_draws") for name in dataset_names):
        return get_mass_definition(10)
    return get_mass_definition(5)


def _load_trend_summary_from_npz(npz_path: Path) -> tuple[np.ndarray, MassDefinition, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """
    Reconstruct percentile bands directly from the saved trend draw arrays.

    The annotator intentionally avoids `fig8_like_summary.json` because the
    `.npz` contains the exact draw arrays that originally drove the figure.
    Re-summarizing them keeps the redraw numerically faithful while remaining
    independent of any historical summary-schema drift.
    """

    with np.load(npz_path) as arrays:
        dataset_names = set(arrays.files)
        mass_definition = _infer_mass_definition_from_trend_npz_keys(dataset_names)
        mass_grid = np.asarray(arrays["mass_bin_centers"], dtype=float)
        summary_payload: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for quantity_name in _trend_quantity_names(mass_definition):
            summary_payload[quantity_name] = {}
            for category_name in TREND_CATEGORY_NAMES:
                draw_key = f"{category_name}_{quantity_name}_draws"
                summary_payload[quantity_name][category_name] = _summarize_trend_draws(
                    np.asarray(arrays[draw_key], dtype=float)
                )
    return mass_grid, mass_definition, summary_payload


def _load_single_quantity_trend_summary_from_npz(
    npz_path: Path,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    """
    Reconstruct one gamma-only structural trend summary from its saved draw NPZ.

    The standalone trend products persist the raw per-draw gamma curves. The
    historical redraw workflow should reuse those arrays directly so the
    resulting composite figure remains numerically tied to the already-saved
    Monte Carlo output rather than any secondary summary JSON.
    """

    with np.load(npz_path) as arrays:
        x_grid = np.asarray(arrays["x_bin_centers"], dtype=float)
        summary_payload = {
            category_name: _summarize_trend_draws(np.asarray(arrays[f"{category_name}_gamma_draws"], dtype=float))
            for category_name in TREND_CATEGORY_NAMES
        }
    return x_grid, summary_payload


def _normalize_gamma_mode(raw_mode: Any) -> str | None:
    """Normalize a raw mode string into one of the supported gamma modes."""

    if not isinstance(raw_mode, str):
        return None
    return _GAMMA_MODE_ALIASES.get(raw_mode.strip().lower())


def _infer_gamma_mode_from_run_name(run_name: str) -> str:
    """
    Infer gamma mode from legacy run-name conventions when metadata is absent.

    Legacy run trees may predate explicit `gamma_mode` serialization. This
    fallback keeps redraws deterministic without mutating historical files.
    """

    lowered = run_name.lower()
    if "sigma_star" in lowered or "sigma-star" in lowered:
        return GAMMA_MODE_SIGMA_STAR_DEPENDENT
    if "independent" in lowered:
        return GAMMA_MODE_INDEPENDENT
    return GAMMA_MODE_DEPENDENT


def _resolve_gamma_mode_for_fig8_run(run_dir: Path, fig8_summary_path: Path) -> str:
    """
    Resolve one run's gamma mode without triggering config migration side effects.

    Resolution order:
    1) `ppc/fig8_like_summary.json` `gamma_mode`
    2) `config_snapshot.yaml` `gamma_model.mode` (read-only YAML parse)
    3) run-name fallback (`independent` / `sigma_star` / default dependent)
    """

    if fig8_summary_path.exists():
        try:
            summary_payload = json.loads(fig8_summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary_payload = None
        if isinstance(summary_payload, dict):
            from_summary = _normalize_gamma_mode(summary_payload.get("gamma_mode"))
            if from_summary is not None:
                return from_summary

    config_snapshot_path = run_dir / "config_snapshot.yaml"
    if config_snapshot_path.exists():
        try:
            raw_config = yaml.safe_load(config_snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            raw_config = None
        if isinstance(raw_config, dict):
            gamma_model = raw_config.get("gamma_model")
            if isinstance(gamma_model, dict):
                from_config = _normalize_gamma_mode(gamma_model.get("mode"))
                if from_config is not None:
                    return from_config

    return _infer_gamma_mode_from_run_name(run_dir.name)


def _format_fig8_like_title(mass_definition: MassDefinition, gamma_mode: str) -> str:
    """Build the user-facing figure title for one Fig. 8-like render."""

    normalized_mode = _normalize_gamma_mode(gamma_mode) or GAMMA_MODE_DEPENDENT
    mode_label = _GAMMA_MODE_TITLE_LABELS[normalized_mode]
    return f"{mass_definition.label} | {mode_label} gamma"


def _load_bic_value_from_run_dir(run_dir: Path) -> float:
    """
    Load the precomputed BIC value for one completed run.

    Why this helper exists:
    - the redraw workflow should reuse the already-materialized scientific
      result instead of silently recomputing model-comparison metrics
    - the user explicitly asked for fail-fast behavior when a target run is
      missing the required BIC artifact
    - keeping the JSON access in one place makes error handling and future
      schema changes easier to audit
    """

    bic_result_path = run_dir / "bic_result.json"
    if not bic_result_path.exists():
        raise FileNotFoundError(
            f"Cannot annotate Fig. 8-like figure for '{run_dir}': missing '{bic_result_path.name}'."
        )

    try:
        bic_payload = json.loads(bic_result_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - surfaced through CLI result payload
        raise ValueError(f"Failed to parse BIC payload '{bic_result_path}': {exc}") from exc

    bic_value = bic_payload.get("bic")
    if bic_value is None:
        raise ValueError(f"BIC payload '{bic_result_path}' does not contain a 'bic' field.")
    return float(bic_value)


def _build_annotated_fig8_title(
    run_dir: Path,
    profile_name: str,
    mass_definition: MassDefinition,
    gamma_mode: str,
) -> str:
    """
    Build the custom one-line title used by the redraw-only annotation command.

    The underlying Fig. 8-like trend computation remains unchanged. This title
    helper only enriches the visual presentation for already-generated figures
    by prepending the profile name and appending the precomputed BIC value.
    """

    base_title = _format_fig8_like_title(mass_definition=mass_definition, gamma_mode=gamma_mode)
    try:
        bic_value = _load_bic_value_from_run_dir(run_dir)
    except FileNotFoundError:
        # Synthetic tests and older PPC-only run directories may not carry the
        # optional BIC artifact. Annotation should still redraw observed
        # overlays in that case; keep the existing Fig. 8 title contract.
        return base_title
    return f"{profile_name} | {base_title} | BIC={bic_value:.2f}"


def _build_fixed_fig8_display_ylim_by_panel(mass_definition: MassDefinition) -> dict[str, tuple[float, float]]:
    """
    Return the fixed per-panel y-axis ranges used for the curated historical redraws.

    Why this helper exists:
    - the user wants the same panel ranges across a selected set of four
      already-generated figures so visual comparison is direct
    - these limits are presentation choices, not scientific recalculations, so
      they should stay explicit and centralized instead of being scattered as
      anonymous literals through the redraw path
    - binding the mass panel key to the active mass definition keeps the helper
      safe for `m10` naming without hardcoding a bare `"m10"` everywhere
    """

    return {
        mass_definition.label: (11.2, 12.3),
        "gamma": (1.45, 2.35),
        "sigma_ap": (140.0, 370.0),
        "gamma_vs_sigma_star": (1.45, 2.35),
        "gamma_vs_logre_kpc": (1.45, 2.35),
    }


def _build_fixed_fig8_display_xlim_by_panel(
    mass_definition: MassDefinition,
    profile_name: str,
) -> dict[str, tuple[float, float]]:
    """
    Return the fixed per-panel x-axis ranges used for the curated historical redraws.

    The first three panels continue to share stellar-mass on the x-axis, while
    the two newly added gamma-only diagnostic panels use their own structural
    coordinates. The user wants `gamma_vs_sigma_star` to be consistent within
    one profile family, but not necessarily identical across `devauc` and
    `sersic`, so that panel gets a profile-specific window.
    """

    normalized_profile_name = profile_name.strip().lower()
    if normalized_profile_name == "devauc":
        sigma_star_xlim = (8.9, 9.6)
        logre_xlim = (0.45, 0.95)
    elif normalized_profile_name == "sersic":
        sigma_star_xlim = (8.1, 9.5)
        logre_xlim = (0.50, 1.40)
    else:
        raise ValueError(f"Unsupported profile '{profile_name}' for fixed Fig. 8 x-axis limits.")

    return {
        mass_definition.label: (11.0, 11.8),
        "gamma": (11.0, 11.8),
        "sigma_ap": (11.0, 11.8),
        "gamma_vs_sigma_star": sigma_star_xlim,
        "gamma_vs_logre_kpc": logre_xlim,
    }


def _update_existing_fig8_summary_metadata(
    fig8_summary_path: Path,
    figure_title: str,
    display_xlim_by_panel: dict[str, tuple[float, float]],
    display_ylim_by_panel: dict[str, tuple[float, float]],
) -> None:
    """
    Keep the Fig. 8 summary JSON aligned with the rewritten PNG.

    The annotate command redraws an already-finished figure without rerunning
    the expensive trend simulation. Once the visual title and visible x-range
    change, the sidecar summary must be updated as well; otherwise the image
    and machine-readable metadata drift apart and become hard to trust later.
    """

    if not fig8_summary_path.exists():
        raise FileNotFoundError(
            f"Cannot update Fig. 8 summary metadata because '{fig8_summary_path}' does not exist."
        )

    try:
        summary_payload = json.loads(fig8_summary_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - surfaced through CLI result payload
        raise ValueError(f"Failed to parse Fig. 8 summary '{fig8_summary_path}': {exc}") from exc

    if not isinstance(summary_payload, dict):
        raise ValueError(f"Fig. 8 summary '{fig8_summary_path}' must contain a JSON object.")

    summary_payload["figure_title"] = figure_title
    summary_payload["layout"] = "5x1"
    summary_payload["panel_order"] = ["m10", "gamma", "sigma_ap", "gamma_vs_sigma_star", "gamma_vs_logre_kpc"]
    summary_payload["display_xlim"] = [11.0, 11.8]
    summary_payload["display_xlim_by_panel"] = {
        quantity_name: [float(axis_limits[0]), float(axis_limits[1])]
        for quantity_name, axis_limits in display_xlim_by_panel.items()
    }
    summary_payload["display_ylim_by_panel"] = {
        quantity_name: [float(axis_limits[0]), float(axis_limits[1])]
        for quantity_name, axis_limits in display_ylim_by_panel.items()
    }
    fig8_summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _resolve_first_matching_attr(group: h5py.Group, aliases: tuple[str, ...]) -> float:
    """
    Resolve the first available HDF5 attribute among a set of aliases.

    The observation files have profile-dependent stellar-mass field names. The
    redraw logic therefore mirrors the inference-side alias policy instead of
    hardcoding a single attribute per profile.
    """

    for alias in aliases:
        if alias in group.attrs:
            return float(group.attrs[alias])
    raise KeyError(f"None of the requested aliases were found in group '{group.name}': {aliases}")


def _stellar_mass_aliases_for_convention(
    profile_spec: ProfileSpec,
    mass_definition: MassDefinition,
) -> tuple[str, ...]:
    """Return observed stellar-mass attrs for the active unit convention."""

    if mass_definition.unit_convention == H_UNITS_V1:
        if profile_spec.name == "devauc":
            return ("logmchab_deV_h2", "logmchab_h2")
        return ("logmchab_h2",)
    return profile_spec.observation_field_aliases["stellar_mass"]


def _size_log_aliases_for_h_units(profile_spec: ProfileSpec) -> tuple[str, ...]:
    """Return observed h-units size-log attrs for the active profile."""

    if profile_spec.name == "devauc":
        return ("log10_reff_deV_hinv_kpc", "log10_re_hinv_kpc")
    return ("log10_re_hinv_kpc",)


def _load_observed_trend_points(
    observation_path: Path,
    profile_name: str,
    mass_definition: MassDefinition,
) -> dict[str, ObservedTrendSeries]:
    """
    Load the observed Fig. 8 scatter points from the raw HDF5 file.

    The user-prepared attrs already store the per-lens flat-prior summaries for
    `m5/m10` and `gamma`, so this helper only needs to map them into plotting
    arrays. The sigma panel is different: it should show every raw
    velocity-dispersion measurement, including both entries for `num_sigma=2`
    lenses, so that panel expands one lens into one or two points.
    """

    profile_spec = build_profile_spec(profile_name)
    mass_quantity = mass_definition.label
    mass_x_values: list[float] = []
    mass_y_values: list[float] = []
    mass_lower_errors: list[float] = []
    mass_upper_errors: list[float] = []
    gamma_x_values: list[float] = []
    gamma_y_values: list[float] = []
    gamma_lower_errors: list[float] = []
    gamma_upper_errors: list[float] = []
    sigma_x_values: list[float] = []
    sigma_y_values: list[float] = []
    sigma_lower_errors: list[float] = []
    sigma_upper_errors: list[float] = []

    with h5py.File(observation_path, "r") as handle:
        for group_name in sorted(handle.keys()):
            group = handle[group_name]
            num_sigma = int(group.attrs.get("num_sigma", 0))
            if num_sigma <= 0:
                continue

            stellar_mass = _resolve_first_matching_attr(
                group,
                _stellar_mass_aliases_for_convention(profile_spec, mass_definition),
            )

            mass_mid = float(group.attrs[f"{mass_quantity}_mid"])
            mass_lower = float(group.attrs[f"{mass_quantity}_lower"])
            mass_upper = float(group.attrs[f"{mass_quantity}_upper"])
            gamma_mid = float(group.attrs["gamma_mid"])
            gamma_lower = float(group.attrs["gamma_lower"])
            gamma_upper = float(group.attrs["gamma_upper"])
            mass_lower_error, mass_upper_error = _coerce_observed_error_components(
                mid_value=mass_mid,
                lower_value=mass_lower,
                upper_value=mass_upper,
            )
            gamma_lower_error, gamma_upper_error = _coerce_observed_error_components(
                mid_value=gamma_mid,
                lower_value=gamma_lower,
                upper_value=gamma_upper,
            )

            mass_x_values.append(stellar_mass)
            mass_y_values.append(mass_mid)
            mass_lower_errors.append(mass_lower_error)
            mass_upper_errors.append(mass_upper_error)

            gamma_x_values.append(stellar_mass)
            gamma_y_values.append(gamma_mid)
            gamma_lower_errors.append(gamma_lower_error)
            gamma_upper_errors.append(gamma_upper_error)

            sigma_values = np.atleast_1d(np.asarray(group.attrs["sigma"], dtype=float))
            sigma_errors = np.atleast_1d(np.asarray(group.attrs["sigma_err"], dtype=float))
            if sigma_values.shape != sigma_errors.shape:
                raise ValueError(
                    f"Group '{group.name}' has mismatched sigma and sigma_err shapes: "
                    f"{sigma_values.shape} vs {sigma_errors.shape}."
                )
            for sigma_value, sigma_error in zip(sigma_values, sigma_errors, strict=True):
                sigma_x_values.append(stellar_mass)
                sigma_y_values.append(float(sigma_value))
                sigma_lower_errors.append(float(sigma_error))
                sigma_upper_errors.append(float(sigma_error))

    return {
        mass_quantity: ObservedTrendSeries(
            x=np.asarray(mass_x_values, dtype=float),
            y=np.asarray(mass_y_values, dtype=float),
            yerr_lower=np.asarray(mass_lower_errors, dtype=float),
            yerr_upper=np.asarray(mass_upper_errors, dtype=float),
        ),
        "gamma": ObservedTrendSeries(
            x=np.asarray(gamma_x_values, dtype=float),
            y=np.asarray(gamma_y_values, dtype=float),
            yerr_lower=np.asarray(gamma_lower_errors, dtype=float),
            yerr_upper=np.asarray(gamma_upper_errors, dtype=float),
        ),
        "sigma_ap": ObservedTrendSeries(
            x=np.asarray(sigma_x_values, dtype=float),
            y=np.asarray(sigma_y_values, dtype=float),
            yerr_lower=np.asarray(sigma_lower_errors, dtype=float),
            yerr_upper=np.asarray(sigma_upper_errors, dtype=float),
        ),
    }


def _load_observed_gamma_measurements(
    observation_path: Path,
    profile_name: str,
    observations: list[ObservationRecord],
    cosmology: FlatLambdaCDM,
    mass_definition: MassDefinition,
) -> ObservedGammaMeasurements:
    """
    Load the observed gamma sample together with structural coordinates.

    The standalone gamma trend figures need the same observed gamma summaries
    used by the Fig. 8 overlay, but they also need structural x-coordinates in
    physical units. We therefore join the raw HDF5 attrs with the already
    prepared observation records returned by `build_compiled_context(...)`.
    """

    profile_spec = build_profile_spec(profile_name)
    observation_by_id = {observation.lens_id: observation for observation in observations}
    lens_ids: list[str] = []
    log_mstar_values: list[float] = []
    log_re_kpc_values: list[float] = []
    n_values: list[float] = []
    gamma_mid_values: list[float] = []
    gamma_lower_errors: list[float] = []
    gamma_upper_errors: list[float] = []

    with h5py.File(observation_path, "r") as handle:
        for group_name in sorted(handle.keys()):
            group = handle[group_name]
            num_sigma = int(group.attrs.get("num_sigma", 0))
            if num_sigma <= 0:
                continue

            if group_name not in observation_by_id:
                raise KeyError(f"Observed gamma group '{group_name}' is missing from the prepared observation map.")
            prepared_observation = observation_by_id[group_name]

            gamma_mid = float(group.attrs["gamma_mid"])
            gamma_lower = float(group.attrs["gamma_lower"])
            gamma_upper = float(group.attrs["gamma_upper"])
            gamma_lower_error, gamma_upper_error = _coerce_observed_error_components(
                mid_value=gamma_mid,
                lower_value=gamma_lower,
                upper_value=gamma_upper,
            )

            lens_ids.append(group_name)
            log_mstar_values.append(
                _resolve_first_matching_attr(
                    group,
                    _stellar_mass_aliases_for_convention(profile_spec, mass_definition),
                )
            )
            if mass_definition.unit_convention == H_UNITS_V1:
                log_re_kpc_values.append(
                    _resolve_first_matching_attr(group, _size_log_aliases_for_h_units(profile_spec))
                )
            else:
                radius_kpc = float(prepared_observation.effective_radius_arcsec) * float(
                    cosmology.kpc_per_arcsec(prepared_observation.z_d)
                )
                log_re_kpc_values.append(math.log10(max(radius_kpc, 1.0e-12)))
            n_values.append(float(profile_spec.fixed_n if profile_spec.fixed_n is not None else prepared_observation.n_observed))
            gamma_mid_values.append(gamma_mid)
            gamma_lower_errors.append(gamma_lower_error)
            gamma_upper_errors.append(gamma_upper_error)

    return ObservedGammaMeasurements(
        lens_ids=tuple(lens_ids),
        log_mstar=np.asarray(log_mstar_values, dtype=float),
        log_re_kpc=np.asarray(log_re_kpc_values, dtype=float),
        log_sigma_star=_compute_log_sigma_star(
            np.asarray(log_mstar_values, dtype=float),
            np.asarray(log_re_kpc_values, dtype=float),
        ),
        n_value=np.asarray(n_values, dtype=float),
        gamma_mid=np.asarray(gamma_mid_values, dtype=float),
        gamma_yerr_lower=np.asarray(gamma_lower_errors, dtype=float),
        gamma_yerr_upper=np.asarray(gamma_upper_errors, dtype=float),
    )


def _build_trend_axis_spec(
    name: str,
    label: str,
    reference_values: np.ndarray,
    n_bins: int,
    padding_fraction: float,
    minimum_padding: float,
    observed_overlay_mode: str,
    figure_label: str | None = None,
) -> TrendAxisSpec:
    """Build one explicit x-axis contract from a reference value range."""

    bin_edges = _build_padded_bin_edges(
        values=reference_values,
        n_bins=n_bins,
        padding_fraction=padding_fraction,
        minimum_padding=minimum_padding,
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return TrendAxisSpec(
        name=name,
        label=label,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        observed_overlay_mode=observed_overlay_mode,
        figure_label=figure_label,
    )


def _finite_observed_gamma_series(x_values: np.ndarray, measurements: ObservedGammaMeasurements) -> ObservedTrendSeries:
    """
    Convert observed gamma measurements into finite plotting points.

    Canonical inference datasets do not yet carry the raw flat-prior gamma
    summary attrs used by legacy overlays.  Those canonical fallbacks represent
    gamma as NaN, and this helper filters them out instead of serializing NaN
    values into JSON artifacts.
    """

    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(measurements.gamma_mid, dtype=float)
    lower_array = np.asarray(measurements.gamma_yerr_lower, dtype=float)
    upper_array = np.asarray(measurements.gamma_yerr_upper, dtype=float)
    finite_mask = (
        np.isfinite(x_array)
        & np.isfinite(y_array)
        & np.isfinite(lower_array)
        & np.isfinite(upper_array)
    )
    return ObservedTrendSeries(
        x=x_array[finite_mask],
        y=y_array[finite_mask],
        yerr_lower=lower_array[finite_mask],
        yerr_upper=upper_array[finite_mask],
    )


def _build_observed_gamma_logre_overlay(measurements: ObservedGammaMeasurements) -> ObservedTrendSeries:
    """Convert observed gamma measurements into fixed `logre_kpc` errorbar points."""

    return _finite_observed_gamma_series(measurements.log_re_kpc, measurements)


def _build_observed_gamma_sigma_star_overlay(measurements: ObservedGammaMeasurements) -> ObservedTrendSeries:
    """
    Convert observed gamma measurements into fixed `log Sigma_*` errorbar points.

    This overlay is a direct point series because the x-coordinate depends only
    on the observed stellar mass and effective radius, not on posterior draws.
    """

    return _finite_observed_gamma_series(measurements.log_sigma_star, measurements)


def _build_observed_gamma_delta_r_overlay(
    measurements: ObservedGammaMeasurements,
    profile: ProfileSpec,
    axis_spec: TrendAxisSpec,
) -> ObservedTrendSeries:
    """
    Convert observed gamma measurements into fixed `delta_r` errorbar points.

    The current model keeps `mu_r` fixed in `ProfileSpec`, so each observed
    lens has one well-defined `delta_r = logre_kpc - mu_r` coordinate. That
    makes the scientifically faithful overlay a point series, not a posterior
    band summarized over bins.
    """

    delta_r = _compute_observed_delta_r(
        log_mstar=measurements.log_mstar,
        n_value=measurements.n_value,
        log_re_kpc=measurements.log_re_kpc,
        profile=profile,
    )
    del axis_spec
    return _finite_observed_gamma_series(delta_r, measurements)


def _serialize_observed_overlay(overlay: ObservedTrendOverlay) -> dict[str, Any]:
    """Convert one observed overlay object into JSON-friendly arrays."""

    if isinstance(overlay, ObservedTrendSeries):
        return {
            "mode": "points",
            "x": overlay.x.tolist(),
            "y": overlay.y.tolist(),
            "yerr_lower": overlay.yerr_lower.tolist(),
            "yerr_upper": overlay.yerr_upper.tolist(),
        }
    return {
        "mode": "band",
        "x": overlay.x.tolist(),
        "p16": overlay.p16.tolist(),
        "p50": overlay.p50.tolist(),
        "p84": overlay.p84.tolist(),
    }


def _coerce_observed_error_components(mid_value: float, lower_value: float, upper_value: float) -> tuple[float, float]:
    """
    Convert raw `lower` / `upper` attrs into strictly non-negative y-errors.

    Two encodings need to be supported:
    - synthetic tests store absolute interval bounds around `mid`
    - production raw files store already-computed lower / upper error magnitudes
    The branch below keeps both representations valid without forcing the raw
    producer and test fixtures into the same convention.
    """

    if lower_value <= mid_value <= upper_value:
        return (mid_value - lower_value, upper_value - mid_value)
    return (abs(lower_value), abs(upper_value))


def _write_trend_panel(
    ax,
    mass_grid: np.ndarray,
    parent_summary: dict[str, np.ndarray],
    detectable_summary: dict[str, np.ndarray],
    selected_summary: dict[str, np.ndarray],
    y_label: str,
    observed_series: ObservedTrendOverlay | None = None,
    observed_label: str | None = None,
) -> None:
    """
    Render one panel of the Fig. 8-like trend figure.

    The styling intentionally assigns one visual channel per population so the
    three model categories remain distinguishable when their envelopes overlap:
    - parent population: magenta uncertainty band spanning `p16-p84`
    - detectable lenses: black solid boundary lines at `p16` and `p84`
    - full_selection: blue dashed boundary lines at `p16` and `p84`
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
        label="full_selection",
    )
    ax.plot(
        mass_grid,
        selected_summary["p84"],
        color="#1565c0",
        linewidth=2.0,
        linestyle="--",
    )
    if isinstance(observed_series, ObservedTrendSeries) and observed_series.x.size > 0:
        ax.errorbar(
            observed_series.x,
            observed_series.y,
            yerr=np.vstack((observed_series.yerr_lower, observed_series.yerr_upper)),
            fmt="o",
            color="#111111",
            ecolor="#111111",
            elinewidth=1.1,
            capsize=2.5,
            markersize=4.5,
            markerfacecolor="#111111",
            markeredgecolor="#111111",
            linestyle="none",
            zorder=6,
            label=observed_label,
        )
    elif isinstance(observed_series, ObservedTrendBand) and observed_series.x.size > 0:
        ax.fill_between(
            observed_series.x,
            observed_series.p16,
            observed_series.p84,
            color="#111111",
            alpha=0.14,
            label=observed_label,
            zorder=5,
        )
        ax.plot(
            observed_series.x,
            observed_series.p50,
            color="#111111",
            linewidth=1.8,
            linestyle="-",
            zorder=6,
        )
    ax.set_ylabel(y_label, fontsize=10)
    ax.tick_params(labelsize=8)


def _write_fig8_like_figure(
    figure_path: Path,
    mass_grid: np.ndarray,
    summary_payload: dict[str, dict[str, dict[str, np.ndarray]]],
    mass_definition: MassDefinition,
    observed_points: dict[str, ObservedTrendSeries] | None = None,
    extra_gamma_panels: list[dict[str, Any]] | None = None,
    figure_title: str | None = None,
    display_xlim_by_panel: dict[str, tuple[float, float]] | None = None,
    display_ylim_by_panel: dict[str, tuple[float, float]] | None = None,
) -> None:
    """
    Render the composite Fig. 8-like trend figure.

    Parameters
    ----------
    extra_gamma_panels:
        Optional standalone gamma-only panels appended below the historical
        three-panel Fig. 8 block. Each panel definition must provide:
        `panel_id`, `x_grid`, `summary_payload`, `x_label`, and
        `observed_overlay`.
    display_xlim_by_panel:
        Optional mapping from panel identifier to its visible x-axis range.
    display_ylim_by_panel:
        Optional mapping from panel quantity name to a fixed visible y-axis
        range. This exists for curated figure redraws where multiple runs must
        share identical panel limits for fair visual comparison.
    """

    historical_panels = [
        {
            "panel_id": mass_definition.label,
            "summary_payload": summary_payload[mass_definition.label],
            "x_grid": mass_grid,
            "y_label": mass_definition.label,
            "x_label": None,
            "observed_overlay": None if observed_points is None else observed_points.get(mass_definition.label),
            "observed_label": "Observed lenses" if observed_points is not None else None,
        },
        {
            "panel_id": "gamma",
            "summary_payload": summary_payload["gamma"],
            "x_grid": mass_grid,
            "y_label": "gamma",
            "x_label": None,
            "observed_overlay": None if observed_points is None else observed_points.get("gamma"),
            "observed_label": None,
        },
        {
            "panel_id": "sigma_ap",
            "summary_payload": summary_payload["sigma_ap"],
            "x_grid": mass_grid,
            "y_label": "sigma_ap [km/s]",
            "x_label": _stellar_mass_axis_label(mass_definition),
            "observed_overlay": None if observed_points is None else observed_points.get("sigma_ap"),
            "observed_label": None,
        },
    ]
    panel_specs = historical_panels + ([] if extra_gamma_panels is None else list(extra_gamma_panels))

    figure_height = 2.7 * len(panel_specs) + 1.2
    figure, axes = plt.subplots(len(panel_specs), 1, figsize=(8, figure_height), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes], dtype=object)

    for axis, panel_spec in zip(axes, panel_specs, strict=True):
        panel_id = str(panel_spec["panel_id"])
        _write_trend_panel(
            axis,
            mass_grid=np.asarray(panel_spec["x_grid"], dtype=float),
            parent_summary=panel_spec["summary_payload"]["parent"],
            detectable_summary=panel_spec["summary_payload"]["detectable"],
            selected_summary=panel_spec["summary_payload"]["selected"],
            y_label=str(panel_spec["y_label"]),
            observed_series=panel_spec.get("observed_overlay"),
            observed_label=panel_spec.get("observed_label"),
        )
        if display_xlim_by_panel is not None and panel_id in display_xlim_by_panel:
            x_axis_limits = display_xlim_by_panel[panel_id]
            axis.set_xlim(float(x_axis_limits[0]), float(x_axis_limits[1]))
        if display_ylim_by_panel is not None and panel_id in display_ylim_by_panel:
            # The redraw command uses explicit, shared panel limits across
            # multiple historical runs. We only apply ranges that are present
            # in the supplied mapping so the base writer remains reusable.
            y_axis_limits = display_ylim_by_panel[panel_id]
            axis.set_ylim(float(y_axis_limits[0]), float(y_axis_limits[1]))
        x_label = panel_spec.get("x_label")
        if x_label:
            axis.set_xlabel(str(x_label), fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper left", fontsize=8, frameon=False)
    if figure_title:
        figure.suptitle(figure_title, fontsize=13)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    else:
        figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def _write_gamma_trend_figure(
    figure_path: Path,
    x_grid: np.ndarray,
    summary_payload: dict[str, dict[str, np.ndarray]],
    x_label: str,
    observed_overlay: ObservedTrendOverlay,
    figure_title: str | None = None,
) -> None:
    """
    Render a standalone gamma trend figure for one structural x-axis.

    Keeping the new plots in their own helper avoids entangling the stable
    three-panel Fig. 8 output with the new one-panel diagnostic products.
    """

    figure, axis = plt.subplots(1, 1, figsize=(8, 4.6))
    _write_trend_panel(
        axis,
        mass_grid=x_grid,
        parent_summary=summary_payload["parent"],
        detectable_summary=summary_payload["detectable"],
        selected_summary=summary_payload["selected"],
        y_label="gamma",
        observed_series=observed_overlay,
        observed_label="Observed lenses",
    )
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles, labels, loc="upper left", fontsize=8, frameon=False)
    axis.set_xlabel(x_label, fontsize=10)
    if figure_title:
        figure.suptitle(figure_title, fontsize=13)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    else:
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
    """Create the deterministic result directory for one PPC run.

    All PPC-family workflows now write artifacts under a dedicated `ppc`
    subdirectory so inference-chain files and PPC products never share the
    same run root.
    """

    result_dir = output_root_dir.expanduser().resolve() / profile_name / run_id / "ppc"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def _write_standalone_gamma_trend_artifacts(
    result_dir: Path,
    artifact_stem: str,
    axis_spec: TrendAxisSpec,
    gamma_summary: dict[str, dict[str, np.ndarray]],
    gamma_draws: dict[str, np.ndarray],
    parent_bin_counts_draws: np.ndarray,
    detectable_weight_sums_draws: np.ndarray,
    selected_weight_sums_draws: np.ndarray,
    observed_overlay: ObservedTrendOverlay,
    observed_overlay_draws: np.ndarray | None,
    base_metadata: dict[str, Any],
    figure_title: str,
) -> None:
    """
    Write one standalone gamma-trend PNG, summary JSON, and raw NPZ arrays.

    The new structural trend diagnostics intentionally live beside
    `fig8_like.*`, but each one carries its own explicit x-axis metadata and
    observed-overlay contract so downstream inspection does not need to infer
    what was plotted.
    """

    serializable_summary = {
        category_name: {key: value.tolist() for key, value in gamma_summary[category_name].items()}
        for category_name in TREND_CATEGORY_NAMES
    }
    summary_payload = {
        **base_metadata,
        "figure_title": figure_title,
        "x_axis_name": axis_spec.name,
        "x_axis_label": axis_spec.label,
        "x_bin_edges": axis_spec.bin_edges.tolist(),
        "x_bin_centers": axis_spec.bin_centers.tolist(),
        "observed_overlay_mode": axis_spec.observed_overlay_mode,
        "quantities": {"gamma": {"label": "gamma"}},
        "bands": {"gamma": serializable_summary},
        "observed_overlay": _serialize_observed_overlay(observed_overlay),
    }
    (result_dir / f"{artifact_stem}_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    np_save_payload: dict[str, np.ndarray] = {
        "x_bin_edges": np.asarray(axis_spec.bin_edges, dtype=float),
        "x_bin_centers": np.asarray(axis_spec.bin_centers, dtype=float),
        "parent_bin_counts_draws": np.asarray(parent_bin_counts_draws),
        "detectable_weight_sums_draws": np.asarray(detectable_weight_sums_draws),
        "selected_weight_sums_draws": np.asarray(selected_weight_sums_draws),
    }
    for category_name in TREND_CATEGORY_NAMES:
        np_save_payload[f"{category_name}_gamma_draws"] = np.asarray(gamma_draws[category_name], dtype=float)
    if isinstance(observed_overlay, ObservedTrendSeries):
        np_save_payload["observed_x"] = np.asarray(observed_overlay.x, dtype=float)
        np_save_payload["observed_y"] = np.asarray(observed_overlay.y, dtype=float)
        np_save_payload["observed_yerr_lower"] = np.asarray(observed_overlay.yerr_lower, dtype=float)
        np_save_payload["observed_yerr_upper"] = np.asarray(observed_overlay.yerr_upper, dtype=float)
    else:
        np_save_payload["observed_p16"] = np.asarray(observed_overlay.p16, dtype=float)
        np_save_payload["observed_p50"] = np.asarray(observed_overlay.p50, dtype=float)
        np_save_payload["observed_p84"] = np.asarray(observed_overlay.p84, dtype=float)
        if observed_overlay_draws is not None:
            np_save_payload["observed_gamma_draws"] = np.asarray(observed_overlay_draws, dtype=float)
    np.savez(result_dir / f"{artifact_stem}_curves.npz", **np_save_payload)

    _write_gamma_trend_figure(
        figure_path=result_dir / f"{artifact_stem}.png",
        x_grid=axis_spec.bin_centers,
        summary_payload=gamma_summary,
        x_label=axis_spec.figure_label or axis_spec.label,
        observed_overlay=observed_overlay,
        figure_title=figure_title,
    )


def _resolve_diagnostics_execution(
    runtime_config: RuntimeConfig,
    requested_worker_processes: int | None,
    n_draws: int,
) -> DiagnosticsExecution:
    """
    Resolve the generic PPC diagnostics execution policy and metadata payload.

    ``run_posterior_diagnostics`` exposes a historical ``worker_processes``
    argument, but the shared-parent diagnostics path is currently kernel-only:
    adapters consume one Numba/math-library thread budget instead of a Python
    process pool.  This helper keeps that policy explicit so artifacts record
    both sides of the contract:
    - the caller-facing process-width request that arrived at the public API
    - the kernel-thread width that the adapter may use for the actual compute

    The resolution mirrors the inference-side CPU-budget rules closely enough
    for reproducibility, while capping kernel width by the number of posterior
    draws so small diagnostics runs do not advertise unusable parallel work.
    """

    cpu_count = max(1, int(os.cpu_count() or 1))
    reserve_cores = max(0, int(runtime_config.runtime.reserve_cores))
    auto_budget = max(1, cpu_count - reserve_cores)
    configured_num_threads = int(runtime_config.runtime.num_threads)
    if configured_num_threads <= 0:
        compute_budget = auto_budget
    else:
        compute_budget = max(1, min(configured_num_threads, auto_budget))

    if requested_worker_processes is None:
        requested_width = compute_budget
        serialized_request = None
    else:
        requested_width = int(requested_worker_processes)
        if requested_width <= 0:
            raise ValueError("worker_processes must be positive when provided.")
        serialized_request = requested_width

    kernel_threads = max(1, min(requested_width, compute_budget, max(1, int(n_draws))))
    return DiagnosticsExecution(
        strategy="kernel_only",
        cpu_count=cpu_count,
        reserve_cores=reserve_cores,
        compute_budget=compute_budget,
        requested_worker_processes=serialized_request,
        worker_processes=0,
        kernel_threads_per_process=kernel_threads,
    )


def run_posterior_diagnostics(
    run_dir: str,
    sigma_table_path: str | Path | None = None,
    output_root_dir: str | Path = DEFAULT_PPC_OUTPUT_ROOT_DIR,
    n_posterior_draws: int | None = DEFAULT_TREND_POSTERIOR_DRAWS,
    burn_in: str | int = "auto",
    random_seed: int = DEFAULT_RANDOM_SEED,
    parent_sample_size: int = DEFAULT_DIAGNOSTICS_PARENT_SAMPLE_SIZE,
    n_mass_bins: int = DEFAULT_TREND_MASS_BIN_COUNT,
    mass_bin_min: float = DEFAULT_TREND_MASS_BIN_MIN,
    mass_bin_max: float = DEFAULT_TREND_MASS_BIN_MAX,
    worker_processes: int | None = None,
    posterior_draw_tail_cap: int = DEFAULT_CANONICAL_POSTERIOR_DRAW_CAP,
) -> PosteriorDiagnosticsResult:
    """
    Run PPC histograms and Fig. 8-like trends from one shared Numba parent sample.

    This is the Numba-accelerated production diagnostics path.  The old standalone
    PPC and trend functions remain available for compatibility, but this joint
    workflow is the only path that guarantees one parent-population realization
    per posterior draw is reused by both downstream diagnostics.
    """

    if parent_sample_size < max(THETA_SAMPLE_SIZE, SIGMA_SAMPLE_SIZE):
        raise ValueError(
            "Shared-parent diagnostics require `parent_sample_size` to be at least "
            f"{max(THETA_SAMPLE_SIZE, SIGMA_SAMPLE_SIZE)}."
        )
    if n_mass_bins < 1:
        raise ValueError("Shared-parent diagnostics require at least one stellar-mass bin.")
    if mass_bin_max <= mass_bin_min:
        raise ValueError("Shared-parent diagnostics require `mass_bin_max` to exceed `mass_bin_min`.")

    resolved_run_dir = Path(run_dir).expanduser().resolve()
    config_snapshot_path = resolved_run_dir / "config_snapshot.yaml"
    chain_path = resolved_run_dir / "chain.h5"
    runtime_config = _load_ppc_runtime_config(config_snapshot_path)
    predictive_definition = get_predictive_definition(runtime_config.model.name)
    predictive_metadata = _predictive_metadata_payload(predictive_definition)
    resolved_sigma_table_path = _resolve_sigma_table_path_for_definition(
        predictive_definition,
        sigma_table_path,
    )
    mass_definition = runtime_config.mass_definition
    mass_label = mass_definition.label
    trend_quantity_names = _trend_quantity_names(mass_definition)
    trend_category_names = predictive_definition.trend_category_names
    trend_panel_order = list(predictive_definition.build_trend_panel_order(mass_definition))

    burn_in_steps = _resolve_burn_in(burn_in, runtime_config.sampling.warmup)
    selection_rng = np.random.default_rng(random_seed)
    posterior_draws, posterior_draw_mode, posterior_artifact = _load_posterior_draws(
        chain_path=chain_path,
        burn_in=burn_in_steps,
        rng=selection_rng,
        n_replicates=n_posterior_draws,
        tail_cap=posterior_draw_tail_cap,
        parameter_names=runtime_config.parameter_schema.internal_parameter_names,
    )
    n_posterior_draws_used = int(posterior_draws.shape[0])
    if n_posterior_draws_used < 1:
        raise ValueError("Shared-parent diagnostics require at least one posterior draw.")
    execution = _resolve_diagnostics_execution(
        runtime_config=runtime_config,
        requested_worker_processes=worker_processes,
        n_draws=n_posterior_draws_used,
    )

    compiled_context, profile_spec, _, cosmology, _, observations = _build_ppc_context(runtime_config)
    observation_contract = _infer_observation_contract_from_runtime_config(runtime_config)
    observation_flavor = observation_contract.observation_flavor
    sigma_table = None
    sigma_table_leaf_path = None
    if resolved_sigma_table_path is not None:
        sigma_table = SigmaUnitTable.from_path(
            resolved_sigma_table_path,
            mass_definition=mass_definition,
            observation_flavor=observation_flavor,
        )
        _assert_sigma_table_matches_run(
            sigma_table=sigma_table,
            profile_name=profile_spec.name,
            mass_definition=mass_definition,
            observation_flavor=observation_flavor,
            observation_contract=observation_contract,
        )
        sigma_table_leaf_path = sigma_table.bundle_leaf_path

    mass_bin_edges = np.linspace(mass_bin_min, mass_bin_max, n_mass_bins + 1, dtype=float)
    mass_bin_centers = 0.5 * (mass_bin_edges[:-1] + mass_bin_edges[1:])
    observed_gamma_measurements = _load_observed_gamma_measurements_for_runtime(
        runtime_config=runtime_config,
        observations=observations,
        cosmology=cosmology,
        mass_definition=mass_definition,
    )
    sigma_star_axis_spec = _build_trend_axis_spec(
        name="sigma_star",
        label="log Σ_* [M_\\odot kpc^{-2}]",
        reference_values=observed_gamma_measurements.log_sigma_star,
        n_bins=n_mass_bins,
        padding_fraction=0.05,
        minimum_padding=0.02,
        observed_overlay_mode="points",
        figure_label=r"log $\Sigma_*$ [$M_\odot$ kpc$^{-2}$]",
    )
    log_re_axis_spec = _build_trend_axis_spec(
        name="logre_kpc",
        label=_effective_radius_axis_label(mass_definition),
        reference_values=observed_gamma_measurements.log_re_kpc,
        n_bins=n_mass_bins,
        padding_fraction=0.05,
        minimum_padding=0.02,
        observed_overlay_mode="points",
    )
    delta_r_reference = _compute_observed_delta_r(
        log_mstar=observed_gamma_measurements.log_mstar,
        n_value=observed_gamma_measurements.n_value,
        log_re_kpc=observed_gamma_measurements.log_re_kpc,
        profile=profile_spec,
        theta=np.median(posterior_draws, axis=0),
        stellar_mass_pivot=float(getattr(compiled_context, "stellar_mass_pivot", 11.4)),
        mu_r0=float(getattr(compiled_context, "mu_r0", profile_spec.mu_r0)),
    )
    delta_r_axis_spec = _build_trend_axis_spec(
        name="delta_r",
        label=f"{_effective_radius_axis_label(mass_definition)} - $\\mu_r$",
        reference_values=delta_r_reference,
        n_bins=n_mass_bins,
        padding_fraction=0.05,
        minimum_padding=0.01,
        observed_overlay_mode="points",
    )

    diagnostics = predictive_definition.run_diagnostics(
        posterior_draws=posterior_draws,
        profile=profile_spec,
        context=compiled_context,
        mass_definition=mass_definition,
        sigma_table=sigma_table,
        mass_bin_edges=mass_bin_edges,
        sigma_star_bin_edges=sigma_star_axis_spec.bin_edges,
        log_re_bin_edges=log_re_axis_spec.bin_edges,
        delta_r_bin_edges=delta_r_axis_spec.bin_edges,
        parent_sample_size=int(parent_sample_size),
        random_seed=int(random_seed),
        execution=execution,
    )

    theta_latent = diagnostics["theta_latent"]
    sigma_latent = diagnostics["sigma_latent"]
    theta_replicated_stats = diagnostics["theta_replicated_stats"]
    sigma_replicated_stats = diagnostics["sigma_replicated_stats"]
    theta_observed = _summary_statistics(_observed_theta_ein_values(observations))
    sigma_observed = _summary_statistics(_observed_sigma_values(observations))
    theta_summary = _summarize_observed_against_replicates(theta_observed, theta_replicated_stats)
    sigma_summary = _summarize_observed_against_replicates(sigma_observed, sigma_replicated_stats)

    trend_draws = diagnostics["trend_draws"]
    parent_bin_counts_draws = diagnostics["parent_bin_counts_draws"]
    detectable_weight_sums_draws = diagnostics["detectable_weight_sums_draws"]
    selected_weight_sums_draws = diagnostics["selected_weight_sums_draws"]
    gamma_vs_logre_draws = diagnostics["gamma_vs_logre_draws"]
    gamma_vs_sigma_star_draws = diagnostics["gamma_vs_sigma_star_draws"]
    gamma_vs_delta_r_draws = diagnostics["gamma_vs_delta_r_draws"]

    trend_summary = {
        quantity_name: {
            category_name: _summarize_trend_draws(trend_draws[quantity_name][category_name])
            for category_name in trend_category_names
        }
        for quantity_name in trend_quantity_names
    }
    gamma_vs_logre_summary = {
        category_name: _summarize_trend_draws(gamma_vs_logre_draws[category_name])
        for category_name in trend_category_names
    }
    gamma_vs_sigma_star_summary = {
        category_name: _summarize_trend_draws(gamma_vs_sigma_star_draws[category_name])
        for category_name in trend_category_names
    }
    gamma_vs_delta_r_summary = {
        category_name: _summarize_trend_draws(gamma_vs_delta_r_draws[category_name])
        for category_name in trend_category_names
    }

    result_dir = _materialize_result_dir(Path(output_root_dir), runtime_config.profile.name, resolved_run_dir.name)
    backend_name = predictive_definition.backend
    observation_path_metadata = (
        None
        if runtime_config.data.observation_path is None
        else str(runtime_config.data.observation_path)
    )
    parallelism_payload = execution.to_dict()

    ppc_summary_payload = {
        "run_id": resolved_run_dir.name,
        "profile_name": runtime_config.profile.name,
        "gamma_mode": _runtime_gamma_mode(runtime_config),
        "parameter_order": list(runtime_config.parameter_schema.public_parameter_names),
        "input_run_dir": str(resolved_run_dir),
        "observation_path": observation_path_metadata,
        "observation_flavor": observation_flavor,
        "result_dir": str(result_dir),
        "burn_in_applied": burn_in_steps,
        "requested_n_replicates": None if n_posterior_draws is None else int(n_posterior_draws),
        "n_posterior_draws_used": n_posterior_draws_used,
        "posterior_draw_mode": posterior_draw_mode,
        "posterior_artifact": posterior_artifact,
        "posterior_draw_tail_cap": int(posterior_draw_tail_cap),
        **predictive_metadata,
        "backend": backend_name,
        "parent_sample_size": int(parent_sample_size),
        "sigma_table_leaf_path": sigma_table_leaf_path,
        "mass_definition": mass_definition_metadata(mass_definition),
        "sample_sizes": {"theta_ein": THETA_SAMPLE_SIZE, "sigma": SIGMA_SAMPLE_SIZE},
        "statistics": {"theta_ein": theta_summary, "sigma": sigma_summary},
        "parallelism": dict(parallelism_payload),
    }
    (result_dir / "ppc_summary.json").write_text(json.dumps(ppc_summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    manifest_payload = {
        **{key: ppc_summary_payload[key] for key in (
            "run_id",
            "profile_name",
            "gamma_mode",
            "parameter_order",
            "observation_path",
            "observation_flavor",
            "burn_in_applied",
            "posterior_draw_mode",
            "posterior_artifact",
            "posterior_draw_tail_cap",
            "backend",
            "parent_sample_size",
            "sigma_table_leaf_path",
            "mass_definition",
        )},
        **predictive_metadata,
        "config_snapshot_path": str(config_snapshot_path),
        "chain_path": str(chain_path),
        "sigma_table_path": None if resolved_sigma_table_path is None else str(resolved_sigma_table_path),
        "random_seed": int(random_seed),
        "n_posterior_draws_used": n_posterior_draws_used,
        "parallelism": dict(parallelism_payload),
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

    serializable_summary = {
        quantity_name: {
            category_name: {
                key: value.tolist()
            for key, value in trend_summary[quantity_name][category_name].items()
            }
            for category_name in trend_category_names
        }
        for quantity_name in trend_quantity_names
    }
    figure_title = _format_fig8_like_title(
        mass_definition=mass_definition,
        gamma_mode=_runtime_gamma_mode(runtime_config),
    )
    trend_summary_payload = {
        "run_id": resolved_run_dir.name,
        "profile_name": runtime_config.profile.name,
        "gamma_mode": _runtime_gamma_mode(runtime_config),
        "parameter_order": list(runtime_config.parameter_schema.public_parameter_names),
        "input_run_dir": str(resolved_run_dir),
        "observation_path": observation_path_metadata,
        "observation_flavor": observation_flavor,
        "result_dir": str(result_dir),
        "burn_in_applied": burn_in_steps,
        "requested_n_posterior_draws": None if n_posterior_draws is None else int(n_posterior_draws),
        "n_posterior_draws": n_posterior_draws_used,
        "n_posterior_draws_used": n_posterior_draws_used,
        "posterior_draw_mode": posterior_draw_mode,
        "posterior_artifact": posterior_artifact,
        "posterior_draw_tail_cap": int(posterior_draw_tail_cap),
        **predictive_metadata,
        "backend": backend_name,
        "parent_sample_size": int(parent_sample_size),
        "sigma_table_leaf_path": sigma_table_leaf_path,
        "n_parent_sample": int(parent_sample_size),
        "n_mass_bins": int(n_mass_bins),
        "mass_bin_min": float(mass_bin_min),
        "mass_bin_max": float(mass_bin_max),
        "mass_bin_edges": mass_bin_edges.tolist(),
        "mass_bin_centers": mass_bin_centers.tolist(),
        "generator_mode": "numba_shared_parent_binned",
        "mass_definition": mass_definition_metadata(mass_definition),
        "parallel_strategy": backend_name,
        "worker_processes": 0,
        "parallelism": dict(parallelism_payload),
        "layout": "5x1",
        "panel_order": trend_panel_order,
        "figure_title": figure_title,
        "quantities": {name: {"label": name} for name in trend_quantity_names},
        "categories": {
            "parent": {"label": "Parent population"},
            "detectable": {"label": "Detectable lenses"},
            "selected": {"label": "full_selection"},
        },
        "bands": serializable_summary,
    }
    (result_dir / "fig8_like_summary.json").write_text(
        json.dumps(trend_summary_payload, indent=2, sort_keys=True),
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
        for category_name in trend_category_names:
            np_save_payload[f"{category_name}_{quantity_name}_draws"] = trend_draws[quantity_name][category_name]
    np.savez(result_dir / "fig8_like_curves.npz", **np_save_payload)

    observed_gamma_sigma_star_overlay = _build_observed_gamma_sigma_star_overlay(observed_gamma_measurements)
    composite_extra_panels = [
        {
            "panel_id": "gamma_vs_sigma_star",
            "summary_payload": gamma_vs_sigma_star_summary,
            "x_grid": sigma_star_axis_spec.bin_centers,
            "y_label": "gamma",
            "x_label": sigma_star_axis_spec.figure_label or sigma_star_axis_spec.label,
            "observed_overlay": observed_gamma_sigma_star_overlay,
            "observed_label": None,
        },
        {
            "panel_id": "gamma_vs_logre_kpc",
            "summary_payload": gamma_vs_logre_summary,
            "x_grid": log_re_axis_spec.bin_centers,
            "y_label": "gamma",
            "x_label": log_re_axis_spec.figure_label or log_re_axis_spec.label,
            "observed_overlay": _build_observed_gamma_logre_overlay(observed_gamma_measurements),
            "observed_label": None,
        },
    ]
    _write_fig8_like_figure(
        figure_path=result_dir / "fig8_like.png",
        mass_grid=mass_bin_centers,
        summary_payload=trend_summary,
        mass_definition=mass_definition,
        observed_points=_load_observed_trend_points_for_runtime(
            runtime_config=runtime_config,
            observations=observations,
            mass_definition=mass_definition,
        ),
        extra_gamma_panels=composite_extra_panels,
        figure_title=figure_title,
    )

    standalone_base_metadata = {
        "run_id": resolved_run_dir.name,
        "profile_name": runtime_config.profile.name,
        "gamma_mode": _runtime_gamma_mode(runtime_config),
        "parameter_order": list(runtime_config.parameter_schema.public_parameter_names),
        "input_run_dir": str(resolved_run_dir),
        "observation_path": observation_path_metadata,
        "observation_flavor": observation_flavor,
        "result_dir": str(result_dir),
        "burn_in_applied": burn_in_steps,
        "requested_n_posterior_draws": None if n_posterior_draws is None else int(n_posterior_draws),
        "n_posterior_draws": n_posterior_draws_used,
        "n_posterior_draws_used": n_posterior_draws_used,
        "posterior_draw_mode": posterior_draw_mode,
        "posterior_artifact": posterior_artifact,
        "posterior_draw_tail_cap": int(posterior_draw_tail_cap),
        **predictive_metadata,
        "backend": backend_name,
        "parent_sample_size": int(parent_sample_size),
        "sigma_table_path": None if resolved_sigma_table_path is None else str(resolved_sigma_table_path),
        "sigma_table_leaf_path": sigma_table_leaf_path,
        "n_parent_sample": int(parent_sample_size),
        "n_bins": int(n_mass_bins),
        "generator_mode": "numba_shared_parent_binned",
        "mass_definition": mass_definition_metadata(mass_definition),
        "parallel_strategy": backend_name,
        "worker_processes": 0,
        "parallelism": dict(parallelism_payload),
        "categories": trend_summary_payload["categories"],
    }
    _write_standalone_gamma_trend_artifacts(
        result_dir=result_dir,
        artifact_stem="gamma_vs_sigma_star",
        axis_spec=sigma_star_axis_spec,
        gamma_summary=gamma_vs_sigma_star_summary,
        gamma_draws=gamma_vs_sigma_star_draws,
        parent_bin_counts_draws=diagnostics["gamma_vs_sigma_star_parent_bin_counts_draws"],
        detectable_weight_sums_draws=diagnostics["gamma_vs_sigma_star_detectable_weight_sums_draws"],
        selected_weight_sums_draws=diagnostics["gamma_vs_sigma_star_selected_weight_sums_draws"],
        observed_overlay=observed_gamma_sigma_star_overlay,
        observed_overlay_draws=None,
        base_metadata=standalone_base_metadata,
        figure_title=f"{figure_title} | gamma vs log $\\Sigma_*$",
    )
    _write_standalone_gamma_trend_artifacts(
        result_dir=result_dir,
        artifact_stem="gamma_vs_logre_kpc",
        axis_spec=log_re_axis_spec,
        gamma_summary=gamma_vs_logre_summary,
        gamma_draws=gamma_vs_logre_draws,
        parent_bin_counts_draws=diagnostics["gamma_vs_logre_parent_bin_counts_draws"],
        detectable_weight_sums_draws=diagnostics["gamma_vs_logre_detectable_weight_sums_draws"],
        selected_weight_sums_draws=diagnostics["gamma_vs_logre_selected_weight_sums_draws"],
        observed_overlay=composite_extra_panels[1]["observed_overlay"],
        observed_overlay_draws=None,
        base_metadata=standalone_base_metadata,
        figure_title=f"{figure_title} | gamma vs log $r_e$",
    )
    _write_standalone_gamma_trend_artifacts(
        result_dir=result_dir,
        artifact_stem="gamma_vs_delta_r",
        axis_spec=delta_r_axis_spec,
        gamma_summary=gamma_vs_delta_r_summary,
        gamma_draws=gamma_vs_delta_r_draws,
        parent_bin_counts_draws=diagnostics["gamma_vs_delta_r_parent_bin_counts_draws"],
        detectable_weight_sums_draws=diagnostics["gamma_vs_delta_r_detectable_weight_sums_draws"],
        selected_weight_sums_draws=diagnostics["gamma_vs_delta_r_selected_weight_sums_draws"],
        observed_overlay=_build_observed_gamma_delta_r_overlay(
            measurements=observed_gamma_measurements,
            profile=profile_spec,
            axis_spec=delta_r_axis_spec,
        ),
        observed_overlay_draws=None,
        base_metadata=standalone_base_metadata,
        figure_title=f"{figure_title} | gamma vs $\\Delta_R$",
    )

    return PosteriorDiagnosticsResult(
        run_id=resolved_run_dir.name,
        profile_name=runtime_config.profile.name,
        input_run_dir=resolved_run_dir,
        result_dir=result_dir,
        status="completed",
        burn_in_applied=burn_in_steps,
        n_posterior_draws=n_posterior_draws_used,
        parent_sample_size=int(parent_sample_size),
        n_mass_bins=int(n_mass_bins),
        sigma_table_path=resolved_sigma_table_path,
        metadata={
            **predictive_metadata,
            "backend": backend_name,
            "parent_sample_size": int(parent_sample_size),
            "requested_n_posterior_draws": None if n_posterior_draws is None else int(n_posterior_draws),
            "n_posterior_draws_used": n_posterior_draws_used,
            "posterior_draw_mode": posterior_draw_mode,
            "posterior_artifact": posterior_artifact,
            "posterior_draw_tail_cap": int(posterior_draw_tail_cap),
            "observation_path": observation_path_metadata,
            "observation_flavor": observation_flavor,
            "sigma_table_path": None if resolved_sigma_table_path is None else str(resolved_sigma_table_path),
            "sigma_table_leaf_path": sigma_table_leaf_path,
            "gamma_mode": _runtime_gamma_mode(runtime_config),
            "parameter_order": list(runtime_config.parameter_schema.public_parameter_names),
            "mass_definition": mass_definition_metadata(mass_definition),
            "parallelism": dict(parallelism_payload),
            "statistics": ppc_summary_payload["statistics"],
        },
    )


def run_posterior_trends(
    run_dir: str,
    sigma_table_path: str | Path | None = None,
    output_root_dir: str | Path = DEFAULT_PPC_OUTPUT_ROOT_DIR,
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
    """Run standalone trend diagnostics through the shared Numba parent-population backend.

    The historical public API is preserved so existing scripts can keep calling
    ``posterior-trends``. Internally this is now only a compatibility wrapper:
    the full ``run_posterior_diagnostics`` workflow writes both PPC and trend
    artifacts from one Numba parent sample per posterior draw, then this function
    maps the joint result back to the older ``PosteriorTrendResult`` contract.
    """

    diagnostics_result = run_posterior_diagnostics(
        run_dir=run_dir,
        sigma_table_path=sigma_table_path,
        output_root_dir=output_root_dir,
        n_posterior_draws=n_posterior_draws,
        burn_in=burn_in,
        random_seed=random_seed,
        parent_sample_size=n_parent_sample,
        n_mass_bins=n_mass_bins,
        mass_bin_min=mass_bin_min,
        mass_bin_max=mass_bin_max,
        worker_processes=worker_processes,
        posterior_draw_tail_cap=posterior_draw_tail_cap,
    )
    summary_payload = json.loads((diagnostics_result.result_dir / "fig8_like_summary.json").read_text(encoding="utf-8"))
    metadata = dict(diagnostics_result.metadata)
    metadata.update(
        {
            "requested_n_posterior_draws": summary_payload["requested_n_posterior_draws"],
            "n_posterior_draws_used": summary_payload["n_posterior_draws_used"],
            "posterior_draw_mode": summary_payload["posterior_draw_mode"],
            "posterior_draw_tail_cap": summary_payload["posterior_draw_tail_cap"],
            "observation_path": summary_payload["observation_path"],
            "observation_flavor": summary_payload["observation_flavor"],
            "sigma_table_path": (
                None
                if diagnostics_result.sigma_table_path is None
                else str(diagnostics_result.sigma_table_path)
            ),
            "sigma_table_leaf_path": summary_payload["sigma_table_leaf_path"],
            "gamma_mode": summary_payload["gamma_mode"],
            "parameter_order": summary_payload["parameter_order"],
            "n_parent_sample": summary_payload["n_parent_sample"],
            "mass_bin_min": summary_payload["mass_bin_min"],
            "mass_bin_max": summary_payload["mass_bin_max"],
            "n_mass_bins": summary_payload["n_mass_bins"],
            "generator_mode": summary_payload["generator_mode"],
            "mass_definition": summary_payload["mass_definition"],
            "parallel_strategy": summary_payload["parallel_strategy"],
            "worker_processes": summary_payload["worker_processes"],
            "parallelism": summary_payload["parallelism"],
            "figure_title": summary_payload["figure_title"],
            "gamma_vs_sigma_star_figure": str(diagnostics_result.result_dir / "gamma_vs_sigma_star.png"),
            "gamma_vs_sigma_star_summary": str(diagnostics_result.result_dir / "gamma_vs_sigma_star_summary.json"),
            "gamma_vs_sigma_star_curves": str(diagnostics_result.result_dir / "gamma_vs_sigma_star_curves.npz"),
            "gamma_vs_logre_kpc_figure": str(diagnostics_result.result_dir / "gamma_vs_logre_kpc.png"),
            "gamma_vs_logre_kpc_summary": str(diagnostics_result.result_dir / "gamma_vs_logre_kpc_summary.json"),
            "gamma_vs_logre_kpc_curves": str(diagnostics_result.result_dir / "gamma_vs_logre_kpc_curves.npz"),
            "gamma_vs_delta_r_figure": str(diagnostics_result.result_dir / "gamma_vs_delta_r.png"),
            "gamma_vs_delta_r_summary": str(diagnostics_result.result_dir / "gamma_vs_delta_r_summary.json"),
            "gamma_vs_delta_r_curves": str(diagnostics_result.result_dir / "gamma_vs_delta_r_curves.npz"),
        }
    )
    return PosteriorTrendResult(
        run_id=diagnostics_result.run_id,
        profile_name=diagnostics_result.profile_name,
        input_run_dir=diagnostics_result.input_run_dir,
        result_dir=diagnostics_result.result_dir,
        status=diagnostics_result.status,
        burn_in_applied=diagnostics_result.burn_in_applied,
        n_posterior_draws=diagnostics_result.n_posterior_draws,
        n_mass_bins=int(n_mass_bins),
        sigma_table_path=diagnostics_result.sigma_table_path,
        metadata=metadata,
    )


def _discover_unarchived_trend_run_dirs(outputs_root: Path) -> list[tuple[str, Path]]:
    """
    Return all profile run directories whose Fig. 8 products live under `ppc/`.

    The user only wants the active, unarchived runs under `outputs/devauc/*`
    and `outputs/sersic/*`. This helper therefore ignores the `archived`
    branch entirely and only returns directories that already contain the
    trend-array artifact needed for redraw.
    """

    discovered: list[tuple[str, Path]] = []
    for profile_name in ("devauc", "sersic"):
        profile_root = outputs_root / profile_name
        if not profile_root.exists():
            continue
        for run_dir in sorted(profile_root.iterdir()):
            if not run_dir.is_dir() or run_dir.name in {"archived", "latest"}:
                continue
            ppc_dir = run_dir / "ppc"
            if (ppc_dir / "fig8_like_curves.npz").exists() and (ppc_dir / "fig8_like.png").exists():
                discovered.append((profile_name, run_dir))
    return discovered


def _resolve_profile_name_for_annotation_run(run_dir: Path, outputs_root: Path) -> str:
    """
    Infer the profile branch for one explicit run directory.

    Explicit `--run-dir` inputs should still share the same raw-observation
    routing as auto-discovered runs. We resolve the profile from the path under
    the configured outputs root so callers do not need to repeat it manually.
    """

    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_outputs_root = outputs_root.expanduser().resolve()
    try:
        relative_parts = resolved_run_dir.relative_to(resolved_outputs_root).parts
    except ValueError as exc:
        raise ValueError(
            f"Run directory '{resolved_run_dir}' is not located under outputs root '{resolved_outputs_root}'."
        ) from exc
    if len(relative_parts) < 2 or relative_parts[0] not in {"devauc", "sersic"}:
        raise ValueError(
            f"Run directory '{resolved_run_dir}' does not follow the expected '<outputs>/<profile>/<run>' layout."
        )
    return relative_parts[0]


def _resolve_requested_annotation_runs(
    outputs_root: Path,
    run_dirs: list[str] | None,
) -> list[tuple[str, Path]]:
    """
    Resolve either explicit run selections or the legacy auto-discovery set.

    The redraw command originally processed every unarchived run. Supporting
    `--run-dir` should not break that workflow, so explicit run lists simply
    override the discovery step while preserving the same `(profile, run_dir)`
    contract for downstream code.
    """

    if not run_dirs:
        return _discover_unarchived_trend_run_dirs(outputs_root)

    resolved_pairs: list[tuple[str, Path]] = []
    for raw_run_dir in run_dirs:
        run_dir = Path(raw_run_dir).expanduser().resolve()
        if not run_dir.exists() or not run_dir.is_dir():
            raise ValueError(f"Explicit run directory '{run_dir}' does not exist or is not a directory.")
        ppc_dir = run_dir / "ppc"
        required_paths = [
            ppc_dir / "fig8_like_curves.npz",
            ppc_dir / "fig8_like.png",
            ppc_dir / "gamma_vs_logre_kpc_curves.npz",
            ppc_dir / "gamma_vs_sigma_star_curves.npz",
        ]
        if not all(path.exists() for path in required_paths):
            raise ValueError(
                f"Explicit run directory '{run_dir}' is missing one or more required trend artifacts for 5-panel redraw."
            )
        resolved_pairs.append((_resolve_profile_name_for_annotation_run(run_dir, outputs_root), run_dir))
    return resolved_pairs


def _backup_existing_figure(figure_path: Path, backup_prefix: str) -> Path:
    """
    Copy the current PNG aside before overwriting it.

    The redraw workflow is intentionally destructive in-place because the user
    wants the same figure path to keep working. Creating a timestamped backup
    first preserves the old render for audit and comparison.
    """

    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S")
    backup_path = figure_path.with_name(f"{figure_path.stem}.{backup_prefix}.{timestamp}.bak{figure_path.suffix}")
    shutil.copy2(figure_path, backup_path)
    return backup_path


def _resolve_annotation_observation_path(
    profile_name: str,
    run_dir: Path,
    raw_devauc_path: str | Path | None,
    raw_sersic_path: str | Path | None,
) -> Path:
    """Resolve the observation file for annotation from overrides or run config."""

    override_path: str | Path | None
    if profile_name == "devauc":
        override_path = raw_devauc_path
    elif profile_name == "sersic":
        override_path = raw_sersic_path
    else:
        raise ValueError(f"Unsupported annotation profile '{profile_name}'.")

    if override_path is not None:
        return Path(override_path).expanduser().resolve()
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    if not config_snapshot_path.exists():
        fig8_summary_path = run_dir / "ppc" / "fig8_like_summary.json"
        if not fig8_summary_path.exists():
            raise FileNotFoundError(
                f"Cannot resolve observation path for '{run_dir}': missing both "
                f"'{config_snapshot_path}' and '{fig8_summary_path}'."
            )
        fig8_summary = json.loads(fig8_summary_path.read_text(encoding="utf-8"))
        input_run_dir = fig8_summary.get("input_run_dir")
        if not input_run_dir:
            raise ValueError(
                f"Fig. 8 summary '{fig8_summary_path}' does not record `input_run_dir`, "
                "so the annotation command cannot infer the observation file."
            )
        config_snapshot_path = Path(str(input_run_dir)).expanduser().resolve() / "config_snapshot.yaml"
    runtime_config = _load_ppc_runtime_config(config_snapshot_path)
    return Path(runtime_config.data.observation_path).expanduser().resolve()


def _resolve_annotation_runtime_config(
    profile_name: str,
    run_dir: Path,
    raw_devauc_path: str | Path | None,
    raw_sersic_path: str | Path | None,
):
    """
    Load the runtime config used to rebuild observed overlays for one run.

    The annotation command may override the raw observation file path while
    still reusing the run's original scientific configuration. We therefore
    load the stored config snapshot and replace only `data.observation_path`
    when an explicit override is provided.
    """

    config_snapshot_path = run_dir / "config_snapshot.yaml"
    if not config_snapshot_path.exists():
        fig8_summary_path = run_dir / "ppc" / "fig8_like_summary.json"
        if not fig8_summary_path.exists():
            raise FileNotFoundError(
                f"Cannot resolve runtime config for '{run_dir}': missing both "
                f"'{config_snapshot_path}' and '{fig8_summary_path}'."
            )
        fig8_summary = json.loads(fig8_summary_path.read_text(encoding="utf-8"))
        input_run_dir = fig8_summary.get("input_run_dir")
        if not input_run_dir:
            raise ValueError(
                f"Fig. 8 summary '{fig8_summary_path}' does not record `input_run_dir`, "
                "so the annotation command cannot infer the stored runtime config."
            )
        config_snapshot_path = Path(str(input_run_dir)).expanduser().resolve() / "config_snapshot.yaml"

    runtime_config = _load_ppc_runtime_config(config_snapshot_path)
    resolved_observation_path = _resolve_annotation_observation_path(
        profile_name=profile_name,
        run_dir=run_dir,
        raw_devauc_path=raw_devauc_path,
        raw_sersic_path=raw_sersic_path,
    )
    if resolved_observation_path == runtime_config.data.observation_path:
        return runtime_config

    return replace(
        runtime_config,
        data=replace(runtime_config.data, observation_path=resolved_observation_path),
    )


def annotate_existing_fig8_like_figures_with_observations(
    outputs_root: str | Path = DEFAULT_PPC_OUTPUT_ROOT_DIR,
    run_dirs: list[str] | None = None,
    raw_devauc_path: str | Path | None = None,
    raw_sersic_path: str | Path | None = None,
    backup_prefix: str = "pre_observed_points",
) -> Fig8ObservationAnnotationResult:
    """
    Re-render existing Fig. 8-like figures with observed lens points overlaid.

    This post-processing command exists specifically for the production use
    case where trend draws are already saved under `fig8_like_curves.npz`.
    Reusing those arrays avoids rerunning the expensive Monte Carlo trend
    workflow while still letting the user iterate on figure presentation.
    """

    resolved_outputs_root = Path(outputs_root).expanduser().resolve()
    processed_runs: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, Any]] = []

    for profile_name, run_dir in _resolve_requested_annotation_runs(resolved_outputs_root, run_dirs):
        ppc_dir = run_dir / "ppc"
        figure_path = ppc_dir / "fig8_like.png"
        fig8_summary_path = ppc_dir / "fig8_like_summary.json"
        try:
            mass_grid, mass_definition, trend_summary = _load_trend_summary_from_npz(ppc_dir / "fig8_like_curves.npz")
            gamma_vs_logre_grid, gamma_vs_logre_summary = _load_single_quantity_trend_summary_from_npz(
                ppc_dir / "gamma_vs_logre_kpc_curves.npz"
            )
            gamma_vs_sigma_star_grid, gamma_vs_sigma_star_summary = _load_single_quantity_trend_summary_from_npz(
                ppc_dir / "gamma_vs_sigma_star_curves.npz"
            )
            gamma_mode = _resolve_gamma_mode_for_fig8_run(
                run_dir=run_dir,
                fig8_summary_path=fig8_summary_path,
            )
            figure_title = _build_annotated_fig8_title(
                run_dir=run_dir,
                profile_name=profile_name,
                mass_definition=mass_definition,
                gamma_mode=gamma_mode,
            )
            runtime_config = _resolve_annotation_runtime_config(
                profile_name=profile_name,
                run_dir=run_dir,
                raw_devauc_path=raw_devauc_path,
                raw_sersic_path=raw_sersic_path,
            )
            resolved_observation_path = runtime_config.data.observation_path
            observed_points = _load_observed_trend_points(
                observation_path=resolved_observation_path,
                profile_name=profile_name,
                mass_definition=mass_definition,
            )
            compiled_context, profile_spec, _, cosmology, _, observations = build_compiled_context(runtime_config)
            del compiled_context
            observed_gamma_measurements = _load_observed_gamma_measurements(
                observation_path=resolved_observation_path,
                profile_name=profile_name,
                observations=observations,
                cosmology=cosmology,
                mass_definition=mass_definition,
            )
            extra_gamma_panels = [
                {
                    "panel_id": "gamma_vs_sigma_star",
                    "summary_payload": gamma_vs_sigma_star_summary,
                    "x_grid": gamma_vs_sigma_star_grid,
                    "y_label": "gamma",
                    "x_label": r"log $\Sigma_*$ [$M_\odot$ kpc$^{-2}$]",
                    "observed_overlay": _build_observed_gamma_sigma_star_overlay(observed_gamma_measurements),
                    "observed_label": None,
                },
                {
                    "panel_id": "gamma_vs_logre_kpc",
                    "summary_payload": gamma_vs_logre_summary,
                    "x_grid": gamma_vs_logre_grid,
                    "y_label": "gamma",
                    "x_label": _effective_radius_axis_label(mass_definition),
                    "observed_overlay": _build_observed_gamma_logre_overlay(observed_gamma_measurements),
                    "observed_label": None,
                },
            ]
            annotated_display_xlim_by_panel = _build_fixed_fig8_display_xlim_by_panel(
                mass_definition=mass_definition,
                profile_name=profile_name,
            )
            annotated_display_ylim_by_panel = _build_fixed_fig8_display_ylim_by_panel(mass_definition)
            backup_path = _backup_existing_figure(figure_path=figure_path, backup_prefix=backup_prefix)
            _write_fig8_like_figure(
                figure_path=figure_path,
                mass_grid=mass_grid,
                summary_payload=trend_summary,
                mass_definition=mass_definition,
                observed_points=observed_points,
                extra_gamma_panels=extra_gamma_panels,
                figure_title=figure_title,
                display_xlim_by_panel=annotated_display_xlim_by_panel,
                display_ylim_by_panel=annotated_display_ylim_by_panel,
            )
            _update_existing_fig8_summary_metadata(
                fig8_summary_path=fig8_summary_path,
                figure_title=figure_title,
                display_xlim_by_panel=annotated_display_xlim_by_panel,
                display_ylim_by_panel=annotated_display_ylim_by_panel,
            )
            processed_runs.append(
                {
                    "profile_name": profile_name,
                    "run_id": run_dir.name,
                    "figure_path": str(figure_path),
                    "backup_path": str(backup_path),
                    "observation_path": str(resolved_observation_path),
                    "gamma_mode": gamma_mode,
                    "figure_title": figure_title,
                    "display_xlim_by_panel": {
                        quantity_name: [float(axis_limits[0]), float(axis_limits[1])]
                        for quantity_name, axis_limits in annotated_display_xlim_by_panel.items()
                    },
                    "display_ylim_by_panel": {
                        quantity_name: [float(axis_limits[0]), float(axis_limits[1])]
                        for quantity_name, axis_limits in annotated_display_ylim_by_panel.items()
                    },
                    "mass_quantity": mass_definition.label,
                    "observed_mass_points": int(observed_points[mass_definition.label].x.size),
                    "observed_gamma_points": int(observed_points["gamma"].x.size),
                    "observed_sigma_points": int(observed_points["sigma_ap"].x.size),
                }
            )
        except Exception as exc:  # pragma: no cover - exercised through CLI result surface
            skipped_runs.append(
                {
                    "profile_name": profile_name,
                    "run_id": run_dir.name,
                    "figure_path": str(figure_path),
                    "reason": str(exc),
                }
            )

    status = "completed" if processed_runs else "no_runs_processed"
    return Fig8ObservationAnnotationResult(
        status=status,
        outputs_root=resolved_outputs_root,
        processed_run_count=len(processed_runs),
        processed_runs=processed_runs,
        skipped_runs=skipped_runs,
        metadata={
            "raw_devauc_path": None if raw_devauc_path is None else str(Path(raw_devauc_path).expanduser().resolve()),
            "raw_sersic_path": None if raw_sersic_path is None else str(Path(raw_sersic_path).expanduser().resolve()),
            "backup_prefix": backup_prefix,
            "requested_run_dirs": [] if not run_dirs else [str(Path(path).expanduser().resolve()) for path in run_dirs],
        },
    )


def run_posterior_predictive(
    run_dir: str,
    sigma_table_path: str | Path | None = None,
    output_root_dir: str | Path = DEFAULT_PPC_OUTPUT_ROOT_DIR,
    n_replicates: int | None = DEFAULT_N_REPLICATES,
    burn_in: str | int = "auto",
    random_seed: int = DEFAULT_RANDOM_SEED,
    candidate_pool_size: int | None = None,
    worker_processes: int | None = None,
    posterior_draw_tail_cap: int = DEFAULT_CANONICAL_POSTERIOR_DRAW_CAP,
) -> PosteriorPredictiveResult:
    """Run standalone PPC through the shared Numba diagnostics backend.

    This function keeps the historical PPC API stable, including the
    ``candidate_pool_size`` default policy, but no longer owns a separate
    NumPy/SciPy simulation path. The full diagnostics workflow writes the PPC
    artifacts and the trend artifacts from the same Numba parent population; this
    wrapper returns the older ``PosteriorPredictiveResult`` view of that joint
    run.
    """

    resolved_run_dir = Path(run_dir).expanduser().resolve()
    runtime_config = _load_ppc_runtime_config(resolved_run_dir / "config_snapshot.yaml")
    compiled_context, _, _, _, _, _ = _build_ppc_context(runtime_config)
    effective_candidate_pool_size = _resolve_candidate_pool_size(
        candidate_pool_size=candidate_pool_size,
        base_normals_count=int(compiled_context.base_normals.shape[0]),
    )
    diagnostics_result = run_posterior_diagnostics(
        run_dir=run_dir,
        sigma_table_path=sigma_table_path,
        output_root_dir=output_root_dir,
        n_posterior_draws=n_replicates,
        burn_in=burn_in,
        random_seed=random_seed,
        parent_sample_size=effective_candidate_pool_size,
        worker_processes=worker_processes,
        posterior_draw_tail_cap=posterior_draw_tail_cap,
    )

    ppc_summary_path = diagnostics_result.result_dir / "ppc_summary.json"
    ppc_summary_payload = json.loads(ppc_summary_path.read_text(encoding="utf-8"))
    manifest_path = diagnostics_result.result_dir / "run_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    ppc_summary_payload["candidate_pool_size"] = effective_candidate_pool_size
    manifest_payload["candidate_pool_size"] = effective_candidate_pool_size
    ppc_summary_path.write_text(json.dumps(ppc_summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

    metadata = dict(diagnostics_result.metadata)
    metadata.update(
        {
            "requested_n_replicates": ppc_summary_payload["requested_n_replicates"],
            "n_posterior_draws_used": ppc_summary_payload["n_posterior_draws_used"],
            "posterior_draw_mode": ppc_summary_payload["posterior_draw_mode"],
            "candidate_pool_size": effective_candidate_pool_size,
            "normalization_samples": runtime_config.integration.normalization_samples,
            "observation_path": ppc_summary_payload["observation_path"],
            "observation_flavor": ppc_summary_payload["observation_flavor"],
            "sigma_table_leaf_path": ppc_summary_payload["sigma_table_leaf_path"],
            "gamma_mode": ppc_summary_payload["gamma_mode"],
            "parameter_order": ppc_summary_payload["parameter_order"],
            "mass_definition": ppc_summary_payload["mass_definition"],
            "parallelism": ppc_summary_payload["parallelism"],
            "statistics": ppc_summary_payload["statistics"],
        }
    )
    return PosteriorPredictiveResult(
        run_id=diagnostics_result.run_id,
        profile_name=diagnostics_result.profile_name,
        input_run_dir=diagnostics_result.input_run_dir,
        result_dir=diagnostics_result.result_dir,
        status=diagnostics_result.status,
        burn_in_applied=diagnostics_result.burn_in_applied,
        n_replicates=diagnostics_result.n_posterior_draws,
        sample_sizes={"theta_ein": THETA_SAMPLE_SIZE, "sigma": SIGMA_SAMPLE_SIZE},
        sigma_table_path=diagnostics_result.sigma_table_path,
        metadata=metadata,
    )
