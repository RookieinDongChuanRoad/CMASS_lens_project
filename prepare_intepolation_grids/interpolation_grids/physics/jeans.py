"""Velocity-dispersion kernels backed by the external `spherical_jeans` package.

This module serves two related workflows:

1. the historical per-galaxy `s2_grid` path used by the raw observation HDF5
   files; and
2. the new PPT-facing large interpolation tables that expose
   `S_unit = sigma^2 / 10**m5`.

The implementation keeps one shared physical kernel so both workflows use the
same aperture policy, the same unit-mass normalization, and the same tracer
profile definitions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.constants import G, M_sun, kpc
from astropy.cosmology import FlatLambdaCDM
from spherical_jeans import sigma_model, tracer_profiles
from spherical_jeans.mass_profiles import powerlaw

from interpolation_grids.config import (
    DEFAULT_RADIAL_GRID_SIZE,
    DEFAULT_PRODUCTION_APERTURE_POLICY,
    H_UNITS_V1,
    LEGACY_FIXED_KPC,
    OBSERVED_APERTURE_SIGMA_DEFINITION,
    WITHIN_RE_SIGMA_DEFINITION,
)
from interpolation_grids.models import AperturePolicy, GalaxyInputs
from interpolation_grids.unit_conventions import Sunit_hinv_from_fixed_kpc


COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3)
SIGMA2_TO_KM2_PER_S2 = (G * M_sun / kpc).to("km2 / s2").value


def _sigma_unit_mass_scale_factor(gamma_grid: np.ndarray, mass_radius_kpc: float) -> np.ndarray:
    """
    Convert the legacy `S_unit(m5)` normalization into `S_unit(m_R)`.

    The current Jeans kernel normalizes the projected power-law mass at 5 kpc.
    For another radius `R`, the enclosed mass rescales by `(R / 5) ** (3-gamma)`,
    so `sigma^2 / 10**m_R = sigma^2 / 10**m5 * (5 / R) ** (3-gamma)`.
    """

    gamma_array = np.asarray(gamma_grid, dtype=float)
    if float(mass_radius_kpc) == 5.0:
        return np.ones_like(gamma_array, dtype=float)
    return np.power(5.0 / float(mass_radius_kpc), 3.0 - gamma_array)


def kpc_per_arcsec(zd: float) -> float:
    """Convert one arcsec at lens redshift `zd` into physical kpc."""

    return COSMOLOGY.kpc_proper_per_arcmin(zd).value / 60.0


def uses_devaucouleurs_branch(source_filename: str) -> bool:
    """Return whether a file belongs to the fixed deV tracer branch.

    Why this helper still exists:
    - the historical HDF5 readers do not carry an explicit top-level profile
      tag, so the per-galaxy `s2_grid` builder still needs one piece of file
      context to choose between the fixed deV tracer and the free-Sersic
      tracer.
    - the public canonical filename migrated from `with_m5_grids` to the
      definition-agnostic `with_mass_grids`, but legacy files remain readable.

    The predicate therefore recognizes both canonical deV filenames while
    remaining conservative for everything else.
    """

    normalized_name = Path(source_filename).name
    return normalized_name.startswith("observations_deV_with_") and normalized_name.endswith("_grids.hdf5")


def _build_aperture_and_seeing_kpc(
    zd: float,
    re_kpc: float | None = None,
    aperture_policy: AperturePolicy | None = None,
    sigma_definition: str = OBSERVED_APERTURE_SIGMA_DEFINITION,
) -> tuple[float | list[float], float]:
    """Convert one explicit aperture policy from arcsec into physical kpc.

    The caller may override the project default when a workflow needs a
    different physical aperture without mutating the behavior of every Jeans
    calculation in the repository.
    """

    normalized_sigma_definition = sigma_definition.strip().lower()
    if normalized_sigma_definition == WITHIN_RE_SIGMA_DEFINITION:
        if re_kpc is None or re_kpc <= 0.0:
            raise ValueError("The within-Re sigma definition requires a strictly positive `re_kpc`.")
        return float(re_kpc), None
    if normalized_sigma_definition != OBSERVED_APERTURE_SIGMA_DEFINITION:
        raise ValueError(f"Unsupported sigma definition: {sigma_definition}")

    physical_kpc_per_arcsec = kpc_per_arcsec(zd)
    resolved_policy = aperture_policy or DEFAULT_PRODUCTION_APERTURE_POLICY

    if resolved_policy.shape == "circular":
        aperture_kpc: float | list[float] = float(resolved_policy.radius_arcsec * physical_kpc_per_arcsec)
    else:
        # The upstream Jeans package interprets a length-2 list as a centered
        # rectangular aperture ordered as [height, width]. We intentionally
        # keep that ordering here so the new typed policy remains compatible
        # with the external solver's API.
        aperture_kpc = [
            float(resolved_policy.height_arcsec * physical_kpc_per_arcsec),
            float(resolved_policy.width_arcsec * physical_kpc_per_arcsec),
        ]
    seeing_kpc = float(resolved_policy.seeing_fwhm_arcsec * physical_kpc_per_arcsec)
    return aperture_kpc, seeing_kpc


def _resolve_tracer_setup(
    profile_name: str,
    re_kpc: float,
    n_value: float | None = None,
) -> tuple[float | tuple[float, float], object, np.ndarray]:
    """Return tracer parameters, tracer profile, and the Jeans radial grid."""

    normalized_profile = profile_name.strip().lower()
    if normalized_profile == "devauc":
        tracer_parameters: float | tuple[float, float] = re_kpc
        tracer_profile = tracer_profiles.deVaucouleurs
        radial_anchor_kpc = re_kpc
    elif normalized_profile == "sersic":
        if n_value is None:
            raise ValueError("Sersic sigma-unit evaluation requires `n_value`.")
        tracer_parameters = (re_kpc, n_value)
        tracer_profile = tracer_profiles.sersic
        radial_anchor_kpc = re_kpc
    else:
        raise ValueError(f"Unsupported tracer profile: {profile_name}")

    radial_grid = np.logspace(
        np.log10(radial_anchor_kpc) - 3.0,
        np.log10(radial_anchor_kpc) + 3.0,
        DEFAULT_RADIAL_GRID_SIZE,
    )
    return tracer_parameters, tracer_profile, radial_grid


def _compute_sigma_unit_values_for_prepared_inputs(
    gamma_grid: np.ndarray,
    aperture_kpc: float | list[float],
    seeing_kpc: float | None,
    tracer_parameters: float | tuple[float, float],
    tracer_profile: object,
    radial_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate one gamma vector after geometry/tracer setup is fixed.

    This helper exists because the full-table workflow evaluates many gamma
    vectors at the same `(zd, re_kpc[, n])` coordinate. Rebuilding the
    aperture conversion and tracer grid inside every gamma point would waste a
    measurable amount of CPU time when the machine is already saturated by the
    expensive Jeans integrations.
    """

    output = np.zeros_like(np.asarray(gamma_grid, dtype=float), dtype=float)
    for index, gamma in enumerate(np.asarray(gamma_grid, dtype=float)):
        normalization = 1.0 / powerlaw.M2d(5.0, gamma)
        enclosed_mass_grid = normalization * powerlaw.M3d(radial_grid, gamma)
        sigma2_kwargs = {}
        if seeing_kpc is not None:
            sigma2_kwargs["seeing"] = seeing_kpc
        sigma2_over_g = sigma_model.sigma2(
            (radial_grid, enclosed_mass_grid),
            aperture_kpc,
            tracer_parameters,
            tracer_profile,
            **sigma2_kwargs,
        )
        output[index] = sigma2_over_g * SIGMA2_TO_KM2_PER_S2
    return output


def compute_sigma_unit(
    profile_name: str,
    gamma: float,
    zd: float,
    re_kpc: float,
    n_value: float | None = None,
    mass_radius_kpc: float = 5.0,
    aperture_policy: AperturePolicy | None = None,
    sigma_definition: str = OBSERVED_APERTURE_SIGMA_DEFINITION,
    unit_convention: str = LEGACY_FIXED_KPC,
    h_ref: float = 0.7,
) -> float:
    """Evaluate the unit-mass Jeans response for one physical lens coordinate.

    Why this function exists:
    - the old `compute_s2_grid()` entrypoint is tied to one real observed
      galaxy and selects the tracer branch from the input filename
    - the PPT large-table workflow needs a direct physical kernel that can be
      reused for arbitrary replicated lenses without dragging file metadata into
      the numerical layer

    The returned quantity is `S_unit = sigma^2 / 10**m5` in `km^2 / s^2` per
    unit `10**m5` normalization.
    """

    aperture_kpc, seeing_kpc = _build_aperture_and_seeing_kpc(
        zd,
        re_kpc=re_kpc,
        aperture_policy=aperture_policy,
        sigma_definition=sigma_definition,
    )
    tracer_parameters, tracer_profile, radial_grid = _resolve_tracer_setup(
        profile_name=profile_name,
        re_kpc=re_kpc,
        n_value=n_value,
    )

    values = _compute_sigma_unit_values_for_prepared_inputs(
        gamma_grid=np.asarray([gamma], dtype=float),
        aperture_kpc=aperture_kpc,
        seeing_kpc=seeing_kpc,
        tracer_parameters=tracer_parameters,
        tracer_profile=tracer_profile,
        radial_grid=radial_grid,
    )
    legacy_value = values[0] * _sigma_unit_mass_scale_factor(np.asarray([gamma], dtype=float), mass_radius_kpc)[0]
    if str(unit_convention).strip() == H_UNITS_V1:
        return float(Sunit_hinv_from_fixed_kpc(legacy_value, gamma, h_ref=h_ref))
    if str(unit_convention).strip() != LEGACY_FIXED_KPC:
        raise ValueError(f"Unsupported unit_convention for sigma-unit calculation: {unit_convention}")
    return float(legacy_value)


def compute_sigma_unit_grid(
    profile_name: str,
    gamma_grid: np.ndarray,
    zd: float,
    re_kpc: float,
    n_value: float | None = None,
    mass_radius_kpc: float = 5.0,
    aperture_policy: AperturePolicy | None = None,
    sigma_definition: str = OBSERVED_APERTURE_SIGMA_DEFINITION,
    unit_convention: str = LEGACY_FIXED_KPC,
    h_ref: float = 0.7,
) -> np.ndarray:
    """Evaluate `S_unit` over one gamma axis for fixed non-gamma lens inputs."""

    aperture_kpc, seeing_kpc = _build_aperture_and_seeing_kpc(
        zd,
        re_kpc=re_kpc,
        aperture_policy=aperture_policy,
        sigma_definition=sigma_definition,
    )
    tracer_parameters, tracer_profile, radial_grid = _resolve_tracer_setup(
        profile_name=profile_name,
        re_kpc=re_kpc,
        n_value=n_value,
    )
    base_values = _compute_sigma_unit_values_for_prepared_inputs(
        gamma_grid=np.asarray(gamma_grid, dtype=float),
        aperture_kpc=aperture_kpc,
        seeing_kpc=seeing_kpc,
        tracer_parameters=tracer_parameters,
        tracer_profile=tracer_profile,
        radial_grid=radial_grid,
    )
    legacy_values = base_values * _sigma_unit_mass_scale_factor(np.asarray(gamma_grid, dtype=float), mass_radius_kpc)
    normalized_convention = str(unit_convention).strip()
    if normalized_convention == H_UNITS_V1:
        return Sunit_hinv_from_fixed_kpc(legacy_values, np.asarray(gamma_grid, dtype=float), h_ref=h_ref)
    if normalized_convention != LEGACY_FIXED_KPC:
        raise ValueError(f"Unsupported unit_convention for sigma-unit grid calculation: {unit_convention}")
    return legacy_values


def compute_s2_grid(
    galaxy: GalaxyInputs,
    gamma_grid: np.ndarray,
    mass_radius_kpc: float = 5.0,
    aperture_policy: AperturePolicy | None = None,
    unit_convention: str = LEGACY_FIXED_KPC,
    h_ref: float = 0.7,
) -> np.ndarray:
    """Compute the per-galaxy `s2_grid` for the requested mass definition.

    Parameters
    ----------
    galaxy:
        Structured galaxy inputs extracted from one HDF5 group.
    gamma_grid:
        Density-slope samples at which the grid is evaluated.

    Returns
    -------
    np.ndarray
        `S_unit` values in `km^2 / s^2` per unit `10**m5`, one per gamma
        sample. The historical dataset name remains `s2_grid` because that is
        the raw-observation file contract, but physically this quantity is the
        same unit-mass Jeans response used by the new PPT interpolation tables.
    """

    physical_kpc_per_arcsec = kpc_per_arcsec(galaxy.zd)

    if uses_devaucouleurs_branch(galaxy.source_filename):
        if galaxy.reff_dev_arcsec is None:
            raise ValueError(f"{galaxy.group_name} is missing reff_deV required for the deV branch")
        return compute_sigma_unit_grid(
            profile_name="devauc",
            gamma_grid=np.asarray(gamma_grid, dtype=float),
            zd=galaxy.zd,
            re_kpc=galaxy.reff_dev_arcsec * physical_kpc_per_arcsec,
            mass_radius_kpc=mass_radius_kpc,
            aperture_policy=aperture_policy,
            unit_convention=unit_convention,
            h_ref=h_ref,
        )
    else:
        if galaxy.re_arcsec is None or galaxy.nser is None:
            raise ValueError(f"{galaxy.group_name} is missing re_arcsec or nser for the Sersic branch")
        return compute_sigma_unit_grid(
            profile_name="sersic",
            gamma_grid=np.asarray(gamma_grid, dtype=float),
            zd=galaxy.zd,
            re_kpc=galaxy.re_arcsec * physical_kpc_per_arcsec,
            n_value=galaxy.nser,
            mass_radius_kpc=mass_radius_kpc,
            aperture_policy=aperture_policy,
            unit_convention=unit_convention,
            h_ref=h_ref,
        )
