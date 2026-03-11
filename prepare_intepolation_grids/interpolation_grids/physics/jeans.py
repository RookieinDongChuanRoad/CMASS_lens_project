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
    APERTURE_HEIGHT_ARCSEC,
    DEFAULT_APERTURE_WIDTH_ARCSEC,
    DEFAULT_RADIAL_GRID_SIZE,
    SEEING_FWHM_ARCSEC,
)
from interpolation_grids.models import GalaxyInputs


COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3)
SIGMA2_TO_KM2_PER_S2 = (G * M_sun / kpc).to("km2 / s2").value


def kpc_per_arcsec(zd: float) -> float:
    """Convert one arcsec at lens redshift `zd` into physical kpc."""

    return COSMOLOGY.kpc_proper_per_arcmin(zd).value / 60.0


def uses_devaucouleurs_branch(source_filename: str) -> bool:
    """Return whether a file belongs to the fixed deV tracer branch."""

    return "observations_deV_with_m5_grids" in Path(source_filename).name


def _build_aperture_and_seeing_kpc(zd: float) -> tuple[list[float], float]:
    """Convert the fixed production aperture and seeing into physical kpc."""

    physical_kpc_per_arcsec = kpc_per_arcsec(zd)

    # The upstream Jeans package interprets a length-2 list as a centered
    # rectangular aperture ordered as [height, width]. We intentionally keep
    # that ordering while applying the project-wide fixed-width production
    # policy requested for both per-galaxy and PPT sigma workflows.
    aperture_kpc = [
        APERTURE_HEIGHT_ARCSEC * physical_kpc_per_arcsec,
        DEFAULT_APERTURE_WIDTH_ARCSEC * physical_kpc_per_arcsec,
    ]
    seeing_kpc = SEEING_FWHM_ARCSEC * physical_kpc_per_arcsec
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
    aperture_kpc: list[float],
    seeing_kpc: float,
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
        sigma2_over_g = sigma_model.sigma2(
            (radial_grid, enclosed_mass_grid),
            aperture_kpc,
            tracer_parameters,
            tracer_profile,
            seeing=seeing_kpc,
        )
        output[index] = sigma2_over_g * SIGMA2_TO_KM2_PER_S2
    return output


def compute_sigma_unit(
    profile_name: str,
    gamma: float,
    zd: float,
    re_kpc: float,
    n_value: float | None = None,
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

    aperture_kpc, seeing_kpc = _build_aperture_and_seeing_kpc(zd)
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
    return float(values[0])


def compute_sigma_unit_grid(
    profile_name: str,
    gamma_grid: np.ndarray,
    zd: float,
    re_kpc: float,
    n_value: float | None = None,
) -> np.ndarray:
    """Evaluate `S_unit` over one gamma axis for fixed non-gamma lens inputs."""

    aperture_kpc, seeing_kpc = _build_aperture_and_seeing_kpc(zd)
    tracer_parameters, tracer_profile, radial_grid = _resolve_tracer_setup(
        profile_name=profile_name,
        re_kpc=re_kpc,
        n_value=n_value,
    )
    return _compute_sigma_unit_values_for_prepared_inputs(
        gamma_grid=np.asarray(gamma_grid, dtype=float),
        aperture_kpc=aperture_kpc,
        seeing_kpc=seeing_kpc,
        tracer_parameters=tracer_parameters,
        tracer_profile=tracer_profile,
        radial_grid=radial_grid,
    )


def compute_s2_grid(galaxy: GalaxyInputs, gamma_grid: np.ndarray) -> np.ndarray:
    """Compute the legacy per-galaxy `s2_grid` from the shared sigma-unit kernel.

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
        )
