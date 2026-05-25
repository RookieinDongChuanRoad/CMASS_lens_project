"""Power-law lensing cross-section generators.

This module replaces two historical standalone scripts with maintainable,
tested package code:

- ``make_lenscrosect_grid.py`` from the local CMASS workflow, which tabulates a
  scale-free ``beta_max / theta_E`` lookup for axisymmetric power-law lenses.
- Sonnenfeld et al.'s ``make_crosssect_grid.py`` from
  ``astrosonnen/strong_lensing_tools``, which tabulates finite-fibre,
  seeing-convolved, flux-thresholded source-plane cross-sections.

The two products intentionally have different physical meanings.  The CMASS
``cs_grid`` stores a source-plane radius ``beta_max``.  The Sonnenfeld
``mufibre*_cs_grid`` datasets store an already integrated area.  Keeping both
generators in one module lets them share the same lensing primitives while
making that convention difference explicit at the type boundary.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import math
import sys
from types import TracebackType

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import splint, splrep
from scipy.optimize import brentq, minimize


DEFAULT_GAMMA_MIN = 1.2
DEFAULT_GAMMA_MAX = 2.8
DEFAULT_GAMMA_POINTS = 81
DEFAULT_CMASS_THETA_E_AXIS = np.linspace(0.1, 5.0, 51, dtype=float)
DEFAULT_CMASS_GAMMA_AXIS = np.linspace(DEFAULT_GAMMA_MIN, DEFAULT_GAMMA_MAX, DEFAULT_GAMMA_POINTS, dtype=float)
DEFAULT_FIBRE_THETA_E_AXIS = np.linspace(0.0, 5.0, 51, dtype=float)
DEFAULT_FIBRE_GAMMA_AXIS = DEFAULT_CMASS_GAMMA_AXIS.copy()
DEFAULT_FIBRE_ARCSEC = 1.5
DEFAULT_SEEING_ARCSEC = 1.5
DEFAULT_MUB_MIN = 1.0
DEFAULT_FIBRE_BETA_POINTS = 1001
DEFAULT_FIBRE_RADIAL_POINTS = 16
SONNENFELD_CROSS_SECTION_REFERENCE_URL = (
    "https://raw.githubusercontent.com/astrosonnen/strong_lensing_tools/main/"
    "papers/slacs_selection/scripts/make_crosssect_grid.py"
)
SONNENFELD_CROSS_SECTION_REFERENCE_SHA = "25873a873a5ecbd61b272e61f1da9a62edada7b5"


@dataclass(frozen=True)
class PowerLawCrossSectionGrid:
    """Legacy CMASS power-law cross-section product.

    Attributes
    ----------
    gamma_axis:
        Density-slope axis.  The legacy HDF5 names this ``gamma_grids``.
    theta_e_axis:
        Einstein-radius axis in arcsec.  The legacy HDF5 names this
        ``theta_ein_grids``.
    cs_grid:
        Source-plane detection radius ``beta_max`` for each
        ``(gamma, theta_E)`` pair.  Shape is ``(N_gamma, N_theta_E)`` to match
        the historical file.
    cs_over_theta_ein_grid:
        Compressed scale-free lookup ``beta_max / theta_E`` as a function of
        gamma.  Downstream code converts it to an area through
        ``pi * (ratio * theta_E)**2``.
    """

    gamma_axis: np.ndarray
    theta_e_axis: np.ndarray
    cs_grid: np.ndarray
    cs_over_theta_ein_grid: np.ndarray


@dataclass(frozen=True)
class FibreCrossSectionGrid:
    """Sonnenfeld finite-fibre cross-section product.

    The area grids use axis order ``(N_theta_E, N_gamma)``, exactly as the
    reference ``make_crosssect_grid.py`` writes them.
    """

    theta_e_axis: np.ndarray
    gamma_axis: np.ndarray
    mufibre2_cs_grid: np.ndarray
    mufibre3_cs_grid: np.ndarray
    ycaust_grid: np.ndarray
    fibre_arcsec: float
    seeing_arcsec: float
    muB_min: float
    beta_points: int
    radial_points: int


class _NullProgress(AbstractContextManager["_NullProgress"]):
    """No-op progress reporter used by library callers and unit tests by default."""

    def update(self, count: int) -> None:
        """Accept progress increments without producing terminal output."""

        _ = int(count)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Return ``None`` so exceptions from the numerical path propagate."""

        _ = (exc_type, exc_value, traceback)
        return None


class _FallbackProgress(AbstractContextManager["_FallbackProgress"]):
    """
    Minimal stderr progress reporter for environments without ``tqdm``.

    The finite-fibre grid can be expensive, so silently losing all feedback when
    ``tqdm`` is absent would be a bad operational failure mode.  This fallback
    prints one line per completed theta row plus the final row, which gives a
    stable pair count without adding another required progress-reporting dependency.
    """

    def __init__(self, *, total: int, gamma_count: int, description: str) -> None:
        self.total = max(0, int(total))
        self.gamma_count = max(1, int(gamma_count))
        self.description = str(description)
        self.completed = 0

    def __enter__(self) -> "_FallbackProgress":
        print(f"{self.description}: pairs 0/{self.total}", file=sys.stderr)
        return self

    def update(self, count: int) -> None:
        """Record completed pair work and emit coarse row-level progress."""

        previous = self.completed
        self.completed = min(self.total, self.completed + max(0, int(count)))
        crossed_row_boundary = previous // self.gamma_count != self.completed // self.gamma_count
        if crossed_row_boundary or self.completed == self.total:
            theta_done = math.ceil(self.completed / self.gamma_count) if self.total else 0
            theta_total = math.ceil(self.total / self.gamma_count) if self.total else 0
            print(
                f"{self.description}: theta {theta_done}/{theta_total}, "
                f"pairs {self.completed}/{self.total}",
                file=sys.stderr,
            )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Return ``None`` so exceptions from the numerical path propagate."""

        _ = (exc_type, exc_value, traceback)
        return None


def _open_pair_progress(
    *,
    enabled: bool,
    total: int,
    gamma_count: int,
    description: str,
) -> AbstractContextManager:
    """
    Return a progress context manager without making ``tqdm`` mandatory.

    ``tqdm`` gives the best ETA/rate display for interactive runs.  The fallback
    keeps long-running data preparation observable even in minimal environments
    such as the package-local ``environment.yml``.
    """

    if not enabled:
        return _NullProgress()

    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return _FallbackProgress(total=total, gamma_count=gamma_count, description=description)

    return tqdm(
        total=int(total),
        desc=description,
        unit="pair",
        dynamic_ncols=True,
    )


def _as_1d_axis(values: np.ndarray, *, name: str) -> np.ndarray:
    """Validate and normalize one numeric axis.

    The generators are expensive enough that failing early on malformed axes is
    worth the small amount of validation code.  Monotonic axes also make HDF5
    outputs easier to reason about for interpolation consumers.
    """

    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} contains non-finite values.")
    if axis.size > 1 and not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return axis


def power_law_alpha(x: float | np.ndarray, theta_e: float, gamma: float) -> float | np.ndarray:
    """Return the axisymmetric power-law deflection angle.

    This is the formula used by both historical scripts:

    ``alpha(x) = theta_E * sign(x) * (abs(x) / theta_E)**(2 - gamma)``.

    ``x=0`` is not a valid evaluation point for the lens equation and callers
    avoid it.  The function still uses NumPy operations so scalar and array
    callers share one implementation.
    """

    x_array = np.asarray(x, dtype=float)
    theta_e_value = float(theta_e)
    gamma_value = float(gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = theta_e_value * np.sign(x_array) * np.power(np.abs(x_array) / theta_e_value, 2.0 - gamma_value)
    if np.ndim(x) == 0:
        return float(value)
    return value


def power_law_kappa(x: float | np.ndarray, theta_e: float, gamma: float) -> float | np.ndarray:
    """Return the convergence for the Einstein-radius-normalized power law."""

    x_array = np.asarray(x, dtype=float)
    theta_e_value = float(theta_e)
    gamma_value = float(gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (3.0 - gamma_value) / 2.0 * np.power(np.abs(x_array) / theta_e_value, 1.0 - gamma_value)
    if np.ndim(x) == 0:
        return float(value)
    return value


def radial_magnification(x: float | np.ndarray, theta_e: float, gamma: float) -> float | np.ndarray:
    """Return radial magnification for the power-law lens."""

    x_array = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1.0 + power_law_alpha(x_array, theta_e, gamma) / x_array - 2.0 * power_law_kappa(x_array, theta_e, gamma)
        value = np.power(inverse, -1.0)
    if np.ndim(x) == 0:
        return float(value)
    return value


def tangential_magnification(x: float | np.ndarray, theta_e: float, gamma: float) -> float | np.ndarray:
    """Return tangential magnification for the power-law lens."""

    x_array = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1.0 - power_law_alpha(x_array, theta_e, gamma) / x_array
        value = np.power(inverse, -1.0)
    if np.ndim(x) == 0:
        return float(value)
    return value


def total_magnification(x: float | np.ndarray, theta_e: float, gamma: float) -> float | np.ndarray:
    """Return signed total magnification ``mu_r * mu_t``."""

    return radial_magnification(x, theta_e, gamma) * tangential_magnification(x, theta_e, gamma)


def _cmass_has_detectable_images(beta: float, gamma: float, theta_e: float) -> bool:
    """Return whether one CMASS source position satisfies the legacy image cut.

    The local historical script solves both images on positive radial
    coordinates and then requires both image magnifications to exceed unity in
    absolute value.  Exceptions mean the image configuration is invalid and
    therefore outside the cross-section.
    """

    def positive_image_zero(theta: float) -> float:
        return theta - power_law_alpha(theta, theta_e, gamma) - beta

    def negative_image_zero(theta: float) -> float:
        return theta - power_law_alpha(theta, theta_e, gamma) + beta

    try:
        if gamma >= 2.0:
            theta_pos = brentq(positive_image_zero, theta_e, 100.0)
            try:
                theta_neg = brentq(negative_image_zero, 0.01, theta_e)
            except ValueError:
                theta_neg = brentq(negative_image_zero, 1.0e-4, theta_e)
        else:
            result = minimize(
                lambda theta: theta[0] - power_law_alpha(theta[0], theta_e, gamma),
                np.asarray([0.1], dtype=float),
                method="Nelder-Mead",
                tol=1.0e-10,
            )
            minimum_beta_function = float(result.fun)
            theta_caustic = float(result.x[0])
            if -float(beta) <= minimum_beta_function:
                return False
            theta_pos = brentq(positive_image_zero, theta_e, 1000.0)
            theta_neg = brentq(negative_image_zero, theta_caustic, theta_e)

        mu_pos = total_magnification(theta_pos, theta_e, gamma)
        mu_neg = total_magnification(theta_neg, theta_e, gamma)
        return bool(abs(mu_pos) >= 1.0 and abs(mu_neg) >= 1.0)
    except Exception:  # noqa: BLE001 - the legacy script treats all solver failures as non-lenses.
        return False


def _cmass_beta_max(gamma: float, theta_e: float, *, binary_iterations: int = 30) -> float:
    """Search the largest source-plane radius accepted by the CMASS cut."""

    if theta_e <= 0.0:
        return 0.0

    if gamma < 2.0:
        result = minimize(
            lambda theta: theta[0] - power_law_alpha(theta[0], theta_e, gamma),
            np.asarray([0.1], dtype=float),
            method="Nelder-Mead",
            tol=1.0e-10,
        )
        beta_low = 0.0
        beta_high = -float(result.fun)
    else:
        beta_low = 0.0
        beta_high = 0.001
        while _cmass_has_detectable_images(beta_high, gamma, theta_e):
            beta_high *= 2.0

    for _ in range(int(binary_iterations)):
        beta_mid = 0.5 * (beta_low + beta_high)
        if _cmass_has_detectable_images(beta_mid, gamma, theta_e):
            beta_low = beta_mid
        else:
            beta_high = beta_mid
    return float(beta_low)


def compute_power_law_cross_section_grid(
    *,
    gamma_axis: np.ndarray | None = None,
    theta_e_axis: np.ndarray | None = None,
    binary_iterations: int = 30,
) -> PowerLawCrossSectionGrid:
    """Compute the legacy CMASS separable power-law cross-section table."""

    gamma_values = _as_1d_axis(DEFAULT_CMASS_GAMMA_AXIS if gamma_axis is None else gamma_axis, name="gamma_axis")
    theta_values = _as_1d_axis(DEFAULT_CMASS_THETA_E_AXIS if theta_e_axis is None else theta_e_axis, name="theta_e_axis")
    if np.any(theta_values <= 0.0):
        raise ValueError("The legacy CMASS theta_e_axis must be strictly positive.")

    cs_grid = np.zeros((gamma_values.size, theta_values.size), dtype=float)
    for gamma_index, gamma in enumerate(gamma_values):
        for theta_index, theta_e in enumerate(theta_values):
            cs_grid[gamma_index, theta_index] = _cmass_beta_max(
                float(gamma),
                float(theta_e),
                binary_iterations=binary_iterations,
            )

    # The power-law lens is scale free, so the historical file kept only one
    # ratio per gamma.  We deliberately use the first theta_E column to match
    # the old script exactly.
    cs_over_theta = (cs_grid / theta_values[None, :])[:, 0]
    return PowerLawCrossSectionGrid(
        gamma_axis=gamma_values,
        theta_e_axis=theta_values,
        cs_grid=cs_grid,
        cs_over_theta_ein_grid=cs_over_theta,
    )


def power_law_radial_caustic(theta_e: float, gamma: float, *, xmin: float = 0.01) -> tuple[float, float]:
    """Return ``(ycaust, xradcrit)`` using Sonnenfeld's reference rule."""

    theta_e_value = float(theta_e)
    if theta_e_value <= 0.0:
        return 0.0, 0.0

    def radial_inverse_magnification(x: float) -> float:
        return 1.0 + power_law_alpha(x, theta_e_value, gamma) / x - 2.0 * power_law_kappa(x, theta_e_value, gamma)

    if radial_inverse_magnification(xmin) * radial_inverse_magnification(theta_e_value) > 0.0:
        xradcrit = float(xmin)
    else:
        xradcrit = float(brentq(radial_inverse_magnification, xmin, theta_e_value))
    ycaust = -(xradcrit - power_law_alpha(xradcrit, theta_e_value, gamma))
    return float(ycaust), xradcrit


def _seeing_convolved_image_flux(
    *,
    image_position: float,
    theta_e: float,
    gamma: float,
    fibre_arcsec: float,
    psf_sigma: float,
    radial_points: int,
) -> float:
    """Integrate one image's magnified point-source flux through the fibre.

    This intentionally follows the Sonnenfeld reference script rather than a
    closed-form approximation.  For every fibre radius sample it integrates the
    Gaussian PSF over azimuth with ``quad``, spline-interpolates the radial
    integrand, and integrates the spline over the circular fibre.
    """

    r_grid = np.linspace(0.0, float(fibre_arcsec), int(radial_points), dtype=float)
    abs_mu_r = abs(radial_magnification(image_position, theta_e, gamma))
    abs_mu_t = abs(tangential_magnification(image_position, theta_e, gamma))

    def angular_integrand(radius: float, phi: float) -> float:
        dx = radius * math.cos(phi) - image_position
        dy = radius * math.sin(phi)
        return (
            1.0
            / (2.0 * math.pi)
            / psf_sigma**2
            * math.exp(-0.5 * (dx * dx + dy * dy) / psf_sigma**2)
            * abs_mu_r
        )

    radial_integrand = np.zeros(radial_points, dtype=float)
    for index, radius in enumerate(r_grid):
        angular_value = quad(lambda phi: angular_integrand(float(radius), phi), 0.0, 2.0 * math.pi)[0]
        radial_integrand[index] = float(radius) * angular_value * abs_mu_t

    spline = splrep(r_grid, radial_integrand)
    return float(splint(0.0, r_grid[-1], spline))


def _fibre_cross_section_for_pair(
    *,
    theta_e: float,
    gamma: float,
    fibre_arcsec: float,
    seeing_arcsec: float,
    muB_min: float,
    beta_points: int,
    radial_points: int,
) -> tuple[float, float, float]:
    """Compute ``(ycaust, mufibre2_area, mufibre3_area)`` for one grid point."""

    if theta_e <= 0.0:
        return 0.0, 0.0, 0.0
    ycaust, xradcrit = power_law_radial_caustic(theta_e, gamma)
    x_max = 5.0 * float(fibre_arcsec)
    psf_sigma = float(seeing_arcsec) / 2.35

    xB_grid = np.linspace(-float(theta_e), -xradcrit, int(beta_points), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        muB_grid = np.abs(radial_magnification(xB_grid, theta_e, gamma) * tangential_magnification(xB_grid, theta_e, gamma))
    beta_grid = xB_grid - power_law_alpha(xB_grid, theta_e, gamma)
    muA_seeing_grid = np.zeros_like(beta_grid)
    muB_seeing_grid = np.zeros_like(beta_grid)

    for beta_index in range(1, int(beta_points)):
        beta_value = float(beta_grid[beta_index])

        def xA_zero_function(xA: float) -> float:
            return xA - power_law_alpha(xA, theta_e, gamma) - beta_value

        if xA_zero_function(x_max) >= 0.0:
            xA_here = float(brentq(xA_zero_function, theta_e, x_max))
            muA_seeing_grid[beta_index] = _seeing_convolved_image_flux(
                image_position=xA_here,
                theta_e=theta_e,
                gamma=gamma,
                fibre_arcsec=fibre_arcsec,
                psf_sigma=psf_sigma,
                radial_points=radial_points,
            )
            muB_seeing_grid[beta_index] = _seeing_convolved_image_flux(
                image_position=float(xB_grid[beta_index]),
                theta_e=theta_e,
                gamma=gamma,
                fibre_arcsec=fibre_arcsec,
                psf_sigma=psf_sigma,
                radial_points=radial_points,
            )

    total_seeing_magnification = muA_seeing_grid + muB_seeing_grid
    outputs: list[float] = []
    for threshold in (2.0, 3.0):
        integrand = 2.0 * math.pi * beta_grid.copy()
        invalid = (total_seeing_magnification <= threshold) | (muB_grid <= float(muB_min))
        integrand[invalid] = 0.0
        spline = splrep(beta_grid, integrand, k=1)
        outputs.append(float(splint(beta_grid[0], beta_grid[-1], spline)))

    return float(ycaust), outputs[0], outputs[1]


def compute_fibre_cross_section_grid(
    *,
    gamma_axis: np.ndarray | None = None,
    theta_e_axis: np.ndarray | None = None,
    fibre_arcsec: float = DEFAULT_FIBRE_ARCSEC,
    seeing_arcsec: float = DEFAULT_SEEING_ARCSEC,
    muB_min: float = DEFAULT_MUB_MIN,
    beta_points: int = DEFAULT_FIBRE_BETA_POINTS,
    radial_points: int = DEFAULT_FIBRE_RADIAL_POINTS,
    progress: bool = False,
) -> FibreCrossSectionGrid:
    """Compute the Sonnenfeld finite-fibre cross-section table."""

    gamma_values = _as_1d_axis(DEFAULT_FIBRE_GAMMA_AXIS if gamma_axis is None else gamma_axis, name="gamma_axis")
    theta_values = _as_1d_axis(DEFAULT_FIBRE_THETA_E_AXIS if theta_e_axis is None else theta_e_axis, name="theta_e_axis")
    if np.any(theta_values < 0.0):
        raise ValueError("The fibre theta_e_axis must be non-negative.")
    if int(beta_points) < 2:
        raise ValueError("beta_points must be at least 2.")
    if int(radial_points) < 4:
        raise ValueError("radial_points must be at least 4 because the reference path uses cubic radial splines.")
    if fibre_arcsec <= 0.0 or seeing_arcsec <= 0.0:
        raise ValueError("fibre_arcsec and seeing_arcsec must be positive.")

    ycaust_grid = np.zeros((theta_values.size, gamma_values.size), dtype=float)
    mufibre2_grid = np.zeros_like(ycaust_grid)
    mufibre3_grid = np.zeros_like(ycaust_grid)
    total_pairs = int(theta_values.size * gamma_values.size)
    with _open_pair_progress(
        enabled=bool(progress),
        total=total_pairs,
        gamma_count=gamma_values.size,
        description="finite-fibre cross-section",
    ) as progress_bar:
        for theta_index, theta_e in enumerate(theta_values):
            if theta_e <= 0.0:
                # The zero-radius row is physically zero but still part of the
                # requested grid.  Counting it keeps ETA aligned with the fixed
                # axis product users see in the CLI arguments.
                progress_bar.update(gamma_values.size)
                continue
            for gamma_index, gamma in enumerate(gamma_values):
                ycaust, mufibre2, mufibre3 = _fibre_cross_section_for_pair(
                    theta_e=float(theta_e),
                    gamma=float(gamma),
                    fibre_arcsec=float(fibre_arcsec),
                    seeing_arcsec=float(seeing_arcsec),
                    muB_min=float(muB_min),
                    beta_points=int(beta_points),
                    radial_points=int(radial_points),
                )
                ycaust_grid[theta_index, gamma_index] = ycaust
                mufibre2_grid[theta_index, gamma_index] = mufibre2
                mufibre3_grid[theta_index, gamma_index] = mufibre3
                progress_bar.update(1)

    return FibreCrossSectionGrid(
        theta_e_axis=theta_values,
        gamma_axis=gamma_values,
        mufibre2_cs_grid=mufibre2_grid,
        mufibre3_cs_grid=mufibre3_grid,
        ycaust_grid=ycaust_grid,
        fibre_arcsec=float(fibre_arcsec),
        seeing_arcsec=float(seeing_arcsec),
        muB_min=float(muB_min),
        beta_points=int(beta_points),
        radial_points=int(radial_points),
    )


__all__ = [
    "DEFAULT_CMASS_GAMMA_AXIS",
    "DEFAULT_CMASS_THETA_E_AXIS",
    "DEFAULT_FIBRE_ARCSEC",
    "DEFAULT_FIBRE_BETA_POINTS",
    "DEFAULT_FIBRE_GAMMA_AXIS",
    "DEFAULT_FIBRE_RADIAL_POINTS",
    "DEFAULT_FIBRE_THETA_E_AXIS",
    "DEFAULT_GAMMA_MAX",
    "DEFAULT_GAMMA_MIN",
    "DEFAULT_GAMMA_POINTS",
    "DEFAULT_MUB_MIN",
    "DEFAULT_SEEING_ARCSEC",
    "FibreCrossSectionGrid",
    "PowerLawCrossSectionGrid",
    "SONNENFELD_CROSS_SECTION_REFERENCE_SHA",
    "SONNENFELD_CROSS_SECTION_REFERENCE_URL",
    "compute_fibre_cross_section_grid",
    "compute_power_law_cross_section_grid",
    "power_law_alpha",
    "power_law_kappa",
    "power_law_radial_caustic",
    "radial_magnification",
    "tangential_magnification",
    "total_magnification",
]
