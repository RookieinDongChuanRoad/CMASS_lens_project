"""Shared typed containers for interpolation-grid processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AperturePolicy:
    """Typed description of the aperture geometry used by Jeans solves.

    The upstream ``spherical_jeans`` dependency accepts two distinct aperture
    representations:
    - a scalar radius for a centered circular aperture
    - a length-2 sequence for a centered rectangular aperture

    The project needs both because the legacy production workflow uses a fixed
    rectangular slit, while the new BOSS rebuild uses a circular aperture with
    a 1 arcsec radius. Carrying that choice as explicit data prevents hidden
    global coupling inside the physics layer.
    """

    shape: str
    seeing_fwhm_arcsec: float
    width_arcsec: float | None = None
    height_arcsec: float | None = None
    radius_arcsec: float | None = None

    def __post_init__(self) -> None:
        """Validate the geometry so invalid policies fail early and clearly."""

        normalized_shape = self.shape.strip().lower()
        object.__setattr__(self, "shape", normalized_shape)

        if self.seeing_fwhm_arcsec <= 0.0:
            raise ValueError("AperturePolicy requires a strictly positive seeing_fwhm_arcsec.")

        if normalized_shape == "rectangular":
            if self.width_arcsec is None or self.height_arcsec is None:
                raise ValueError("Rectangular AperturePolicy requires width_arcsec and height_arcsec.")
            if self.width_arcsec <= 0.0 or self.height_arcsec <= 0.0:
                raise ValueError("Rectangular AperturePolicy dimensions must be strictly positive.")
            if self.radius_arcsec is not None:
                raise ValueError("Rectangular AperturePolicy must not define radius_arcsec.")
            return

        if normalized_shape == "circular":
            if self.radius_arcsec is None:
                raise ValueError("Circular AperturePolicy requires radius_arcsec.")
            if self.radius_arcsec <= 0.0:
                raise ValueError("Circular AperturePolicy radius must be strictly positive.")
            if self.width_arcsec is not None or self.height_arcsec is not None:
                raise ValueError("Circular AperturePolicy must not define width_arcsec or height_arcsec.")
            return

        raise ValueError(f"Unsupported AperturePolicy shape: {self.shape}")

    @classmethod
    def rectangular(
        cls,
        width_arcsec: float,
        height_arcsec: float,
        seeing_fwhm_arcsec: float,
    ) -> "AperturePolicy":
        """Construct the centered rectangular policy used by existing products."""

        return cls(
            shape="rectangular",
            width_arcsec=float(width_arcsec),
            height_arcsec=float(height_arcsec),
            seeing_fwhm_arcsec=float(seeing_fwhm_arcsec),
        )

    @classmethod
    def circular(
        cls,
        radius_arcsec: float,
        seeing_fwhm_arcsec: float,
    ) -> "AperturePolicy":
        """Construct the centered circular policy required by the BOSS files."""

        return cls(
            shape="circular",
            radius_arcsec=float(radius_arcsec),
            seeing_fwhm_arcsec=float(seeing_fwhm_arcsec),
        )


@dataclass(frozen=True)
class GalaxyInputs:
    """Physical inputs extracted from one HDF5 group.

    Attributes
    ----------
    group_name:
        Name of the galaxy group inside the HDF5 file.
    source_filename:
        Input filename. We use this to choose the deV vs free-Sersic branch
        because the two project files encode different tracer assumptions.
    zd, zs:
        Lens and source redshift.
    sigma_crit:
        Critical surface density used by the power-law normalization.
    rein_arcsec, r_ein_kpc:
        Einstein radius in arcsec and kpc.
    re_arcsec, reff_dev_arcsec:
        Effective radius for the free-Sersic and deV branches respectively.
    nser:
        Sersic index used only for the free-Sersic branch.
    aperture_width_arcsec:
        Stored historical slit width from the input file. This is kept for
        reference-script regression tests and for documenting previous data
        products, but the current production `s2_grid` policy uses the fixed
        aperture width defined in `config.py` instead.
    has_s2_grid:
        Whether the input group already contains the velocity-dispersion grid.
        The business rule is to update only existing `s2_grid` datasets.
    """

    group_name: str
    source_filename: str
    zd: float
    zs: float
    sigma_crit: float
    rein_arcsec: float
    r_ein_kpc: float
    re_arcsec: float | None
    reff_dev_arcsec: float | None
    nser: float | None
    aperture_width_arcsec: float | None
    has_s2_grid: bool


@dataclass
class ProcessingSummary:
    """Aggregated result of processing one HDF5 file."""

    input_path: Path
    output_path: Path
    total_groups: int = 0
    updated_m5: int = 0
    updated_dm5: int = 0
    updated_s2: int = 0
    failures: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SigmaUnitTable:
    """In-memory representation of one PPT-facing sigma-unit interpolation table.

    Attributes
    ----------
    profile_name:
        Tracer profile branch. Supported values are `devauc` and `sersic`.
    mass_definition_label:
        Public label for the enclosed-mass definition used by this table, for
        example `m5` or `m10`.
    mass_radius_kpc:
        Physical radius in kpc at which the enclosed mass is defined.
    gamma_axis, zd_axis, log_re_kpc_axis:
        Explicit interpolation axes consumed downstream by PPC.
    values:
        Tabulated `S_unit = sigma^2 / 10**m_R` values for the selected mass
        definition. The axis order is fixed to `(gamma, zd, log_re_kpc)` for
        deV and `(gamma, zd, log_re_kpc, n)` for Sersic so downstream code can
        build one unambiguous interpolator.
    n_axis:
        Optional Sersic-index axis. It must be absent for the deV table.
    observation_flavor:
        Observation flavor associated with this table. The current supported
        values are `slit` for the legacy rectangular aperture and `boss` for
        the circular 1 arcsec aperture.
    aperture_shape, aperture_width_arcsec, aperture_height_arcsec,
    aperture_radius_arcsec, seeing_fwhm_arcsec:
        Explicit aperture metadata written into bundle leaves so downstream
        PPC code can validate that the selected table matches the run's raw
        observation contract.
    """

    profile_name: str
    mass_definition_label: str
    mass_radius_kpc: float
    gamma_axis: np.ndarray
    zd_axis: np.ndarray
    log_re_kpc_axis: np.ndarray
    values: np.ndarray
    n_axis: np.ndarray | None = None
    observation_flavor: str = "slit"
    aperture_shape: str = "rectangular"
    aperture_width_arcsec: float | None = 1.6
    aperture_height_arcsec: float | None = 0.9
    aperture_radius_arcsec: float | None = None
    seeing_fwhm_arcsec: float = 0.9
