"""
Minimal flat Lambda-CDM cosmology utilities.

The project requirements fix the cosmology, so this module keeps the distance
calculations local and deterministic. The numerical implementation is
deliberately straightforward: astropy provides the authoritative cosmological
distances, while the project keeps a precomputed comoving-distance table and
linear interpolation so backend kernels receive fixed-shape numeric arrays.
"""

from __future__ import annotations

import math

import numpy as np
from astropy.cosmology import FlatLambdaCDM as AstropyFlatLambdaCDM
from astropy import constants as astropy_constants


# The requirements document fixes the interpolation support range and
# resolution. These constants remain internal implementation details so the
# kernels keep receiving the same table shape without exposing new tuning
# surface area in user configs.
DEFAULT_DISTANCE_TABLE_MAX_Z = 5.0
DEFAULT_DISTANCE_TABLE_SIZE = 8001


class FlatLambdaCDM:
    """
    Flat Lambda-CDM helper with an astropy-backed comoving-distance table.

    Why we keep this wrapper instead of pushing astropy objects through the
    whole codebase:
    - callers already expect plain floats and ndarrays
    - the performance-critical kernels require contiguous numeric arrays, not
      `Quantity` objects
    - one shared lookup table keeps the Python helpers and the kernel path on
      the same numerical source of truth
    """

    def __init__(
        self,
        h0: float = 70.0,
        omega_m: float = 0.3,
    ) -> None:
        self.h0 = h0
        self.omega_m = omega_m
        self.omega_lambda = 1.0 - omega_m
        # self.speed_of_light_km_s = 299792.458
        # # In this unit system, G converts directly into a critical surface
        # # density in Msun / kpc^2 once distances are expressed in kpc.
        # self.gravitational_constant = 4.30091e-6  # kpc (km/s)^2 / Msun
        self._astropy = AstropyFlatLambdaCDM(H0=h0, Om0=omega_m)
        self.speed_of_light_km_s = astropy_constants.c.to("km/s").value
        self.gravitational_constant = astropy_constants.G.to("kpc km^2 /s^2 Msun").value
        self.z_table = np.ascontiguousarray(
            np.linspace(0.0, DEFAULT_DISTANCE_TABLE_MAX_Z, DEFAULT_DISTANCE_TABLE_SIZE, dtype=np.float64)
        )
        self.comoving_distance_table_mpc = self._build_comoving_distance_table()

    def _build_comoving_distance_table(self) -> np.ndarray:
        """
        Precompute the line-of-sight comoving distance table from astropy.

        The wrapper strips the `Quantity` layer immediately so every downstream
        consumer sees the same plain `float64` array contract the old
        implementation exposed.
        """

        return np.ascontiguousarray(
            self._astropy.comoving_distance(self.z_table).to("Mpc").value, dtype=np.float64
        )

    def comoving_distance_mpc(self, z_value: float) -> float:
        """Interpolate the comoving distance table."""

        return float(np.interp(z_value, self.z_table, self.comoving_distance_table_mpc))

    def angular_diameter_distance_mpc(self, z_value: float) -> float:
        """Return the observer-to-redshift angular-diameter distance."""

        return self.comoving_distance_mpc(z_value) / (1.0 + z_value)

    def angular_diameter_distance_between_mpc(self, z_d: float, z_s: float) -> float:
        """Return the angular-diameter distance between deflector and source."""

        if z_s <= z_d:
            return 0.0
        chi_d = self.comoving_distance_mpc(z_d)
        chi_s = self.comoving_distance_mpc(z_s)
        return (chi_s - chi_d) / (1.0 + z_s)

    def kpc_per_arcsec(self, z_value: float) -> float:
        """Convert one arcsecond at `z` into a proper kpc scale."""

        distance_kpc = self.angular_diameter_distance_mpc(z_value) * 1000.0
        return distance_kpc / 206265.0

    def theta_ein_from_mass_gamma(
        self,
        z_d: float,
        z_s: float,
        mass_value: float,
        gamma: float,
        mass_radius_kpc: float,
    ) -> float:
        """
        Compute the Einstein radius in arcseconds from `(m_R, gamma)`.

        The generalized relation is

        `10**m_R = pi * Sigma_c * r_ein**(gamma-1) * R**(3-gamma)`.

        This lets the caller choose whether the enclosed-mass observable is
        defined at 5 kpc, 10 kpc, or any other explicitly supported radius.
        The function returns zero for all physically invalid front/back
        ordering cases.
        """

        if z_d >= z_s:
            return 0.0

        dl_kpc = self.angular_diameter_distance_mpc(z_d) * 1000.0
        ds_kpc = self.angular_diameter_distance_mpc(z_s) * 1000.0
        dls_kpc = self.angular_diameter_distance_between_mpc(z_d, z_s) * 1000.0
        if dl_kpc <= 0.0 or ds_kpc <= 0.0 or dls_kpc <= 0.0:
            return 0.0

        sigma_critical = (
            self.speed_of_light_km_s**2 / (4.0 * math.pi * self.gravitational_constant)
        ) * (ds_kpc / (dl_kpc * dls_kpc))
        r_ein_kpc = (
            10.0**mass_value / (math.pi * sigma_critical * float(mass_radius_kpc) ** (3.0 - gamma))
        ) ** (1.0 / (gamma - 1.0))
        return float(r_ein_kpc / dl_kpc * 206265.0)

    def theta_ein_from_m5_gamma(self, z_d: float, z_s: float, m5: float, gamma: float) -> float:
        """
        Backward-compatible wrapper for the historical `m5`-specific API.

        Older tests and helper code still call this method directly. Keeping
        the wrapper lets the new generalized implementation land without
        breaking the rest of the repository all at once.
        """

        return self.theta_ein_from_mass_gamma(
            z_d=z_d,
            z_s=z_s,
            mass_value=m5,
            gamma=gamma,
            mass_radius_kpc=5.0,
        )
