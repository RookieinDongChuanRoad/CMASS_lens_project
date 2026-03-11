"""
Minimal flat Lambda-CDM cosmology utilities.

The project requirements fix the cosmology, so this module keeps the distance
calculations local and deterministic. The numerical implementation is
deliberately straightforward: a precomputed distance table and linear
interpolation are sufficient for the current project scope.
"""

from __future__ import annotations

import math

import numpy as np


class FlatLambdaCDM:
    """
    Flat Lambda-CDM helper with a precomputed angular-diameter distance table.

    Distances are stored in Mpc internally because that is the conventional
    unit for cosmology summaries, then converted to kpc where the lensing
    equations require it.
    """

    def __init__(
        self,
        h0: float = 70.0,
        omega_m: float = 0.3,
        table_max_z: float = 5.0,
        table_size: int = 8001,
    ) -> None:
        self.h0 = h0
        self.omega_m = omega_m
        self.omega_lambda = 1.0 - omega_m
        self.speed_of_light_km_s = 299792.458
        # In this unit system, G converts directly into a critical surface
        # density in Msun / kpc^2 once distances are expressed in kpc.
        self.gravitational_constant = 4.30091e-6  # kpc (km/s)^2 / Msun
        self.z_table = np.linspace(0.0, table_max_z, table_size)
        self.comoving_distance_table_mpc = self._build_comoving_distance_table()

    def _e_z(self, z_values: np.ndarray) -> np.ndarray:
        """Return the dimensionless expansion rate E(z)."""

        return np.sqrt(self.omega_m * (1.0 + z_values) ** 3 + self.omega_lambda)

    def _build_comoving_distance_table(self) -> np.ndarray:
        """
        Precompute the line-of-sight comoving distance table.

        The cumulative trapezoid is implemented manually to avoid another
        dependency. The result is smooth enough for the project's interpolation
        use case.
        """

        integrand = 1.0 / self._e_z(self.z_table)
        cumulative = np.zeros_like(self.z_table)
        dz = np.diff(self.z_table)
        cumulative[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dz)
        return (self.speed_of_light_km_s / self.h0) * cumulative

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

    def theta_ein_from_m5_gamma(self, z_d: float, z_s: float, m5: float, gamma: float) -> float:
        """
        Compute the Einstein radius in arcseconds from `(m5, gamma)`.

        The equation follows the requirement document directly. The function
        returns zero for all physically invalid front/back ordering cases.
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
        r_ein_kpc = (10.0**m5 / (math.pi * sigma_critical * 5.0 ** (3.0 - gamma))) ** (1.0 / (gamma - 1.0))
        return float(r_ein_kpc / dl_kpc * 206265.0)
