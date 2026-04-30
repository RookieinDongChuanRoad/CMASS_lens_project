"""
Build the compiled array context used by the production `log_prob` path.

The current project keeps rich typed records for readability and schema
validation, but the hot path should not traverse Python objects. This module
performs the one-time transformation from `ObservationRecord` instances to the
contiguous arrays consumed by the JAX production backend and the retained
legacy numba regression kernels.
"""

from __future__ import annotations

import math

import numpy as np

from .cosmology import FlatLambdaCDM
from .io import load_cross_section_grid, load_observation_contract, load_observations, load_sigma_unit_table
from .profiles import build_profile_spec
from .types import CompiledModelContext, RandomBasis, RuntimeConfig

LOG10_2PI = math.log10(2.0 * math.pi)


def build_random_basis(num_samples: int, seed: int) -> RandomBasis:
    """
    Create the fixed random basis used by Monte Carlo normalization.

    The matrix shape matches the reference implementation's normalization
    kernel contract. Each row is one deterministic set of standard-normal
    variates that can be reinterpreted inside the kernel without any Python
    control flow.
    """

    rng = np.random.default_rng(seed)
    return RandomBasis(base_normals=np.ascontiguousarray(rng.normal(size=(num_samples, 8)), dtype=np.float64))


def _effective_radius_log10_kpc(effective_radius_arcsec: float, z_d: float, cosmology: FlatLambdaCDM) -> float:
    """Convert an observed effective radius in arcseconds to `log10(kpc)`."""

    radius_kpc = effective_radius_arcsec * cosmology.kpc_per_arcsec(z_d)
    return math.log10(max(radius_kpc, 1.0e-12))


def build_compiled_context(runtime_config: RuntimeConfig) -> tuple[CompiledModelContext, object, object, FlatLambdaCDM, RandomBasis]:
    """
    Build the numerical context and keep the high-level objects alongside it.

    Returning both the compiled context and the original typed helpers lets the
    runner continue writing readable metadata while the sampler uses only the
    array representation during inference.
    """

    profile = build_profile_spec(runtime_config.profile.name)
    observations = load_observations(
        runtime_config.data.observation_path,
        profile,
        runtime_config.mass_definition,
    )
    cross_section_grid = load_cross_section_grid(runtime_config.data.cross_section_path)
    sigma_table = None
    if runtime_config.fp_prior.enabled:
        if runtime_config.data.sigma_table_path is None:
            raise ValueError("FP prior is enabled but no sigma table path was configured.")
        observation_contract = load_observation_contract(runtime_config.data.observation_path)
        sigma_table = load_sigma_unit_table(
            runtime_config.data.sigma_table_path,
            profile,
            runtime_config.mass_definition,
            observation_flavor=str(observation_contract["observation_flavor"]),
        )
    cosmology = FlatLambdaCDM(
        h0=runtime_config.cosmology.h0,
        omega_m=runtime_config.cosmology.omega_m,
    )
    random_basis = build_random_basis(
        runtime_config.integration.normalization_samples,
        runtime_config.sampling.random_seed,
    )

    n_lens = len(observations)
    n_gamma = runtime_config.integration.gamma_points #* default = 200
    n_mstar = runtime_config.integration.mstar_points #* default = 200

    gamma_grid_int = np.linspace(
        float(observations[0].gamma_grid_17[0]),
        float(observations[0].gamma_grid_17[-1]),
        n_gamma,
        dtype=np.float64,
    )
    cs_over_theta_int = np.interp(
        gamma_grid_int,
        cross_section_grid.gamma_grid,
        cross_section_grid.cs_over_theta_ein,
        left=float(cross_section_grid.cs_over_theta_ein[0]),
        right=float(cross_section_grid.cs_over_theta_ein[-1]),
    ).astype(np.float64)

    zd = np.zeros(n_lens, dtype=np.float64)
    zs = np.zeros(n_lens, dtype=np.float64)
    has_s2 = np.zeros(n_lens, dtype=np.int64)
    num_sigma = np.zeros(n_lens, dtype=np.int64)
    sigma_obs = np.zeros((n_lens, 2), dtype=np.float64)
    sigma_err = np.ones((n_lens, 2), dtype=np.float64)
    mass_grid_int = np.zeros((n_lens, n_gamma), dtype=np.float64)
    dmass_dthetaein_grid_int = np.zeros((n_lens, n_gamma), dtype=np.float64)
    s2_grid_int = np.zeros((n_lens, n_gamma), dtype=np.float64)
    re_logkpc = np.zeros(n_lens, dtype=np.float64)
    mstar_grid = np.zeros((n_lens, n_mstar), dtype=np.float64)

    for lens_index, observation in enumerate(observations):
        zd[lens_index] = observation.z_d
        zs[lens_index] = observation.z_s
        has_s2[lens_index] = 1 if observation.s2_grid_17 is not None else 0
        num_sigma[lens_index] = observation.num_sigma
        if observation.num_sigma > 0:
            count = min(2, observation.sigma_observed.shape[0], observation.sigma_error.shape[0])
            sigma_obs[lens_index, :count] = observation.sigma_observed[:count]
            sigma_err[lens_index, :count] = observation.sigma_error[:count]

        mass_grid_int[lens_index] = np.interp(
            gamma_grid_int,
            observation.gamma_grid_17,
            observation.mass_grid_17,
            left=float(observation.mass_grid_17[0]),
            right=float(observation.mass_grid_17[-1]),
        )
        dmass_dthetaein_grid_int[lens_index] = np.interp(
            gamma_grid_int,
            observation.gamma_grid_17,
            observation.dmass_dthetaein_grid_17,
            left=float(observation.dmass_dthetaein_grid_17[0]),
            right=float(observation.dmass_dthetaein_grid_17[-1]),
        )
        if observation.s2_grid_17 is not None:
            s2_grid_int[lens_index] = np.interp(
                gamma_grid_int,
                observation.gamma_grid_17,
                observation.s2_grid_17,
                left=float(observation.s2_grid_17[0]),
                right=float(observation.s2_grid_17[-1]),
            )

        re_logkpc[lens_index] = _effective_radius_log10_kpc(
            observation.effective_radius_arcsec,
            observation.z_d,
            cosmology,
        )
        lo = observation.log_stellar_mass_obs - 5.0 * observation.log_stellar_mass_err
        hi = observation.log_stellar_mass_obs + 5.0 * observation.log_stellar_mass_err
        mstar_grid[lens_index] = np.linspace(lo, hi, n_mstar, dtype=np.float64)

    sqrt2 = math.sqrt(2.0)
    sqrt2pi = math.sqrt(2.0 * math.pi)
    p_zd_fixed = np.exp(-0.5 * ((zd - 0.558) / 0.085) ** 2) / (0.085 * sqrt2pi)

    mstar_shift11p4 = np.zeros((n_lens, n_mstar), dtype=np.float64)
    sigma_star_shift9p0_grid = np.zeros((n_lens, n_mstar), dtype=np.float64)
    mstar_integrand_base = np.zeros((n_lens, n_mstar), dtype=np.float64)
    delta_r_grid = np.zeros((n_lens, n_mstar), dtype=np.float64)

    for lens_index, observation in enumerate(observations):
        # `n` only affects the precomputed size-relation term. Once `delta_r`
        # and the fixed `m*` base are built, the hot path never needs the
        # lens-wise `n` array again, so we keep it local instead of exporting
        # another compiled-context field.
        n_value = profile.fixed_n if profile.fixed_n is not None else max(observation.n_observed, 1.0e-8)
        for mstar_index in range(n_mstar):
            mstar = mstar_grid[lens_index, mstar_index]
            shift = mstar - 11.4
            mu_r_value = profile.mu_r0 + profile.beta_r * shift
            if profile.nu_r is not None:
                mu_r_value += profile.nu_r * (math.log10(max(n_value, 1.0e-12)) - math.log10(4.0))
            delta_r = re_logkpc[lens_index] - mu_r_value

            p_mobs = np.exp(-0.5 * ((observation.log_stellar_mass_obs - mstar) / observation.log_stellar_mass_err) ** 2)
            p_mobs /= observation.log_stellar_mass_err * sqrt2pi
            t = (mstar - profile.mass_function_loc) / profile.mass_function_scale
            p_s = 2.0 * np.exp(-0.5 * t * t) / sqrt2pi
            p_s *= 0.5 * (1.0 + math.erf(profile.mass_function_alpha * t / sqrt2))
            p_s /= profile.mass_function_scale
            p_r = np.exp(-0.5 * ((re_logkpc[lens_index] - mu_r_value) / profile.sigma_r) ** 2)
            p_r /= profile.sigma_r * sqrt2pi

            mstar_shift11p4[lens_index, mstar_index] = shift
            sigma_star_shift9p0_grid[lens_index, mstar_index] = mstar - LOG10_2PI - 2.0 * re_logkpc[lens_index] - 9.0
            delta_r_grid[lens_index, mstar_index] = delta_r
            # Store only the parameter-independent part of the inner `m*`
            # integrand. The likelihood kernel now applies the hyper-parameter
            # dependent Gaussian terms first and lets `np.trapezoid(...)`
            # handle quadrature weights explicitly from `mstar_grid`.
            mstar_integrand_base[lens_index, mstar_index] = p_mobs * p_s * p_r

    if sigma_table is None:
        fp_gamma_axis = np.zeros(1, dtype=np.float64)
        fp_zd_axis = np.zeros(1, dtype=np.float64)
        fp_log_re_kpc_axis = np.zeros(1, dtype=np.float64)
        fp_n_axis = np.zeros(1, dtype=np.float64)
        fp_sigma_unit_grid = np.zeros((1, 1, 1, 1), dtype=np.float64)
        fp_has_n_axis = 0
    else:
        fp_gamma_axis = np.asarray(sigma_table.gamma_axis, dtype=np.float64)
        fp_zd_axis = np.asarray(sigma_table.zd_axis, dtype=np.float64)
        fp_log_re_kpc_axis = np.asarray(sigma_table.log_re_kpc_axis, dtype=np.float64)
        if sigma_table.n_axis is None:
            fp_n_axis = np.asarray([profile.fixed_n if profile.fixed_n is not None else 4.0], dtype=np.float64)
            fp_sigma_unit_grid = np.asarray(sigma_table.sigma_unit_grid, dtype=np.float64)[..., None]
            fp_has_n_axis = 0
        else:
            fp_n_axis = np.asarray(sigma_table.n_axis, dtype=np.float64)
            fp_sigma_unit_grid = np.asarray(sigma_table.sigma_unit_grid, dtype=np.float64)
            fp_has_n_axis = 1

    context = CompiledModelContext(
        z_grid=np.ascontiguousarray(cosmology.z_table, dtype=np.float64),
        chi_kpc_grid=np.ascontiguousarray(cosmology.comoving_distance_table_mpc * 1000.0, dtype=np.float64),
        cs_gamma_grid=np.ascontiguousarray(cross_section_grid.gamma_grid, dtype=np.float64),
        cs_over_theta_grid=np.ascontiguousarray(cross_section_grid.cs_over_theta_ein, dtype=np.float64),
        cs_over_theta_int=np.ascontiguousarray(cs_over_theta_int, dtype=np.float64),
        gamma_grid_int=np.ascontiguousarray(gamma_grid_int, dtype=np.float64),
        mass_grid_int=np.ascontiguousarray(mass_grid_int, dtype=np.float64),
        dmass_dthetaein_grid_int=np.ascontiguousarray(dmass_dthetaein_grid_int, dtype=np.float64),
        s2_grid_int=np.ascontiguousarray(s2_grid_int, dtype=np.float64),
        has_s2=np.ascontiguousarray(has_s2, dtype=np.int64),
        num_sigma=np.ascontiguousarray(num_sigma, dtype=np.int64),
        sigma_obs=np.ascontiguousarray(sigma_obs, dtype=np.float64),
        sigma_err=np.ascontiguousarray(sigma_err, dtype=np.float64),
        zd=np.ascontiguousarray(zd, dtype=np.float64),
        zs=np.ascontiguousarray(zs, dtype=np.float64),
        p_zd_fixed=np.ascontiguousarray(p_zd_fixed, dtype=np.float64),
        mstar_grid=np.ascontiguousarray(mstar_grid, dtype=np.float64),
        mstar_shift11p4=np.ascontiguousarray(mstar_shift11p4, dtype=np.float64),
        sigma_star_shift9p0_grid=np.ascontiguousarray(sigma_star_shift9p0_grid, dtype=np.float64),
        mstar_integrand_base=np.ascontiguousarray(mstar_integrand_base, dtype=np.float64),
        delta_r_grid=np.ascontiguousarray(delta_r_grid, dtype=np.float64),
        base_normals=random_basis.base_normals,
        mass_radius_kpc=float(runtime_config.mass_definition.radius_kpc),
        use_sersic_index=1 if profile.uses_observed_n_in_likelihood else 0,
        n_fixed=profile.fixed_n if profile.fixed_n is not None else 4.0,
        mu_n0=profile.mu_n0 if profile.mu_n0 is not None else 0.0,
        beta_n=profile.beta_n if profile.beta_n is not None else 0.0,
        sigma_n=profile.sigma_n if profile.sigma_n is not None else 0.0,
        mass_function_loc=profile.mass_function_loc,
        mass_function_scale=profile.mass_function_scale,
        mass_function_alpha=profile.mass_function_alpha,
        mu_r0=profile.mu_r0,
        beta_r=profile.beta_r,
        sigma_r=profile.sigma_r,
        nu_r=profile.nu_r if profile.nu_r is not None else 0.0,
        mu_d=0.558,
        sigma_d=0.085,
        gamma_trunc_low=1.2,
        gamma_trunc_high=2.8,
        normalization_min_value=1.0e-10,
        gamma_mode_code=runtime_config.parameter_schema.gamma_mode_code,
        fp_enabled=1 if runtime_config.fp_prior.enabled else 0,
        fp_fit_mstar_min=runtime_config.fp_prior.fit_mstar_min,
        fp_pivot_mstar=runtime_config.fp_prior.pivot_mstar,
        fp_fiducial_scatter=runtime_config.fp_prior.fiducial_scatter,
        fp_scatter_error=runtime_config.fp_prior.scatter_error,
        fp_mu_v_prior=runtime_config.fp_prior.mu_v_prior,
        fp_mu_v_error=runtime_config.fp_prior.mu_v_error,
        fp_beta_v_prior=runtime_config.fp_prior.beta_v_prior,
        fp_beta_v_error=runtime_config.fp_prior.beta_v_error,
        fp_gamma_axis=np.ascontiguousarray(fp_gamma_axis, dtype=np.float64),
        fp_zd_axis=np.ascontiguousarray(fp_zd_axis, dtype=np.float64),
        fp_log_re_kpc_axis=np.ascontiguousarray(fp_log_re_kpc_axis, dtype=np.float64),
        fp_n_axis=np.ascontiguousarray(fp_n_axis, dtype=np.float64),
        fp_sigma_unit_grid=np.ascontiguousarray(fp_sigma_unit_grid, dtype=np.float64),
        fp_has_n_axis=fp_has_n_axis,
    )
    return context, profile, cross_section_grid, cosmology, random_basis, observations
