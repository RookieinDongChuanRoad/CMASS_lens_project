"""
Runtime adapter for the default CMASS model.

This module is the thin engineering boundary that still remains before a fully
generic data-loading layer exists.  It performs two jobs:

1. build the validated CMASS NumPy source context from the canonical inference
   dataset;
2. declare the source-context fields that backends may consume.

The scientific equations stay outside this runtime glue.  The generic backend
consumes this declaration and handles `CompiledModel` construction.
"""

from __future__ import annotations

from statistical_sl.models.interfaces import (
    CompiledContextBundle,
    ContextArraySpec,
    ContextScalarSpec,
    DataSpec,
    ModelRuntimeAdapter,
    StaticContextSpec,
)
from statistical_sl.inference.profiles import build_profile_spec
from statistical_sl.inference.types import RuntimeConfig
from .preprocessing import (
    build_cmass_context_from_canonical_dataset,
    load_cmass_canonical_dataset,
)


def build_context_bundle(runtime_config: RuntimeConfig) -> CompiledContextBundle:
    """
    Build the CMASS source-context bundle for the generic backend.

    Production inference is canonical-only: raw observation/cross-section
    products must be normalized by the data-preparation step before this
    adapter is called.  Keeping the guard here gives programmatic callers the
    same clear failure as the YAML parser if they construct `RuntimeConfig`
    manually.
    """

    if runtime_config.data.inference_dataset_path is None:
        raise ValueError(
            "The CMASS runtime requires data.inference_dataset_path. "
            "Raw observation/cross-section inputs must be normalized by "
            "data preparation before entering production inference."
        )
    profile = build_profile_spec(runtime_config.profile.name)
    dataset = load_cmass_canonical_dataset(runtime_config, profile=profile)
    return build_cmass_context_from_canonical_dataset(
        runtime_config,
        dataset=dataset,
        profile=profile,
    )


def get_data_spec() -> DataSpec:
    """
    Return the CMASS context declaration.

    The scalar field order intentionally preserves the previous backend
    contract.  Backend kernels may pack these fields into compact arrays, so
    changing the order is a scientific behavior change and must be covered by
    explicit tests.
    """

    return DataSpec(
        backend_context_type=object,
        array_fields=(
            ContextArraySpec("z_grid"),
            ContextArraySpec("chi_kpc_grid"),
            ContextArraySpec("cs_gamma_grid"),
            ContextArraySpec("cs_over_theta_grid"),
            ContextArraySpec("cs_theta_e_axis"),
            ContextArraySpec("cs_cross_section_grid"),
            ContextArraySpec("cs_over_theta_int"),
            ContextArraySpec("gamma_grid_int"),
            ContextArraySpec("mass_grid_int"),
            ContextArraySpec("dmass_dthetaein_grid_int"),
            ContextArraySpec("s2_grid_int"),
            ContextArraySpec("has_s2"),
            ContextArraySpec("num_sigma"),
            ContextArraySpec("sigma_obs"),
            ContextArraySpec("sigma_err"),
            ContextArraySpec("zd"),
            ContextArraySpec("zs"),
            ContextArraySpec("p_zd_fixed"),
            ContextArraySpec("mstar_grid"),
            ContextArraySpec("mstar_shift11p4"),
            ContextArraySpec("sigma_star_shift9p0_grid"),
            ContextArraySpec("mstar_integrand_base"),
            ContextArraySpec("delta_r_grid"),
            ContextArraySpec("base_normals"),
            ContextArraySpec("fp_gamma_axis"),
            ContextArraySpec("fp_zd_axis"),
            ContextArraySpec("fp_log_re_kpc_axis"),
            ContextArraySpec("fp_n_axis"),
            ContextArraySpec("fp_sigma_unit_grid"),
        ),
        scalar_fields=(
            ContextScalarSpec("mass_radius_kpc"),
            ContextScalarSpec("n_fixed"),
            ContextScalarSpec("mu_n0"),
            ContextScalarSpec("beta_n"),
            ContextScalarSpec("sigma_n"),
            ContextScalarSpec("mass_function_loc"),
            ContextScalarSpec("mass_function_scale"),
            ContextScalarSpec("mass_function_alpha"),
            ContextScalarSpec("mu_r0"),
            ContextScalarSpec("beta_r"),
            ContextScalarSpec("sigma_r"),
            ContextScalarSpec("nu_r"),
            ContextScalarSpec("mu_d"),
            ContextScalarSpec("sigma_d"),
            ContextScalarSpec("gamma_trunc_low"),
            ContextScalarSpec("gamma_trunc_high"),
            ContextScalarSpec("normalization_min_value"),
            ContextScalarSpec("fp_fit_mstar_min"),
            ContextScalarSpec("fp_pivot_mstar"),
            ContextScalarSpec("fp_fiducial_scatter"),
            ContextScalarSpec("fp_scatter_error"),
            ContextScalarSpec("fp_mu_v_prior"),
            ContextScalarSpec("fp_mu_v_error"),
            ContextScalarSpec("fp_beta_v_prior"),
            ContextScalarSpec("fp_beta_v_error"),
            ContextScalarSpec("stellar_mass_pivot"),
            ContextScalarSpec("mass_log_physical_offset"),
        ),
        static_fields=(
            StaticContextSpec("use_sersic_index"),
            StaticContextSpec("cross_section_mode_code"),
            StaticContextSpec("fp_enabled"),
            StaticContextSpec("fp_has_n_axis"),
        ),
        normalization_samples_field="base_normals",
        normalization_min_value_field="normalization_min_value",
    )


def get_runtime_adapter() -> ModelRuntimeAdapter:
    """
    Return the runtime adapter paired with `models.cmass.get_model_spec()`.

    No sampler or backend implementation is exposed here.  The generic backend
    derives its compiled-model container from this runtime adapter.
    """

    return ModelRuntimeAdapter(
        build_context_bundle=build_context_bundle,
        data_spec=get_data_spec(),
    )


__all__ = ["build_context_bundle", "get_data_spec", "get_runtime_adapter"]
