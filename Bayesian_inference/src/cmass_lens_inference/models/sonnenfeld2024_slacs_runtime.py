"""Runtime adapter for the Sonnenfeld 2024 SLACS model.

This module mirrors ``cmass_runtime.py`` but remains Sonnenfeld-specific where
the source context is built.  The generic backend owns compiled-model
construction through the declarative ``DataSpec`` below.
"""

from __future__ import annotations

from ..model_interfaces import (
    CompiledContextBundle,
    ContextArraySpec,
    ContextScalarSpec,
    DataSpec,
    ModelRuntimeAdapter,
    StaticContextSpec,
)
from ..profiles import build_profile_spec
from ..types import RuntimeConfig
from .components.sonnenfeld2024_slacs.preprocessing import (
    build_sonnenfeld_context_from_canonical_dataset,
    load_sonnenfeld_canonical_dataset,
)


def build_context_bundle(runtime_config: RuntimeConfig) -> CompiledContextBundle:
    """
    Build the Sonnenfeld source-context bundle for the generic backend.

    Production inference is canonical-only.  This guard keeps programmatic
    callers from bypassing the config parser and accidentally trying to feed
    raw observation or grid products into the runtime.
    """

    if runtime_config.data.inference_dataset_path is None:
        raise ValueError("The Sonnenfeld runtime requires data.inference_dataset_path.")
    profile = build_profile_spec(runtime_config.profile.name)
    dataset = load_sonnenfeld_canonical_dataset(runtime_config, profile=profile)
    return build_sonnenfeld_context_from_canonical_dataset(
        runtime_config,
        dataset=dataset,
        profile=profile,
    )


def get_data_spec() -> DataSpec:
    """
    Return the Sonnenfeld context-packing declaration.

    The scalar order is part of the backend hot-path contract.  Kernels may pack
    this compact array instead of repeatedly traversing Python attributes.
    """

    return DataSpec(
        backend_context_type=object,
        array_fields=(
            ContextArraySpec("z_grid"),
            ContextArraySpec("chi_kpc_grid"),
            ContextArraySpec("cs_theta_e_axis"),
            ContextArraySpec("cs_gamma_axis"),
            ContextArraySpec("cs_cross_section_grid"),
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
            ContextArraySpec("log_mstar_obs"),
            ContextArraySpec("log_mstar_err"),
            ContextArraySpec("log_re_obs"),
            ContextArraySpec("n_obs"),
            ContextArraySpec("theta_e_obs"),
            ContextArraySpec("mstar_grid"),
            ContextArraySpec("parent_mstar_density_grid"),
            ContextArraySpec("size_density_grid"),
            ContextArraySpec("delta_r_grid"),
            ContextArraySpec("mstar_shift_grid"),
            ContextArraySpec("base_normals"),
            ContextArraySpec("population_gamma_axis"),
            ContextArraySpec("population_zd_axis"),
            ContextArraySpec("population_log_re_kpc_axis"),
            ContextArraySpec("population_n_axis"),
            ContextArraySpec("population_sigma_unit_grid"),
        ),
        scalar_fields=(
            ContextScalarSpec("mass_radius_kpc"),
            ContextScalarSpec("mass_log_physical_offset"),
            ContextScalarSpec("mstar_pivot"),
            ContextScalarSpec("mbar"),
            ContextScalarSpec("parent_alpha"),
            ContextScalarSpec("truncation_mass_scatter"),
            ContextScalarSpec("size_mu0"),
            ContextScalarSpec("size_mu1"),
            ContextScalarSpec("size_sigma"),
            ContextScalarSpec("size_mu2"),
            ContextScalarSpec("n_fixed"),
            ContextScalarSpec("gamma_trunc_low"),
            ContextScalarSpec("gamma_trunc_high"),
            ContextScalarSpec("parent_zd_min"),
            ContextScalarSpec("parent_zd_max"),
            ContextScalarSpec("parent_mstar_min"),
            ContextScalarSpec("parent_mstar_max"),
            ContextScalarSpec("sigma_proxy_fractional_scatter"),
            ContextScalarSpec("normalization_min_value"),
        ),
        static_fields=(StaticContextSpec("use_sersic_index"),),
        normalization_samples_field="base_normals",
        normalization_min_value_field="normalization_min_value",
    )


def get_runtime_adapter() -> ModelRuntimeAdapter:
    """Return the runtime adapter paired with the Sonnenfeld ``ModelSpec``."""

    return ModelRuntimeAdapter(
        build_context_bundle=build_context_bundle,
        data_spec=get_data_spec(),
    )


__all__ = ["build_context_bundle", "get_data_spec", "get_runtime_adapter"]
