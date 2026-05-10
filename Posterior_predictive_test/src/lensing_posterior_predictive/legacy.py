"""Legacy CMASS PPC config compatibility helpers.

This module is the quarantine zone for pre-registry raw-path run snapshots.
Those snapshots predate `model.name` and canonical inference datasets, so they
can only be interpreted as CMASS.  New model-aware PPT paths must enter through
the production config parser and predictive registry instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cmass_lens_inference.mass_definition import LEGACY_FIXED_KPC, get_mass_definition
from cmass_lens_inference.parameter_schema import ParameterSchema
from cmass_lens_inference.types import (
    CosmologyConfig,
    DataConfig,
    FPPriorConfig,
    HyperParams,
    IntegrationConfig,
    ModelConfig,
    OutputConfig,
    ProfileConfig,
    RuntimeConfig,
    RuntimeOptions,
    SamplingConfig,
)


GAMMA_MODE_DEPENDENT = "dependent"
GAMMA_MODE_INDEPENDENT = "independent"
GAMMA_MODE_SIGMA_STAR_DEPENDENT = "sigma_star_dependent"
GAMMA_MODE_DEPENDENT_CODE = 0
GAMMA_MODE_INDEPENDENT_CODE = 1
GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE = 2
_GAMMA_MODE_ALIASES = {
    GAMMA_MODE_DEPENDENT: GAMMA_MODE_DEPENDENT,
    GAMMA_MODE_INDEPENDENT: GAMMA_MODE_INDEPENDENT,
    GAMMA_MODE_SIGMA_STAR_DEPENDENT: GAMMA_MODE_SIGMA_STAR_DEPENDENT,
    "sigma-star-dependent": GAMMA_MODE_SIGMA_STAR_DEPENDENT,
    "sigma_star": GAMMA_MODE_SIGMA_STAR_DEPENDENT,
    "sigma-star": GAMMA_MODE_SIGMA_STAR_DEPENDENT,
}


def _normalize_gamma_mode(raw_mode: str | None) -> str | None:
    """Normalize one legacy CMASS gamma-mode spelling."""

    if raw_mode is None:
        return None
    return _GAMMA_MODE_ALIASES.get(str(raw_mode).strip().lower())


def _gamma_mode_code_from_name(gamma_mode: str) -> int:
    """Return the integer code used by CMASS Numba kernels for one gamma mode."""

    normalized = _normalize_gamma_mode(gamma_mode)
    if normalized == GAMMA_MODE_DEPENDENT:
        return GAMMA_MODE_DEPENDENT_CODE
    if normalized == GAMMA_MODE_INDEPENDENT:
        return GAMMA_MODE_INDEPENDENT_CODE
    if normalized == GAMMA_MODE_SIGMA_STAR_DEPENDENT:
        return GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE
    raise ValueError(f"Unsupported CMASS gamma mode '{gamma_mode}'.")


def _legacy_ppc_parameter_order(mass_radius_kpc: int, gamma_mode: str) -> tuple[str, ...]:
    """Return the old raw-config parameter order used by archived CMASS PPC runs."""

    mass_prefix = f"mu{int(mass_radius_kpc)}_0"
    mass_parameters = (
        mass_prefix,
        f"beta{int(mass_radius_kpc)}",
        f"xi{int(mass_radius_kpc)}",
        f"sigma{int(mass_radius_kpc)}",
    )
    normalized = _normalize_gamma_mode(gamma_mode)
    if normalized == GAMMA_MODE_DEPENDENT:
        gamma_parameters = ("mu_gamma_0", "beta_gamma", "xi_gamma", "sigma_gamma")
    elif normalized == GAMMA_MODE_INDEPENDENT:
        gamma_parameters = ("mu_gamma_0", "sigma_gamma")
    elif normalized == GAMMA_MODE_SIGMA_STAR_DEPENDENT:
        gamma_parameters = ("mu_gamma_0", "beta_sigma_star_gamma", "sigma_gamma")
    else:
        raise ValueError(f"Unsupported legacy PPC gamma mode '{gamma_mode}'.")
    return mass_parameters + gamma_parameters + ("mu_zs", "sigma_zs", "theta0", "loga")


def load_legacy_ppc_runtime_config(config_path: Path, raw_data: dict[str, Any]) -> RuntimeConfig:
    """
    Parse a pre-registry CMASS PPC config snapshot into current dataclasses.

    The parser rejects any snapshot that explicitly names a non-CMASS model.
    This keeps old raw observation/cross-section handling from becoming an
    accidental compatibility path for newer registry-driven models.
    """

    model_raw = raw_data.get("model")
    if isinstance(model_raw, dict) and str(model_raw.get("name", "cmass")) != "cmass":
        raise ValueError("Legacy raw-path PPC config parsing only supports model.name='cmass'.")

    profile_raw = raw_data["profile"]
    mass_raw = raw_data["mass_definition"]
    gamma_raw = raw_data.get("gamma_model", {"mode": GAMMA_MODE_DEPENDENT})
    data_raw = raw_data["data"]
    sampling_raw = raw_data["sampling"]
    integration_raw = raw_data["integration"]
    cosmology_raw = raw_data["cosmology"]
    runtime_raw = raw_data["runtime"]
    output_raw = raw_data["output"]

    mass_radius_kpc = int(mass_raw["enclosed_radius_kpc"])
    unit_convention = str(raw_data.get("unit_convention", LEGACY_FIXED_KPC))
    mass_definition = get_mass_definition(mass_radius_kpc, unit_convention=unit_convention)
    gamma_mode = _normalize_gamma_mode(str(gamma_raw["mode"]))
    if gamma_mode is None:
        raise ValueError(f"Unsupported legacy PPC gamma mode '{gamma_raw['mode']}'.")

    parameter_order = _legacy_ppc_parameter_order(mass_radius_kpc, gamma_mode)
    box_prior_raw = raw_data["box_prior"]
    prior_bounds = tuple(tuple(float(value) for value in box_prior_raw[name]) for name in parameter_order)
    parameter_schema = ParameterSchema(
        model_name="cmass",
        model_component_key=gamma_mode,
        internal_parameter_names=parameter_order,
        public_parameter_names=parameter_order,
        prior_bounds=prior_bounds,
        static_codes={"gamma_mode": _gamma_mode_code_from_name(gamma_mode)},
        model_metadata={"gamma_distribution": gamma_mode},
    )
    initial_center = HyperParams.from_public_dict(
        public_values=sampling_raw["initial_center"],
        parameter_schema=parameter_schema,
    )

    cosmology_h0 = float(cosmology_raw["h0"])
    return RuntimeConfig(
        unit_convention=unit_convention,
        h_ref=cosmology_h0 / 100.0,
        profile=ProfileConfig(name=str(profile_raw["name"])),
        model=ModelConfig(name="cmass"),
        mass_definition=mass_definition,
        parameter_schema=parameter_schema,
        fp_prior=FPPriorConfig(enabled=False),
        data=DataConfig(
            observation_path=Path(data_raw["observation_path"]).expanduser().resolve(),
            cross_section_path=Path(data_raw["cross_section_path"]).expanduser().resolve(),
            sigma_table_path=(
                None
                if data_raw.get("sigma_table_path") is None
                else Path(data_raw["sigma_table_path"]).expanduser().resolve()
            ),
        ),
        sampling=SamplingConfig(
            random_seed=int(sampling_raw["random_seed"]),
            initial_center=initial_center,
            initial_jitter_scale=float(sampling_raw.get("initial_jitter_scale", 1.0e-3)),
            n_walkers=int(sampling_raw.get("n_walkers", 24)),
            n_steps=int(sampling_raw["n_steps"]),
            burn_in=int(sampling_raw.get("burn_in", sampling_raw.get("warmup", 0))),
        ),
        integration=IntegrationConfig(
            gamma_points=int(integration_raw["gamma_points"]),
            mstar_points=int(integration_raw["mstar_points"]),
            normalization_samples=int(integration_raw["normalization_samples"]),
        ),
        cosmology=CosmologyConfig(
            h0=cosmology_h0,
            omega_m=float(cosmology_raw["omega_m"]),
        ),
        runtime=RuntimeOptions(
            checkpoint_every=int(runtime_raw.get("checkpoint_every", 1)),
            parallel_strategy=str(runtime_raw.get("parallel_strategy", "off")),
            progress=bool(runtime_raw.get("progress", False)),
            progress_summary_every=int(runtime_raw.get("progress_summary_every", 1)),
            show_stage_timing=bool(runtime_raw.get("show_stage_timing", False)),
            disable_hdf5_file_locking=bool(runtime_raw.get("disable_hdf5_file_locking", False)),
            num_threads=int(runtime_raw.get("num_threads", 0)),
            reserve_cores=int(runtime_raw.get("reserve_cores", 0)),
        ),
        output=OutputConfig(
            root_dir=Path(output_raw.get("root_dir", config_path.parent)).expanduser().resolve(),
            run_label=str(output_raw.get("run_label", config_path.parent.name)),
            overwrite_latest=bool(output_raw.get("overwrite_latest", True)),
        ),
    )


__all__ = ["load_legacy_ppc_runtime_config"]
