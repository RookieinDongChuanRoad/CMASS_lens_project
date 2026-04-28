"""
Configuration parsing.

The loader converts YAML into typed dataclasses immediately so downstream code
works with validated, explicit objects rather than raw nested dictionaries.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .mass_definition import (
    H_UNITS_V1,
    LEGACY_FIXED_KPC,
    get_mass_definition,
    validate_h_ref,
    validate_unit_convention,
)
from .parameter_schema import GammaModelConfig, build_parameter_schema, default_public_box_prior
from .types import (
    CosmologyConfig,
    DataConfig,
    FPPriorConfig,
    HyperParams,
    IntegrationConfig,
    OutputConfig,
    ProfileConfig,
    RuntimeConfig,
    RuntimeOptions,
    SamplingConfig,
)


DEFAULT_OUTPUT_ROOT = Path("/Users/liurongfu/Work/CMASS_lens_project/outputs")


def _require_section(data: dict, section_name: str) -> dict:
    """Extract a mandatory configuration section with a clear error message."""

    if section_name not in data:
        raise KeyError(f"Missing required config section: {section_name}")
    section = data[section_name]
    if not isinstance(section, dict):
        raise TypeError(f"Config section '{section_name}' must be a mapping.")
    return section


def _load_gamma_model_section(path: Path, raw_data: dict) -> GammaModelConfig:
    """
    Load the gamma-model section and migrate legacy run snapshots when needed.

    Only stored run snapshots are auto-migrated because those files are part of
    the pipeline's own managed state. User-authored source configs must declare
    the mode explicitly so the chosen scientific model is always unambiguous.
    """

    if "gamma_model" not in raw_data:
        if path.name != "config_snapshot.yaml":
            raise KeyError("Missing required config section: gamma_model")

        raw_data["gamma_model"] = {"mode": "dependent"}
        path.write_text(yaml.safe_dump(raw_data, sort_keys=False), encoding="utf-8")

    gamma_model_raw = _require_section(raw_data, "gamma_model")
    return GammaModelConfig(mode=str(gamma_model_raw["mode"]))


def _load_fp_prior_section(raw_data: dict) -> FPPriorConfig:
    """
    Load the optional FP-prior section with the 1D sigma-logM* defaults.

    The section is intentionally optional so existing configurations keep the
    same posterior unless users opt in explicitly.
    """

    fp_prior_raw = raw_data.get("fp_prior")
    if fp_prior_raw is None:
        return FPPriorConfig(enabled=False)
    if not isinstance(fp_prior_raw, dict):
        raise TypeError("Config section 'fp_prior' must be a mapping.")

    return FPPriorConfig(
        enabled=bool(fp_prior_raw.get("enabled", False)),
        fit_mstar_min=float(fp_prior_raw.get("fit_mstar_min", 11.0)),
        pivot_mstar=float(fp_prior_raw.get("pivot_mstar", 11.3)),
        fiducial_scatter=float(fp_prior_raw.get("fiducial_scatter", 0.075)),
        scatter_error=float(fp_prior_raw.get("scatter_error", 0.003)),
        mu_v_prior=float(fp_prior_raw.get("mu_v_prior", 2.34548)),
        mu_v_error=float(fp_prior_raw.get("mu_v_error", 0.00611)),
        beta_v_prior=float(fp_prior_raw.get("beta_v_prior", 0.176)),
        beta_v_error=float(fp_prior_raw.get("beta_v_error", 0.011)),
    )


def _load_box_prior_section(
    path: Path,
    raw_data: dict,
    *,
    gamma_model: GammaModelConfig,
    mass_definition,
) -> dict:
    """
    Load the required explicit box-prior mapping.

    User-authored source configs must declare the section explicitly. Historical
    run snapshots are the only files that may be auto-migrated because those
    files are pipeline-owned state.
    """

    if "box_prior" not in raw_data:
        if path.name != "config_snapshot.yaml":
            raise KeyError("Missing required config section: box_prior")

        raw_data["box_prior"] = default_public_box_prior(
            gamma_model=gamma_model,
            mass_definition=mass_definition,
        )
        path.write_text(yaml.safe_dump(raw_data, sort_keys=False), encoding="utf-8")

    return _require_section(raw_data, "box_prior")


def _resolve_unit_convention(raw_data: dict, mass_definition_raw: dict) -> str:
    """
    Resolve the run's unit convention from top-level config and legacy shape.

    New source configs should set `unit_convention` explicitly. The fallback is
    intentionally conservative for existing project fixtures and historical
    configs: if an old config only declares `enclosed_radius_kpc`, it is treated
    as `legacy_fixed_kpc` instead of silently reinterpreting the same number as
    an h-dependent aperture.
    """

    if "unit_convention" in raw_data:
        return validate_unit_convention(raw_data["unit_convention"])
    if "enclosed_radius_kpc" in mass_definition_raw:
        return LEGACY_FIXED_KPC
    return H_UNITS_V1


def _load_mass_definition(mass_definition_raw: dict, unit_convention: str):
    """
    Load the convention-aware mass definition from the public YAML surface.

    The key names are part of the data contract:
    - h-units runs use `aperture_hinv_kpc` because `5` means `5 h^-1 kpc`
    - legacy runs use `enclosed_radius_kpc` because `5` means fixed physical kpc
    """

    if unit_convention == H_UNITS_V1:
        if "aperture_hinv_kpc" not in mass_definition_raw:
            raise KeyError(
                "h_units_v1 requires mass_definition.aperture_hinv_kpc "
                "with value 5 or 10."
            )
        return get_mass_definition(
            mass_definition_raw["aperture_hinv_kpc"],
            unit_convention=H_UNITS_V1,
        )

    if "enclosed_radius_kpc" not in mass_definition_raw:
        raise KeyError(
            "legacy_fixed_kpc requires mass_definition.enclosed_radius_kpc "
            "with value 5 or 10."
        )
    return get_mass_definition(
        mass_definition_raw["enclosed_radius_kpc"],
        unit_convention=LEGACY_FIXED_KPC,
    )


def load_runtime_config(config_path: str | Path) -> RuntimeConfig:
    """
    Load, validate, and normalize the project YAML configuration.

    Why this function exists:
    - The scientific pipeline depends on many numeric controls.
    - A typo in a raw dictionary would be easy to miss and difficult to debug.
    - Parsing once into dataclasses makes later modules significantly clearer.
    """

    path = Path(config_path).expanduser().resolve()
    raw_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise TypeError("Top-level configuration must be a YAML mapping.")

    profile_raw = _require_section(raw_data, "profile")
    mass_definition_raw = _require_section(raw_data, "mass_definition")
    data_raw = _require_section(raw_data, "data")
    sampling_raw = _require_section(raw_data, "sampling")
    integration_raw = _require_section(raw_data, "integration")
    cosmology_raw = _require_section(raw_data, "cosmology")
    runtime_raw = _require_section(raw_data, "runtime")
    output_raw = _require_section(raw_data, "output")
    unit_convention = _resolve_unit_convention(raw_data, mass_definition_raw)
    mass_definition = _load_mass_definition(
        mass_definition_raw=mass_definition_raw,
        unit_convention=unit_convention,
    )
    gamma_model = _load_gamma_model_section(path, raw_data)
    box_prior_raw = _load_box_prior_section(
        path,
        raw_data,
        gamma_model=gamma_model,
        mass_definition=mass_definition,
    )
    fp_prior = _load_fp_prior_section(raw_data)

    sigma_table_path_raw = data_raw.get("sigma_table_path")
    sigma_table_path = (
        Path(sigma_table_path_raw).expanduser().resolve()
        if sigma_table_path_raw is not None
        else None
    )
    if fp_prior.enabled and sigma_table_path is None:
        raise ValueError("FP prior requires data.sigma_table_path when fp_prior.enabled is true.")

    initial_center_raw = _require_section(sampling_raw, "initial_center")
    parameter_schema = build_parameter_schema(
        gamma_model=gamma_model,
        mass_definition=mass_definition,
        public_box_prior=box_prior_raw,
    )
    initial_center = HyperParams.from_public_dict(
        public_values=initial_center_raw,
        parameter_schema=parameter_schema,
    )
    parameter_schema.validate_theta_in_bounds(
        initial_center.to_array(),
        label="Initial center",
    )

    h_ref = validate_h_ref(float(cosmology_raw["h0"]) / 100.0)

    return RuntimeConfig(
        unit_convention=unit_convention,
        h_ref=h_ref,
        profile=ProfileConfig(name=str(profile_raw["name"])),
        mass_definition=mass_definition,
        gamma_model=gamma_model,
        parameter_schema=parameter_schema,
        fp_prior=fp_prior,
        data=DataConfig(
            observation_path=Path(data_raw["observation_path"]).expanduser().resolve(),
            cross_section_path=Path(data_raw["cross_section_path"]).expanduser().resolve(),
            sigma_table_path=sigma_table_path,
        ),
        sampling=SamplingConfig(
            n_walkers=int(sampling_raw["n_walkers"]),
            n_steps=int(sampling_raw["n_steps"]),
            warmup=int(sampling_raw["warmup"]),
            random_seed=int(sampling_raw["random_seed"]),
            initial_center=initial_center,
            initial_jitter_scale=float(sampling_raw.get("initial_jitter_scale", 1.0e-3)),
        ),
        integration=IntegrationConfig(
            gamma_points=int(integration_raw["gamma_points"]),
            mstar_points=int(integration_raw["mstar_points"]),
            normalization_samples=int(integration_raw["normalization_samples"]),
        ),
        cosmology=CosmologyConfig(
            h0=float(cosmology_raw["h0"]),
            omega_m=float(cosmology_raw["omega_m"]),
        ),
        runtime=RuntimeOptions(
            checkpoint_every=int(runtime_raw["checkpoint_every"]),
            parallel_strategy=str(runtime_raw.get("parallel_strategy", "auto")),
            progress=bool(runtime_raw["progress"]),
            progress_summary_every=int(runtime_raw.get("progress_summary_every", 25)),
            show_stage_timing=bool(runtime_raw.get("show_stage_timing", True)),
            disable_hdf5_file_locking=bool(runtime_raw["disable_hdf5_file_locking"]),
            num_threads=int(runtime_raw.get("num_threads", 0)),
            reserve_cores=int(runtime_raw.get("reserve_cores", 2)),
        ),
        output=OutputConfig(
            root_dir=Path(output_raw.get("root_dir", DEFAULT_OUTPUT_ROOT)).expanduser().resolve(),
            run_label=str(output_raw.get("run_label", "default")),
            overwrite_latest=bool(output_raw.get("overwrite_latest", True)),
        ),
    )
