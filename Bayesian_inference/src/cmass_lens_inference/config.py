"""
Configuration parsing.

The loader converts YAML into typed dataclasses immediately so downstream code
works with validated, explicit objects rather than raw nested dictionaries.
Scientific model choices are now resolved exclusively through
``model.name``/``model.components`` and the model registry.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .mass_definition import validate_h_ref, validate_unit_convention
from .model_registry import get_model_definition
from .types import (
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


DEFAULT_OUTPUT_ROOT = Path("/Users/liurongfu/Work/CMASS_lens_project/outputs")
REMOVED_TOP_LEVEL_MODEL_SECTIONS = ("mass_definition", "gamma_model")


def _require_section(data: dict, section_name: str) -> dict:
    """Extract a mandatory configuration section with a clear error message."""

    if section_name not in data:
        raise KeyError(f"Missing required config section: {section_name}")
    section = data[section_name]
    if not isinstance(section, dict):
        raise TypeError(f"Config section '{section_name}' must be a mapping.")
    return section


def _reject_removed_model_sections(raw_data: dict) -> None:
    """
    Reject YAML fields that were removed by the model-registry refactor.

    The old config surface split one scientific model across top-level
    ``mass_definition`` and ``gamma_model`` sections.  Keeping those fields
    around would make it ambiguous whether the registry or legacy parser owns
    the model choice, so the loader fails before normalizing anything else.
    """

    removed_present = [name for name in REMOVED_TOP_LEVEL_MODEL_SECTIONS if name in raw_data]
    if removed_present:
        raise ValueError(
            "Top-level mass_definition and gamma_model sections are no longer supported. "
            "Move model-specific choices under model.components, for example "
            "model.components.mass_definition and model.components.gamma_distribution. "
            f"Removed sections present: {', '.join(removed_present)}."
        )


def _load_fp_prior_section(raw_data: dict) -> FPPriorConfig:
    """
    Load the optional FP-prior section with the 1D sigma-logM* defaults.

    The section is intentionally optional so existing model configs can opt in
    only when the required sigma-unit table is available.
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


def _load_unit_convention(raw_data: dict) -> str:
    """Load the global unit convention required by model components and I/O."""

    if "unit_convention" not in raw_data:
        raise KeyError("Missing required config field: unit_convention")
    return validate_unit_convention(raw_data["unit_convention"])


def _load_model_section(raw_data: dict) -> tuple[ModelConfig, object]:
    """
    Load and normalize the scientific model registry selection.

    A model definition owns its allowed component keys.  The config layer only
    requires ``model.name`` and delegates component validation to the registry.
    """

    model_raw = _require_section(raw_data, "model")
    model_name = str(model_raw["name"])
    model_definition = get_model_definition(model_name)

    component_overrides = model_raw.get("components")
    if component_overrides is not None and not isinstance(component_overrides, dict):
        raise TypeError("Config section 'model.components' must be a mapping.")
    components = model_definition.normalize_components(component_overrides)
    return ModelConfig(name=model_name, components=components), model_definition


def _load_box_prior_section(raw_data: dict) -> dict:
    """Load the required explicit box-prior mapping."""

    return _require_section(raw_data, "box_prior")


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

    _reject_removed_model_sections(raw_data)

    profile_raw = _require_section(raw_data, "profile")
    model, model_definition = _load_model_section(raw_data)
    data_raw = _require_section(raw_data, "data")
    sampling_raw = _require_section(raw_data, "sampling")
    integration_raw = _require_section(raw_data, "integration")
    cosmology_raw = _require_section(raw_data, "cosmology")
    runtime_raw = _require_section(raw_data, "runtime")
    output_raw = _require_section(raw_data, "output")
    unit_convention = _load_unit_convention(raw_data)
    h_ref = validate_h_ref(float(cosmology_raw["h0"]) / 100.0)
    mass_definition = model_definition.resolve_mass_definition(
        model.components,
        unit_convention,
    )
    box_prior_raw = _load_box_prior_section(raw_data)
    fp_prior = _load_fp_prior_section(raw_data)

    sigma_table_path_raw = data_raw.get("sigma_table_path")
    sigma_table_path = (
        Path(sigma_table_path_raw).expanduser().resolve()
        if sigma_table_path_raw is not None
        else None
    )
    if fp_prior.enabled and sigma_table_path is None:
        raise ValueError("FP prior requires data.sigma_table_path when fp_prior.enabled is true.")

    parameter_schema = model_definition.build_parameter_schema(
        components=model.components,
        mass_definition=mass_definition,
        public_box_prior=box_prior_raw,
    )
    initial_center_raw = _require_section(sampling_raw, "initial_center")
    initial_center = HyperParams.from_public_dict(
        public_values=initial_center_raw,
        parameter_schema=parameter_schema,
    )
    parameter_schema.validate_theta_in_bounds(
        initial_center.to_array(),
        label="Initial center",
    )

    n_steps = int(sampling_raw.get("n_steps", sampling_raw.get("num_samples", 0)))
    warmup = int(sampling_raw.get("warmup", sampling_raw.get("num_warmup", 0)))
    n_walkers = int(sampling_raw.get("n_walkers", sampling_raw.get("num_chains", 1)))

    return RuntimeConfig(
        unit_convention=unit_convention,
        h_ref=h_ref,
        profile=ProfileConfig(name=str(profile_raw["name"])),
        model=model,
        mass_definition=mass_definition,
        parameter_schema=parameter_schema,
        fp_prior=fp_prior,
        data=DataConfig(
            observation_path=Path(data_raw["observation_path"]).expanduser().resolve(),
            cross_section_path=Path(data_raw["cross_section_path"]).expanduser().resolve(),
            sigma_table_path=sigma_table_path,
        ),
        sampling=SamplingConfig(
            n_walkers=n_walkers,
            n_steps=n_steps,
            warmup=warmup,
            random_seed=int(sampling_raw["random_seed"]),
            initial_center=initial_center,
            initial_jitter_scale=float(sampling_raw.get("initial_jitter_scale", 1.0e-3)),
            num_chains=int(sampling_raw.get("num_chains", max(1, 2 * parameter_schema.n_dim))),
            num_samples=int(sampling_raw.get("num_samples", n_steps)),
            num_warmup=int(sampling_raw.get("num_warmup", warmup)),
            thinning=int(sampling_raw.get("thinning", 1)),
            chain_method=str(sampling_raw.get("chain_method", "sequential")),
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
