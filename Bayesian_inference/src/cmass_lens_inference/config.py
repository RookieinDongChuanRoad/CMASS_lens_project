"""
Configuration parsing.

The loader converts YAML into typed dataclasses immediately so downstream code
works with validated, explicit objects rather than raw nested dictionaries.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .mass_definition import get_mass_definition
from .parameter_schema import GammaModelConfig, build_parameter_schema
from .types import (
    CosmologyConfig,
    DataConfig,
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
    gamma_model = _load_gamma_model_section(path, raw_data)

    mass_definition = get_mass_definition(mass_definition_raw["enclosed_radius_kpc"])
    initial_center_raw = _require_section(sampling_raw, "initial_center")
    parameter_schema = build_parameter_schema(
        gamma_model=gamma_model,
        mass_definition=mass_definition,
    )
    initial_center = HyperParams.from_public_dict(
        public_values=initial_center_raw,
        parameter_schema=parameter_schema,
    )

    return RuntimeConfig(
        profile=ProfileConfig(name=str(profile_raw["name"])),
        mass_definition=mass_definition,
        gamma_model=gamma_model,
        parameter_schema=parameter_schema,
        data=DataConfig(
            observation_path=Path(data_raw["observation_path"]).expanduser().resolve(),
            cross_section_path=Path(data_raw["cross_section_path"]).expanduser().resolve(),
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
