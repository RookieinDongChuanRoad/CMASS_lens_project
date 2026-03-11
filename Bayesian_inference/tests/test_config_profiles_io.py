"""
Tests for configuration loading, profile selection, and HDF5 compatibility.

These tests lock the public API for the configuration and input layers before
the implementation exists. The goal is to make the structure explicit:
configuration must produce typed objects, profile-specific aliases must be
resolved centrally, and input files must be converted into normalized runtime
records that the numerical code can consume.
"""

from __future__ import annotations

from pathlib import Path

from cmass_lens_inference.config import load_runtime_config
from cmass_lens_inference.io import load_cross_section_grid, load_observations
from cmass_lens_inference.profiles import build_profile_spec


def test_load_runtime_config_builds_typed_sections(synthetic_config_path: Path) -> None:
    """
    The configuration loader should convert YAML into a typed runtime config.

    This test locks the project's public contract:
    - profile, sampling, integration, runtime, and output sections exist
    - the configured output root is preserved
    - default run labels remain user-visible because they are part of `run_id`
    """

    runtime_config = load_runtime_config(synthetic_config_path)

    assert runtime_config.profile.name == "sersic"
    assert runtime_config.sampling.n_walkers == 24
    assert runtime_config.integration.normalization_samples == 128
    assert runtime_config.output.run_label == "synthetic"
    assert runtime_config.output.root_dir == synthetic_config_path.parent / "outputs"
    assert runtime_config.runtime.parallel_strategy == "auto"
    assert runtime_config.runtime.num_threads == 0
    assert runtime_config.runtime.reserve_cores == 2
    assert runtime_config.runtime.progress_summary_every == 1
    assert runtime_config.runtime.show_stage_timing is True


def test_build_profile_spec_exposes_profile_specific_rules() -> None:
    """
    The profile builder should isolate all devauc/sersic differences.

    This protects the main inference pipeline from accumulating profile
    conditionals in numerically sensitive code paths.
    """

    devauc = build_profile_spec("devauc")
    sersic = build_profile_spec("sersic")

    assert devauc.fixed_n == 4.0
    assert devauc.uses_observed_n_in_likelihood is False
    assert devauc.observation_field_aliases["stellar_mass"] == ("logmchab_deV", "logmchab")
    assert devauc.observation_field_aliases["effective_radius_arcsec"] == ("reff_deV", "re_arcsec")

    assert sersic.fixed_n is None
    assert sersic.uses_observed_n_in_likelihood is True
    assert sersic.observation_field_aliases["stellar_mass"] == ("logmchab",)
    assert sersic.observation_field_aliases["effective_radius_arcsec"] == ("re_arcsec",)


def test_load_observations_uses_devauc_aliases(
    synthetic_devauc_observation_file: Path,
) -> None:
    """
    The devauc reader must prefer de Vaucouleurs-specific aliases when present.

    This is a critical compatibility rule from the requirements document and
    must be enforced in the I/O layer rather than inside the statistical model.
    """

    profile_spec = build_profile_spec("devauc")
    observations = load_observations(synthetic_devauc_observation_file, profile_spec)

    assert len(observations) == 1
    observation = observations[0]

    assert observation.lens_id == "lens-devauc"
    assert observation.log_stellar_mass_obs == 11.1
    assert observation.log_stellar_mass_err == 0.04
    assert observation.effective_radius_arcsec == 1.4
    assert observation.n_observed == 4.0
    assert observation.num_sigma == 0


def test_load_cross_section_grid_supports_real_world_alias_names(
    synthetic_cross_section_file: Path,
) -> None:
    """
    The cross-section loader must accept the alias field names observed in the
    real `cs_grid_power.h5` file.
    """

    cross_section_grid = load_cross_section_grid(synthetic_cross_section_file)

    assert cross_section_grid.gamma_grid.shape == (25,)
    assert cross_section_grid.cs_over_theta_ein.shape == (25,)
    assert cross_section_grid.gamma_grid[0] == 1.2
