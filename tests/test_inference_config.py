from __future__ import annotations

from statistical_sl.inference.config import load_runtime_config


def test_parallel_strategy_off_yaml_scalar_is_loaded_as_string(tmp_path) -> None:
    """Unquoted YAML `off` should mean the backend strategy, not Python False."""

    config_path = tmp_path / "toy_off_strategy.yaml"
    config_path.write_text(
        """
profile:
  name: toy
unit_convention: h_units_v1
model:
  name: toy_hierarchical
data:
  inference_dataset_path: workspace/data/canonical/toy_synthetic_context.hdf5
box_prior:
  population_mean: [-5.0, 5.0]
  log_population_scatter: [-5.0, 1.0]
sampling:
  n_walkers: 8
  n_steps: 6
  burn_in: 1
  random_seed: 11
  initial_jitter_scale: 1.0e-3
  initial_center:
    population_mean: 0.1
    log_population_scatter: -2.0
integration:
  gamma_points: 1
  mstar_points: 1
  normalization_samples: 1
cosmology:
  h0: 70.0
  omega_m: 0.3
runtime:
  checkpoint_every: 1
  parallel_strategy: off
  progress: false
  progress_summary_every: 1
  show_stage_timing: false
  disable_hdf5_file_locking: false
  num_threads: 1
  reserve_cores: 0
output:
  root_dir: workspace/outputs/_verification
  run_label: toy-smoke
  overwrite_latest: true
""".strip(),
        encoding="utf-8",
    )

    runtime_config = load_runtime_config(config_path)

    assert runtime_config.runtime.parallel_strategy == "off"
