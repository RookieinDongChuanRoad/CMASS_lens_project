from __future__ import annotations

from pathlib import Path

import pytest


POSTERIOR_CONFIG_ROOT = Path("workspace/configs/posterior_predictive")


def test_workspace_posterior_diagnostics_configs_lock_production_execution_contract() -> None:
    """Production PPC configs should encode the canonical full-chain diagnostics contract."""

    from statistical_sl.posterior_predictive.config import load_posterior_diagnostics_config

    config_paths = sorted(POSTERIOR_CONFIG_ROOT.glob("**/*_diagnostics.yaml"))
    assert config_paths, "Expected at least one workspace posterior diagnostics config."

    for config_path in config_paths:
        config = load_posterior_diagnostics_config(config_path)
        assert config.n_posterior_draws is None, config_path
        assert config.parent_sample_size == 10000, config_path
        assert config.worker_processes is None, config_path


def test_load_posterior_diagnostics_config_normalizes_paths() -> None:
    """The workspace YAML should become explicit run kwargs without running PPC."""

    from statistical_sl.posterior_predictive.config import load_posterior_diagnostics_config

    config = load_posterior_diagnostics_config(
        "workspace/configs/posterior_predictive/cmass/devauc_diagnostics.yaml"
    )

    assert config.model_name == "cmass"
    assert config.profile_name == "devauc"
    assert config.inference_run_dir is None
    assert config.sigma_table_path == Path("workspace/data/external/hunits_v1/jeans_deV_sigma_bundle.h5").resolve()
    assert config.output_root_dir == Path("workspace/outputs").resolve()
    assert config.n_posterior_draws is None
    assert config.burn_in == "auto"
    assert config.random_seed == 20260309
    assert config.parent_sample_size == 10000
    assert config.worker_processes is None
    assert config.n_mass_bins == 19
    assert config.mass_bin_min == 10.15
    assert config.mass_bin_max == 12.05


def test_posterior_diagnostics_config_requires_run_dir_from_yaml_or_cli() -> None:
    """A reusable config may omit run_dir, but execution kwargs may not."""

    from statistical_sl.posterior_predictive.config import load_posterior_diagnostics_config

    config = load_posterior_diagnostics_config(
        "workspace/configs/posterior_predictive/cmass/devauc_diagnostics.yaml"
    )

    with pytest.raises(ValueError, match="inference run directory"):
        config.to_run_kwargs()

    kwargs = config.to_run_kwargs(run_dir_override="workspace/outputs/devauc/latest")
    assert kwargs["run_dir"] == str(Path("workspace/outputs/devauc/latest").resolve())
    assert kwargs["sigma_table_path"] == Path("workspace/data/external/hunits_v1/jeans_deV_sigma_bundle.h5").resolve()
    assert kwargs["output_root_dir"] == Path("workspace/outputs").resolve()


def test_posterior_diagnostics_cli_merges_config_and_cli_overrides(monkeypatch, capsys) -> None:
    """The CLI should merge YAML defaults and explicit CLI overrides before dispatch."""

    import sys

    import statistical_sl.posterior_predictive.cli as posterior_cli

    captured_kwargs: dict[str, object] = {}

    class FakeDiagnosticsResult:
        def to_dict(self) -> dict[str, object]:
            return {"status": "completed", "result_dir": "unused"}

    def fake_run_posterior_diagnostics(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeDiagnosticsResult()

    monkeypatch.setattr(posterior_cli, "run_posterior_diagnostics", fake_run_posterior_diagnostics)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "statistical-sl posterior-predictive",
            "posterior-diagnostics",
            "--config",
            "workspace/configs/posterior_predictive/cmass/devauc_diagnostics.yaml",
            "--run-dir",
            "workspace/outputs/devauc/latest",
            "--diagnostic-run-id",
            "diagnostic-smoke",
            "--n-posterior-draws",
            "17",
            "--parent-sample-size",
            "321",
            "--worker-processes",
            "2",
            "--seed",
            "123",
        ],
    )

    posterior_cli.main()

    captured = capsys.readouterr()
    assert '"status": "completed"' in captured.out
    assert captured_kwargs["run_dir"] == str(Path("workspace/outputs/devauc/latest").resolve())
    assert captured_kwargs["diagnostic_run_id"] == "diagnostic-smoke"
    assert captured_kwargs["n_posterior_draws"] == 17
    assert captured_kwargs["parent_sample_size"] == 321
    assert captured_kwargs["worker_processes"] == 2
    assert captured_kwargs["random_seed"] == 123
    assert captured_kwargs["sigma_table_path"] == Path("workspace/data/external/hunits_v1/jeans_deV_sigma_bundle.h5").resolve()
