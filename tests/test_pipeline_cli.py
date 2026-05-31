from __future__ import annotations


def test_pipeline_validate_cli_accepts_devauc_post_canonical_recipe(capsys) -> None:
    """The public CLI should validate recipes without launching inference."""

    from statistical_sl.cli import main

    exit_code = main([
        "pipeline",
        "validate",
        "--recipe",
        "workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "cmass_devauc_diagnostics_from_canonical" in captured.out
    assert "valid" in captured.out


def test_pipeline_run_dry_run_cli_prints_planned_actions(capsys) -> None:
    """Dry-run is the safe default verification path for long scientific jobs."""

    from statistical_sl.cli import main

    exit_code = main([
        "pipeline",
        "run",
        "--recipe",
        "workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml",
        "--diagnostic-run-id",
        "diagnostic-smoke",
        "--dry-run",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "run_inference" in captured.out
    assert "run_posterior_corner" in captured.out
    assert "run_posterior_diagnostics" in captured.out
    assert "diagnostic-smoke" in captured.out


def test_pipeline_run_dry_run_does_not_call_scientific_runners(monkeypatch, capsys) -> None:
    """Dry-run must stay side-effect free even if the real runners are importable."""

    import statistical_sl.pipeline.runner as pipeline_runner
    from statistical_sl.cli import main

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("dry-run must not call scientific runners")

    monkeypatch.setattr(pipeline_runner, "run_inference", fail_if_called)
    monkeypatch.setattr(pipeline_runner, "run_posterior_corner", fail_if_called)
    monkeypatch.setattr(pipeline_runner, "run_posterior_diagnostics", fail_if_called)

    exit_code = main([
        "pipeline",
        "run",
        "--recipe",
        "workspace/recipes/cmass/devauc_diagnostics_from_canonical.yaml",
        "--diagnostic-run-id",
        "diagnostic-smoke",
        "--dry-run",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "run_inference" in captured.out
    assert "run_posterior_corner" in captured.out
    assert "run_posterior_diagnostics" in captured.out
