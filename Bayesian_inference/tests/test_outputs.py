"""
Tests for run-directory creation and the `latest` pointer contract.

The output layer is intentionally tested before implementation because the
directory layout is part of the external contract for long-running sampling
jobs, restarts, and later profile-to-profile comparisons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cmass_lens_inference.outputs import (
    create_run_layout,
    load_emcee_checkpoint,
    refresh_latest_pointer,
    save_emcee_checkpoint,
)


def test_create_run_layout_uses_profile_bucket_and_readable_run_id(tmp_path: Path) -> None:
    """
    Run directories must live under `<root>/<profile>/<run_id>` and include the
    required subdirectories for checkpoints and logs.
    """

    run_layout = create_run_layout(
        root_dir=tmp_path,
        profile_name="sersic",
        run_label="baseline",
        timestamp_text="20260308_161500",
    )

    assert run_layout.profile_dir == tmp_path / "sersic"
    assert run_layout.run_id == "20260308_161500_sersic_baseline"
    assert run_layout.run_dir == tmp_path / "sersic" / "20260308_161500_sersic_baseline"
    assert run_layout.checkpoints_dir == run_layout.run_dir / "checkpoints"
    assert run_layout.logs_dir == run_layout.run_dir / "logs"


def test_refresh_latest_pointer_falls_back_to_text_file_when_symlink_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    The implementation should prefer a symlink, but it must degrade gracefully.

    The test forces the symlink creation path to fail so we can lock in the
    fallback behavior on platforms or filesystems where symlinks are not
    available.
    """

    run_layout = create_run_layout(
        root_dir=tmp_path,
        profile_name="devauc",
        run_label="baseline",
        timestamp_text="20260308_161500",
    )

    def raise_oserror(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError("synthetic symlink failure")

    monkeypatch.setattr(Path, "symlink_to", raise_oserror)
    refresh_latest_pointer(run_layout.profile_dir, run_layout.run_id)

    latest_file = run_layout.profile_dir / "LATEST_RUN"
    assert latest_file.read_text(encoding="utf-8").strip() == run_layout.run_id


def test_save_emcee_checkpoint_writes_all_required_files(tmp_path: Path) -> None:
    """
    The checkpoint writer should persist emcee walker coordinates and log_prob.

    `chain.h5` is the production chain source of truth.  The checkpoint arrays
    are a lightweight resume safety net, so their shape and step marker must
    remain stable.
    """

    run_layout = create_run_layout(
        root_dir=tmp_path,
        profile_name="sersic",
        run_label="resume",
        timestamp_text="20260308_170000",
    )

    walker_coords = np.ones((24, 3))
    log_prob = np.linspace(-1.0, 1.0, 24)
    save_emcee_checkpoint(
        run_layout.checkpoints_dir,
        walker_coords=walker_coords,
        log_prob=log_prob,
        step=9,
    )

    assert (run_layout.checkpoints_dir / "latest_walker_coords.npy").exists()
    assert (run_layout.checkpoints_dir / "latest_log_prob.npy").exists()
    assert (run_layout.checkpoints_dir / "latest_step.txt").read_text(encoding="utf-8").strip() == "9"

    loaded_coords, loaded_log_prob, loaded_step = load_emcee_checkpoint(run_layout.checkpoints_dir)
    np.testing.assert_allclose(loaded_coords, walker_coords)
    np.testing.assert_allclose(loaded_log_prob, log_prob)
    assert loaded_step == 9
