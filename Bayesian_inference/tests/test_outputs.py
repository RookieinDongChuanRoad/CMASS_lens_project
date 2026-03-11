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
    refresh_latest_pointer,
    save_checkpoint,
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


def test_save_checkpoint_writes_all_required_files(tmp_path: Path) -> None:
    """
    The checkpoint writer should persist walker coordinates, log-probabilities,
    and the latest step number using the required filenames.
    """

    run_layout = create_run_layout(
        root_dir=tmp_path,
        profile_name="sersic",
        run_label="resume",
        timestamp_text="20260308_170000",
    )

    coords = np.ones((4, 3))
    log_prob = np.array([0.1, 0.2, 0.3, 0.4])
    save_checkpoint(run_layout.checkpoints_dir, coords, log_prob, step=9)

    assert (run_layout.checkpoints_dir / "latest_coords.npy").exists()
    assert (run_layout.checkpoints_dir / "latest_log_prob.npy").exists()
    assert (run_layout.checkpoints_dir / "latest_step.txt").read_text(encoding="utf-8").strip() == "9"

