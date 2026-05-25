"""
Orchestrate the full isolated key-tests workflow.

The script performs the plan in one reproducible command:

1. prepare the workspace
2. run current/reference smoke jobs for both profiles
3. run current/reference compare jobs for both profiles
4. write the report

Notebook creation and execution remain separate so they can be rerun without
repeating MCMC jobs.
"""

from __future__ import annotations

import json
from pathlib import Path

from current_pipeline.run_current_pipeline import run_current_pipeline
from reference_pipeline.run_reference_pipeline import run_reference_pipeline
from reporting import write_report
from setup_workspace import prepare_workspace
from workspace_support import WORKSPACE_ROOT, archive_existing_compare_artifacts


def run_all() -> dict[str, object]:
    """
    Execute the full planned smoke + compare comparison workflow.

    Before any run starts, stale compare artifacts are archived out of the
    canonical output/report locations so the regenerated long-run results are
    the only compare artifacts living under the default paths.
    """

    setup_summary = prepare_workspace()
    archive_summary = archive_existing_compare_artifacts()
    summary_paths: list[Path] = []
    run_summaries: list[dict[str, object]] = []

    for mode_name in ("smoke", "compare"):
        for profile_name in ("sersic", "devauc"):
            current_summary = run_current_pipeline(profile_name, mode_name)
            current_summary_path = Path(str(current_summary["output_root"])) / "current_run_summary.json"
            summary_paths.append(current_summary_path)
            run_summaries.append(current_summary)

            reference_summary = run_reference_pipeline(profile_name, mode_name)
            reference_summary_path = Path(str(reference_summary["summary_path"]))
            summary_paths.append(reference_summary_path)
            run_summaries.append(reference_summary)

    report_path = write_report(
        summary_paths=summary_paths,
        report_path=WORKSPACE_ROOT / "reports" / "pipeline_comparison.md",
    )
    payload = {
        "setup_summary": setup_summary,
        "archive_summary": archive_summary,
        "run_summaries": run_summaries,
        "report_path": str(report_path),
    }
    result_path = WORKSPACE_ROOT / "reports" / "run_manifest.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2, sort_keys=True))
