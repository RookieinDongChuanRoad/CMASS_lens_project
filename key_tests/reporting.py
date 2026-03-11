"""
Utilities for summarizing isolated key-test runs into a Markdown report.

The report is intentionally concise but machine-generated from summary JSONs so
it stays synchronized with the actual run artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import emcee
import numpy as np

from workspace_support import CURRENT_PARAMETER_ORDER, REFERENCE_PARAMETER_ORDER, WORKSPACE_ROOT


def _load_summary(path: Path) -> dict[str, object]:
    """Read one run summary JSON into memory."""

    return json.loads(path.read_text(encoding="utf-8"))


def _load_log_prob_stats(summary: dict[str, object]) -> dict[str, float | int]:
    """Read chain/log-prob statistics from one chain artifact."""

    chain_path = Path(str(summary["chain_path"]))
    discard = int(summary["discard"])
    backend = emcee.backends.HDFBackend(str(chain_path), read_only=True)
    flat_chain = backend.get_chain(flat=True, discard=discard)
    flat_log_prob = backend.get_log_prob(flat=True, discard=discard)
    return {
        "raw_steps": int(backend.iteration),
        "flat_samples": int(flat_chain.shape[0]),
        "median_log_prob": float(np.median(flat_log_prob)),
    }


def write_report(summary_paths: list[Path], report_path: Path) -> Path:
    """Generate the structured comparison report from all run summaries."""

    summaries = [_load_summary(path) for path in summary_paths]
    lines = [
        "# Pipeline Comparison Report",
        "",
        "## Scope",
        "",
        "- 当前实现与参考实现均在 `key_tests` 隔离工作区内运行。",
        "- 两套实现统一读取 `/Users/liurongfu/Work/CMASS_lens_project/data` 下的数据。",
        "- `smoke` 仅用于链路打通检查，不作为科学比较结果。",
        "- `compare` 是正式长跑对比模式，本轮固定为 `10000` 步并统一丢弃前 `2000` 步样本。",
        "",
        "## Parameter Order Notes",
        "",
        f"- Current order: `{', '.join(CURRENT_PARAMETER_ORDER)}`",
        f"- Reference order: `{', '.join(REFERENCE_PARAMETER_ORDER)}`",
        "- 参考实现最后两维是 `loga, theta0`；corner notebook 会在读取后重排为当前实现顺序。",
        "",
        "## Run Summary",
        "",
        "| Implementation | Profile | Mode | Requested Steps | Raw Steps | Discard | Flat Samples | Median Log Prob | Chain Path |",
        "|----------------|---------|------|-----------------|-----------|---------|--------------|-----------------|------------|",
    ]

    for summary in summaries:
        stats = _load_log_prob_stats(summary)
        lines.append(
            "| "
            f"{summary['implementation']} | "
            f"{summary['profile_name']} | "
            f"{summary['mode_name']} | "
            f"{summary['requested_steps']} | "
            f"{stats['raw_steps']} | "
            f"{summary['discard']} | "
            f"{stats['flat_samples']} | "
            f"{stats['median_log_prob']:.6f} | "
            f"`{summary['chain_path']}` |"
        )

    lines.extend(
        [
            "",
            "## Observed Differences",
            "",
            "- 代码实现差异：当前实现使用模块化 compiled context 与 richer metadata；参考实现仍保持旧脚本结构。",
            "- 输入数据差异：本轮已强制统一为 `CMASS_lens_project/data`，因此正式 compare 结果主要反映实现差异，而不是数据副本差异。",
            "- 参数顺序差异：参考实现的 `loga/theta0` 顺序与当前实现相反，已在 notebook 和报告中显式标注。",
            "",
            "## Output Locations",
            "",
            f"- Workspace root: `{WORKSPACE_ROOT}`",
            f"- Figures directory: `{WORKSPACE_ROOT / 'output' / 'figures'}`",
            f"- Notebook: `{WORKSPACE_ROOT / 'notebooks' / 'compare_corner.ipynb'}`",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
