"""Shared manifest and run-layout names for Statistical_SL workflows.

Workflow stages write related JSON files under run-owned directories.  This
module defines the stable names shared by data preparation, inference, and
posterior predictive code without giving ``core`` ownership over file I/O.
"""

from __future__ import annotations

RUN_MANIFEST_FILENAME = "run_manifest.json"
CONFIG_SNAPSHOTS_DIRNAME = "config_snapshots"
DATA_PREPARATION_DIRNAME = "data_preparation"
INFERENCE_DIRNAME = "inference"
POSTERIOR_PREDICTIVE_DIRNAME = "posterior_predictive"
DIAGNOSTICS_DIRNAME = "diagnostics"

__all__ = [
    "CONFIG_SNAPSHOTS_DIRNAME",
    "DATA_PREPARATION_DIRNAME",
    "DIAGNOSTICS_DIRNAME",
    "INFERENCE_DIRNAME",
    "POSTERIOR_PREDICTIVE_DIRNAME",
    "RUN_MANIFEST_FILENAME",
]
