# Statistical_SL Workspace

This directory is the canonical user workspace for the integrated repository.
It contains run-facing configuration, recipes, scripts, notebooks, reports,
local data roots, and local output roots.  Reusable Python code belongs under
`src/statistical_sl/`; workflow artifacts belong here.

The workspace is designed so it can later be split into a separate repository:

- `configs/` contains single-step configuration for advanced reproduction.
- `recipes/` contains pipeline-level recipes for daily end-to-end runs.
- `scripts/`, `notebooks/`, and `reports/` contain user-facing analysis work.
- `data/` contains local input and intermediate data products.
- `outputs/` contains run directories, with one directory per run.

Large data files and generated outputs are intentionally ignored by git.  The
tracked README files document the expected layout without versioning local
research artifacts.
