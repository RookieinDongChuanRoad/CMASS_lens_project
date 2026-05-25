# Statistical_SL Workspace

`Statistical_SL` is the integrated strong-lensing workspace for reusable code,
run configuration, local data, and generated outputs.

The repository is organized around a single importable package,
`statistical_sl`, and a separate `workspace/` tree for analysis-facing files.

## Layout

- `src/statistical_sl/`: reusable Python package code
- `workspace/configs/`: single-step configs
- `workspace/recipes/`: pipeline recipes
- `workspace/data/`: local raw, external, and canonical data roots
- `workspace/outputs/`: per-run output directories
- `workspace/scripts/`, `workspace/notebooks/`, `workspace/reports/`: user-facing work files
- `tests/`: package and workflow checks
- `docs/`: design and migration notes

## Install

Use the checked-in conda environment `cmass_lens`, then install the package
editable from the repository root:

```bash
cd /Users/liurongfu/Work/CMASS_lens_project
conda run -n cmass_lens python -m pip install -e . --no-deps
```

## CLI

The public command is `statistical-sl`.

Help for the three workflow groups:

```bash
statistical-sl prepare-dataset --help
statistical-sl inference --help
statistical-sl posterior-predictive --help
```

## Workspace contract

- New runtime data should live under `workspace/data/`
- New run output should live under `workspace/outputs/`
- Each run should keep its own directory and then split inference and
  posterior-predictive artifacts underneath that run directory
- Configs should declare `workspace_root` explicitly in pipeline recipes

## Legacy material

Historical comparison data and archived implementation notes may remain in the
repository for reference, but they are not part of the default production path.
Current workflows should use the package and workspace layout above.
