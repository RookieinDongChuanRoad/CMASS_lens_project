# Direct Data Preparation On Statistical_SL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. All Python commands must run in the `cmass_lens` environment.

**Goal:** Re-implement the 2026-05-21 direct data-preparation goals on top of the current `src/statistical_sl` package and `workspace/` layout, without reintroducing the old `prepare_dataset` package boundary.

**Architecture:** Treat the 2026-05-21 worktree as a validated source of contracts, tests, and implementation ideas, not as a branch to merge. Port the direct source-to-canonical pipeline into `statistical_sl.data_preparation`, keep shared schema strings in `statistical_sl.core`, keep numerical kernels in the existing data-preparation physics layer, and expose the workflow only through `statistical-sl prepare-dataset`.

**Tech Stack:** Python in `cmass_lens`, `numpy`, `h5py`, `astropy`, `PyYAML`, existing `statistical_sl.data_preparation.physics` kernels, and pytest.

---

## Source Plans Read

- Current integration plan: `docs/superpowers/plans/2026-05-24-repository-integration-structure.md`
- Earlier direct pipeline plan: `docs/superpowers/plans/2026-05-21-prepare-dataset-direct-canonical-pipeline.md`
- Earlier implementation progress: `.worktrees/prepare-dataset-direct-canonical-pipeline/docs/superpowers/plans/2026-05-21-prepare-dataset-direct-canonical-pipeline.progress.md`
- Earlier velocity contract: `.worktrees/prepare-dataset-direct-canonical-pipeline/prepare_dataset/docs/velocity_measurements_v1.md`

## Non-Negotiable Constraints

- Do not `git merge` the 2026-05-21 worktree into the current tree.
- Do not recreate `prepare_dataset/`, `Bayesian_inference/`, or `Posterior_predictive_test/` as production source roots.
- Do not introduce imports containing the old package identity `prepare_dataset`; `tests/test_dependency_boundaries.py` is expected to reject that. This also forbids wrappers, import aliases, `sys.path` edits, dynamic imports, or compatibility re-export modules that reach into old `prepare_dataset` code at runtime.
- Do not write new defaults pointing at root-level `data/` or `outputs/`; use `workspace/data` and `workspace/outputs`.
- Do not move direct-pipeline contracts into `core/` unless they are stable cross-workflow schema constants. Workflow orchestration belongs under `data_preparation`.
- Do not change science semantics while porting. Any required change to mass definitions, unit conventions, sigma likelihood semantics, cross-section boundary behavior, or model IDs must stop and become a separate design issue.
- Do not interrupt any running data generation, inference, diagnostics, or test job without explicit user permission.
- Do not treat this plan as optional guidance. Execution must move through these gates task-by-task; skipping tests, docs, config consumption, or boundary checks invalidates the completion claim.
- Do not land implementation changes directly in the current main worktree unless the user explicitly approves that choice for the implementation run. The default execution target is a fresh isolated worktree created from the current intended base branch. The existing 5.21 worktree remains read-only reference material, not the destination worktree.

## What 5.21 Actually Adds

The 2026-05-21 line adds a data-preparation architecture that starts from source tables and trusted velocity measurements, then writes a canonical inference HDF5 directly. Its important contracts are:

- Source catalogs describe lens facts; they are not automatically trusted velocity-dispersion sources.
- `velocity_measurements_v1` is the preferred upstream CSV contract for sigma measurements.
- The legacy pPXF CSV is only an adapter source, not the permanent interface.
- `num_sigma = 0` is a legal state for lenses missing trusted measurements.
- Accepted sigma rows must carry explicit aperture geometry and seeing.
- Heterogeneous apertures across lenses are allowed; incompatible apertures within the same lens are rejected.
- The old observation HDF5 is a reference or compatibility input, not the core direct-pipeline data model.
- Payload assembly, HDF5 writing, and validation are separated from catalog reading and numerical builders.

The earlier worktree reports that this was implemented and verified there with direct-pipeline tests, full `prepare_dataset/tests`, an environment check, and `git diff --check`. That evidence is useful, but it does not make the code safe to merge after the 2026-05-24 package/workspace restructure.

## Current Tree Facts

- The current public package is `src/statistical_sl`.
- Current CLI entry is `statistical-sl`, routed by `src/statistical_sl/cli.py`.
- Current data-preparation code lives under `src/statistical_sl/data_preparation`.
- Current canonical schema constants already live in `src/statistical_sl/core/canonical_schema.py`.
- Current cross-section source and boundary-policy contracts already live in `src/statistical_sl/core/cross_section_policy.py`.
- Current data-preparation physics kernels already live in `src/statistical_sl/data_preparation/physics`.
- Current data-preparation canonical writer still depends on prepared observation HDF5 input via `write_canonical_inference_dataset(...)`.
- Current tests explicitly guard against production dependencies on the old package names and old source directories.
- Current `workspace/configs/data_preparation` contains a lightweight data-preparation contract, not the 5.21 direct-pipeline config shape.

## Target Placement

Use a new internal workflow package rather than a top-level legacy-style package:

```text
src/statistical_sl/data_preparation/direct_pipeline/
  __init__.py
  records.py
  policies.py
  catalogs.py
  measurements.py
  sigma_resolver.py
  lens_preparer.py
  grid_builders.py
  cross_sections.py
  payload.py
  provenance.py
  validator.py
  writer.py
  config.py
  runner.py
```

Rationale:

- `direct_pipeline` describes the workflow and does not revive the old public package identity.
- Existing `data_preparation/physics` remains the owner of numerical formulas.
- Existing `core` remains the owner of canonical schema and cross-section policy strings.
- Existing `dataset_schema/writer.py` can remain as the compatibility writer from prepared observation HDF5. The new direct writer should write a `CanonicalDatasetPayload` and can later be unified with the compatibility writer after both paths share a validator.

New tests should live under:

```text
tests/data_preparation/
  test_direct_pipeline_records.py
  test_direct_pipeline_catalogs.py
  test_direct_pipeline_measurements.py
  test_direct_pipeline_sigma_resolver.py
  test_direct_pipeline_lens_preparer.py
  test_direct_pipeline_grid_builders.py
  test_direct_pipeline_cross_sections.py
  test_direct_pipeline_payload.py
  test_direct_pipeline_writer.py
  test_direct_pipeline_cli.py
  test_direct_pipeline_integration.py
  test_direct_pipeline_legacy_reference.py
```

New docs and configs should move to the 5.24 layout:

```text
docs/contracts/velocity_measurements_v1.md
docs/data-preparation/direct-canonical-pipeline.md
workspace/configs/data_preparation/cmass/devauc_direct_hunits.yaml
workspace/configs/data_preparation/cmass/sersic_direct_hunits.yaml
workspace/configs/data_preparation/sonnenfeld2024_slacs/paper_native_direct.yaml
```

## Execution Boundary Contract

This section is a gate for the main agent and any implementation subagent. It exists to prevent the direct pipeline from being ported only superficially.

Allowed:

- Use the 5.21 worktree as read-only reference material for behavior, tests, contract wording, and implementation ideas.
- Create a new isolated implementation worktree for the actual port, for example a branch such as `codex/direct-data-prep-statistical-sl`. The exact branch name may vary, but the worktree path, branch name, starting commit, and initial `git status --short --untracked-files=all` must be recorded before code migration starts.
- Reimplement behavior inside `src/statistical_sl/data_preparation/direct_pipeline/`.
- Reuse current shared constants from `src/statistical_sl/core/` when they are already stable cross-workflow contracts.
- Reuse current numerical kernels from `src/statistical_sl/data_preparation/physics/`.
- Keep the existing prepared-observation-HDF5 compatibility writer as its own mode while adding the direct writer as a separate source-to-canonical mode.
- Add direct-pipeline configs only under `workspace/configs/data_preparation/` and make the CLI consume those configs in real tests.
- Mention historical 5.21 paths in planning, migration notes, or history sections only when they are explicitly labeled as historical reference material.

Forbidden:

- No implementation code migration may start in the current main worktree unless the user explicitly overrides the isolated-worktree default. Keeping this planning file in the current worktree is allowed; landing production code, tests, configs, or docs for the port there is not the default.
- No production import, CLI, config loader, test helper, fixture builder, or documented current workflow may depend on `.worktrees/prepare-dataset-direct-canonical-pipeline`, `prepare_dataset`, `legacy/`, `Bayesian_inference`, or `Posterior_predictive_test`.
- No wrapper may preserve the old identity by forwarding `statistical_sl` calls into old `prepare_dataset` code.
- No `workspace/configs/data_preparation` file may be a decorative artifact; at least one CLI/integration test must prove that `statistical-sl prepare-dataset --build-canonical-direct --config <that config>` parses and consumes the new direct config shape.
- The old observation HDF5 may be read only by compatibility writer tests or legacy-reference audit tests. It may not become the source model for the direct pipeline.
- The compatibility writer must not be relabeled as the direct pipeline. A direct build must start from catalog plus trusted measurement source and assemble a `CanonicalDatasetPayload`.
- Real-data audit tests skipped because local files are absent do not prove numerical equivalence. They prove only that optional audit coverage is safely skippable.
- Documentation must not turn old paths, old module names, or old CLI commands into current workflow instructions.
- A subagent or implementation worker may not mark a task complete just because code was ported. The corresponding tests, docs/config expectations, and boundary checks must also be satisfied.
- Boundary tests that need to assert forbidden tokens must not write those forbidden tokens as contiguous string literals in test source. Follow the existing `tests/test_dependency_boundaries.py` pattern and assemble them from string fragments, otherwise static boundary searches will be polluted by the guard tests themselves.

Path and name ownership:

- Production direct workflow code: `src/statistical_sl/data_preparation/direct_pipeline/`.
- Production numerical kernels: existing `src/statistical_sl/data_preparation/physics/`.
- Cross-workflow schema and policy constants: `src/statistical_sl/core/`.
- Direct-pipeline tests: `tests/data_preparation/`.
- Public contract docs: `docs/contracts/` and `docs/data-preparation/`.
- User-facing direct configs: `workspace/configs/data_preparation/`.
- Public CLI: `statistical-sl prepare-dataset` only.
- Historical reference wording may mention old paths only in `docs/superpowers/plans/` or clearly marked migration/history notes.

## Implementation Tasks

### Task 0: Prepare the Implementation Worktree

**Files:** no production edits in the current main worktree.

- [ ] Create or select a fresh isolated worktree for the implementation branch unless the user explicitly approves using the current main worktree.
- [ ] Record the implementation worktree path, branch name, starting commit, and `git status --short --untracked-files=all`.
- [ ] Confirm that `.worktrees/prepare-dataset-direct-canonical-pipeline` is treated only as read-only reference input.
- [ ] Ensure the implementation worktree can read this plan file, either because it already exists there or because the plan is copied/recreated as planning context before edits begin.
- [ ] Run all subsequent code, test, config, and docs edits inside the implementation worktree.

Verification:

```bash
conda run -n cmass_lens git worktree list --porcelain
conda run -n cmass_lens git status --short --untracked-files=all
```

### Task 1: Snapshot and Diff the 5.21 Worktree

**Files:** no production edits.

- [ ] Record that the source implementation is the dirty worktree at `.worktrees/prepare-dataset-direct-canonical-pipeline`.
- [ ] Generate a file inventory of `prepare_dataset/prepare_dataset/canonical_pipeline`, `prepare_dataset/tests/test_direct_pipeline_*.py`, docs, and examples.
- [ ] Inspect changed tracked files in that worktree before copying logic, especially `prepare_dataset/prepare_dataset/cli.py` and `prepare_dataset/README.md`.
- [ ] Record which old files are copied as behavior references and which ones are intentionally not ported because they encode old package identity, old CLI identity, or root-level paths.
- [ ] Do not run destructive cleanup in that worktree; it contains uncommitted source material.

Verification:

```bash
conda run -n cmass_lens git -C .worktrees/prepare-dataset-direct-canonical-pipeline status --short --untracked-files=all
conda run -n cmass_lens git -C .worktrees/prepare-dataset-direct-canonical-pipeline diff --stat
```

### Task 2: Port Tests First With Current Imports

**Files:**

- Create tests under `tests/data_preparation/`.

- [ ] Copy the 5.21 direct-pipeline tests as behavioral specifications.
- [ ] Rewrite imports from the old package path to `statistical_sl.data_preparation.direct_pipeline`.
- [ ] Rewrite canonical schema imports to `statistical_sl.core.canonical_schema`.
- [ ] Rewrite CLI tests to target `statistical-sl prepare-dataset --build-canonical-direct --config ...`.
- [ ] Keep real-data comparison tests skipped unless local files are present.
- [ ] Run the new tests and confirm they fail because implementation modules are absent.
- [ ] Do not mark this task complete if copied tests are weakened, deleted, converted to broad smoke tests, or xfailed to avoid porting a contract.

Verification:

```bash
conda run -n cmass_lens python -m pytest tests/data_preparation/test_direct_pipeline_records.py -q
conda run -n cmass_lens python -m pytest tests/test_dependency_boundaries.py -q
```

Expected before implementation: direct-pipeline tests fail on missing modules; boundary tests still pass.

### Task 3: Port Domain Records and Policy Objects

**Files:**

- Create `src/statistical_sl/data_preparation/direct_pipeline/__init__.py`
- Create `src/statistical_sl/data_preparation/direct_pipeline/records.py`
- Create `src/statistical_sl/data_preparation/direct_pipeline/policies.py`

- [ ] Port `BaseLensRecord`, `SigmaObservation`, `PreparedLensRecord`, and `CanonicalDatasetPayload`.
- [ ] Port policy objects for units, profiles, mass definitions, aperture references, and sigma resolution.
- [ ] Use `statistical_sl.data_preparation.models.AperturePolicy` for aperture geometry.
- [ ] Use unit names from `statistical_sl.core.unit_conventions` or existing data-preparation config, not duplicated string literals when a current constant already exists.
- [ ] Keep validation docstrings explicit because these records define the direct-pipeline boundary.

Verification:

```bash
conda run -n cmass_lens python -m pytest tests/data_preparation/test_direct_pipeline_records.py -q
```

### Task 4: Port Catalog and Measurement Sources

**Files:**

- Create `src/statistical_sl/data_preparation/direct_pipeline/catalogs.py`
- Create `src/statistical_sl/data_preparation/direct_pipeline/measurements.py`
- Create `docs/contracts/velocity_measurements_v1.md`

- [ ] Port CMASS summary-table and SLACS table readers.
- [ ] Preserve the rule that CMASS catalog `sigma` and `sigma_err` are provenance only unless explicitly trusted.
- [ ] Port the `velocity_measurements_v1` CSV reader.
- [ ] Port the pPXF adapter as a compatibility adapter only.
- [ ] Preserve rejected-row audit output.
- [ ] Preserve accepted-row aperture and seeing requirements.
- [ ] Update contract docs so new upstream work emits `velocity_measurements_v1` directly.

Verification:

```bash
conda run -n cmass_lens python -m pytest \
  tests/data_preparation/test_direct_pipeline_catalogs.py \
  tests/data_preparation/test_direct_pipeline_measurements.py \
  -q
```

### Task 5: Port Sigma Resolution and Lens Preparation

**Files:**

- Create `src/statistical_sl/data_preparation/direct_pipeline/sigma_resolver.py`
- Create `src/statistical_sl/data_preparation/direct_pipeline/lens_preparer.py`

- [ ] Preserve deterministic sigma ordering: untagged single row, otherwise `A`, `B`.
- [ ] Preserve failure on duplicate tags and more than two accepted measurements.
- [ ] Preserve `missing_policy: num_sigma_zero` for external CMASS-like measurements.
- [ ] Compute or carry `Sigma_crit` exactly once per prepared record.
- [ ] Resolve physical scales through current unit-convention helpers.
- [ ] Preserve aperture override rules: measurement rows can supply per-lens aperture, dataset-level default is only a fallback when allowed by config.

Verification:

```bash
conda run -n cmass_lens python -m pytest \
  tests/data_preparation/test_direct_pipeline_sigma_resolver.py \
  tests/data_preparation/test_direct_pipeline_lens_preparer.py \
  -q
```

### Task 6: Port Derived Builders and Cross-Section Providers

**Files:**

- Create `src/statistical_sl/data_preparation/direct_pipeline/grid_builders.py`
- Create `src/statistical_sl/data_preparation/direct_pipeline/cross_sections.py`

- [ ] Reuse `statistical_sl.data_preparation.physics.m5.compute_mass_grid`.
- [ ] Reuse `statistical_sl.data_preparation.physics.m5.compute_dmass_dthetaein_grid`.
- [ ] Reuse `statistical_sl.data_preparation.physics.jeans.compute_sigma_unit_grid`.
- [ ] Preserve the rule that `s2_grid` is built only for `num_sigma > 0`.
- [ ] Preserve CMASS separable cross-section conversion from `cs_over_theta_ein` to theta-gamma area grid.
- [ ] Preserve Sonnenfeld finite-fibre `mufibre3_cs_grid` without applying the CMASS area formula.
- [ ] Write cross-section source metadata using `statistical_sl.core.cross_section_policy` constants.

Verification:

```bash
conda run -n cmass_lens python -m pytest \
  tests/data_preparation/test_direct_pipeline_grid_builders.py \
  tests/data_preparation/test_direct_pipeline_cross_sections.py \
  -q
```

### Task 7: Port Payload, Validator, and Direct Writer

**Files:**

- Create `src/statistical_sl/data_preparation/direct_pipeline/payload.py`
- Create `src/statistical_sl/data_preparation/direct_pipeline/provenance.py`
- Create `src/statistical_sl/data_preparation/direct_pipeline/validator.py`
- Create `src/statistical_sl/data_preparation/direct_pipeline/writer.py`

- [ ] Build payload blocks using names from `statistical_sl.core.canonical_schema`.
- [ ] Preserve payload provenance: catalog path, measurement path, rejected rows, ignored catalog sigma fields, and `num_sigma` distribution.
- [ ] Preserve `aperture_contract = per_lens` semantics for direct external measurements.
- [ ] Preserve validation failures for non-finite arrays, missing capability blocks, sigma without `s2_grid`, sigma without aperture metadata, and mismatched dimensions.
- [ ] Use atomic write behavior: validation failure leaves no partial output.

Verification:

```bash
conda run -n cmass_lens python -m pytest \
  tests/data_preparation/test_direct_pipeline_payload.py \
  tests/data_preparation/test_direct_pipeline_writer.py \
  -q
```

### Task 8: Add Config Parser, Runner, and CLI Hook

**Files:**

- Create `src/statistical_sl/data_preparation/direct_pipeline/config.py`
- Create `src/statistical_sl/data_preparation/direct_pipeline/runner.py`
- Modify `src/statistical_sl/data_preparation/cli.py`
- Modify `tests/test_smoke.py` only if the help-page expectations need stronger coverage.

- [ ] Add `--build-canonical-direct`.
- [ ] Add `--config <path>` for the direct pipeline without changing existing `--build-canonical-inference-dataset`.
- [ ] Include direct mode in the one-special-build-mode guard.
- [ ] Route direct mode to `direct_pipeline.runner.run_direct_canonical_build`.
- [ ] Keep existing HDF5-compatibility writer available as a separate mode.
- [ ] Keep the public path as `statistical-sl prepare-dataset`; do not add `python -m prepare_dataset` or old CLI shims.
- [ ] Add a test proving a config from `workspace/configs/data_preparation/...` is consumed by the runner path, not merely present on disk.
- [ ] Add a negative test that direct mode cannot be selected together with the prepared-HDF5 compatibility writer mode.

Verification:

```bash
conda run -n cmass_lens python -m pytest tests/data_preparation/test_direct_pipeline_cli.py tests/test_smoke.py -q
conda run -n cmass_lens python -m statistical_sl.cli prepare-dataset --help
```

### Task 9: Move Examples Into Workspace and Docs

**Files:**

- Create `docs/data-preparation/direct-canonical-pipeline.md`
- Create or update direct configs under `workspace/configs/data_preparation/...`
- Optionally create full pipeline recipes under `workspace/recipes/...` after post-canonical recipes remain green.
- Modify `tests/test_workspace_layout.py` if it should validate the new full data-preparation recipe contract.

- [ ] Translate 5.21 example YAMLs from `prepare_dataset/examples` to `workspace/configs/data_preparation`.
- [ ] Use `workspace/data/raw`, `workspace/data/external`, and `workspace/data/canonical` paths.
- [ ] Keep post-canonical recipes intact.
- [ ] Add full recipes only after config parsing and dry-run semantics are clear.
- [ ] Document the distinction between catalog source, trusted measurement source, and old HDF5 reference.
- [ ] Ensure docs present `velocity_measurements_v1` as the current upstream contract and label pPXF CSV as an adapter input.
- [ ] Ensure docs do not present `.worktrees/...`, `prepare_dataset/examples`, or `python -m prepare_dataset` as current workflow commands.

Verification:

```bash
conda run -n cmass_lens python -m pytest tests/test_workspace_layout.py tests/test_pipeline_recipe.py -q
```

### Task 10: Restore Integration and Numeric Audit Coverage

**Files:**

- Create `tests/data_preparation/test_direct_pipeline_integration.py`
- Create `tests/data_preparation/test_direct_pipeline_legacy_reference.py`
- Create small fixtures under `tests/fixtures/data_preparation/direct_pipeline/` if they are not inlined.

- [ ] Port fixture-level CMASS/slit build with `[1, 2, 0]` sigma counts.
- [ ] Port fixture-level SLACS build with trusted catalog sigma.
- [ ] Port real-data legacy-reference comparison as skipped when local files are absent.
- [ ] Keep the audit distinction clear: passing unit tests proves the workflow contract; it does not prove `s2_grid` numerical identity against every historical HDF5.
- [ ] If `s2_grid` still differs from the old reference, record it as a real numeric audit issue rather than hiding it behind green tests.
- [ ] Report skipped real-data audit tests explicitly in the final evidence. A skip is acceptable only as "not verified locally", never as "numeric parity confirmed".

Verification:

```bash
conda run -n cmass_lens python -m pytest tests/data_preparation/test_direct_pipeline_integration.py -q
conda run -n cmass_lens python -m pytest tests/data_preparation/test_direct_pipeline_legacy_reference.py -q -rs
```

### Task 11: Boundary and Regression Gate

**Files:** no planned source edits unless tests expose violations.

- [ ] Run all new direct-pipeline tests.
- [ ] Run existing root tests.
- [ ] Run the dependency-boundary guard.
- [ ] Check that no new tracked config points at root-level `data/` or `outputs/`.
- [ ] Check that no production file imports old package names.
- [ ] Check that the direct pipeline does not depend on `legacy/`.
- [ ] Check that docs/current workflow instructions do not tell users to run old module paths or old CLIs.
- [ ] Check that each newly created `workspace/configs/data_preparation` direct config is covered by parsing or CLI-consumption tests.
- [ ] If adding or modifying boundary tests, build forbidden-token assertions from string fragments so the tests do not create false positives in the static search.

Verification:

```bash
conda run -n cmass_lens python -m pytest tests/data_preparation -q
conda run -n cmass_lens python -m pytest tests -q
conda run -n cmass_lens python -m statistical_sl.cli prepare-dataset --help
conda run -n cmass_lens sh -c 'rg -n "prepare_dataset|Bayesian_inference|Posterior_predictive_test|legacy/" src tests workspace pyproject.toml; status=$?; test "$status" -eq 1'
conda run -n cmass_lens sh -c 'rg -n "python -m prepare_dataset|prepare_dataset/examples|\\.worktrees/prepare-dataset-direct-canonical-pipeline" README.md docs/contracts docs/data-preparation workspace; status=$?; test "$status" -eq 1'
conda run -n cmass_lens git diff --check
conda run -n cmass_lens git status --short --untracked-files=all
```

Expected:

- The static searches should print no matches. In raw `rg` terms, "no matches" exits with code 1 and is the expected pass condition; the `sh -c ... status=$?; test "$status" -eq 1` wrapper converts only that case into a passing command. Exit 0 from `rg` means forbidden text was found and must fail the gate. Exit 2 or another error code also fails the gate.
- Current workflow documentation is limited here to `README.md`, `docs/contracts`, `docs/data-preparation`, and `workspace`. Historical planning files under `docs/superpowers/plans` are intentionally excluded because they may legitimately mention old paths as history.
- If historical plans mention old paths, that is allowed. If current runbook, contract, data-preparation docs, workspace docs, configs, scripts, or recipes recommend old paths as the active workflow, that is a failure.

## Completion Definition And Evidence Gate

This plan is complete only when the code, tests, config, CLI, docs, and boundary checks all agree on the same direct-pipeline contract. The main agent must not accept a completion report that proves only one of these layers.

Per-task completion conditions:

- Task 0 is complete only when the implementation target worktree is recorded and the current main worktree is not being used for implementation without explicit user approval.
- Task 1 is complete only when the old worktree inventory is recorded and the implementation worker has explicitly separated reference-only files from portable contracts. Merely seeing the old worktree diff is not enough.
- Task 2 is complete only when the current-tree tests exist under `tests/data_preparation/`, use `statistical_sl` imports, fail for missing implementation before the implementation is added, and keep the behavioral assertions from 5.21 instead of replacing them with smoke tests.
- Tasks 3-7 are complete only when the corresponding direct-pipeline modules exist under `src/statistical_sl/data_preparation/direct_pipeline/`, their focused tests pass, and no implementation imports old package names or old worktree paths.
- Task 8 is complete only when the public CLI path `statistical-sl prepare-dataset --build-canonical-direct --config ...` reaches the direct runner and consumes a `workspace/configs/data_preparation` config. A private helper that works only by direct Python import does not satisfy this task.
- Task 9 is complete only when public docs and example configs describe the current `statistical_sl` workflow, not the old `prepare_dataset` workflow, and at least one test validates workspace config layout or parsing.
- Task 10 is complete only when fixture-level integration tests pass and the legacy-reference audit is run with skip reasons visible, for example `-rs` or an equivalent pytest reporting option. Any real-data audit skip must be reported as unverified numeric parity. If a real-data audit runs and finds `s2_grid` mismatch, that mismatch must be documented as an open numeric-audit risk.
- Task 11 is complete only when the full direct-pipeline suite, full root test suite, dependency-boundary search, docs/current-workflow search, `git diff --check`, and `git status --short --untracked-files=all` evidence have all been collected.

Final completion requires all of the following:

- Direct source-to-canonical production code lives in `src/statistical_sl/data_preparation/direct_pipeline/`.
- Implementation changes were made in the recorded isolated worktree, unless the final report cites the user's explicit approval to use the current main worktree.
- Tests live in `tests/data_preparation/` and include unit, CLI, fixture integration, and legacy-reference audit coverage.
- `docs/contracts/velocity_measurements_v1.md` exists and describes the current upstream measurement contract, including per-lens aperture and seeing requirements for accepted external rows.
- `docs/data-preparation/direct-canonical-pipeline.md` exists and clearly distinguishes catalog facts, trusted velocity measurements, pPXF adapter input, old HDF5 reference, direct payload, direct writer, and compatibility writer.
- Direct configs live under `workspace/configs/data_preparation/` and are consumed by tested CLI or runner paths.
- `statistical-sl prepare-dataset --build-canonical-direct --config <workspace config>` is the documented public entrypoint.
- Existing prepared-HDF5 compatibility behavior still has a separate mode and has not been misrepresented as the new direct pipeline.
- No production path depends on old package names, old source roots, `.worktrees`, or `legacy/`.
- No root-level `data/` or `outputs/` default is introduced.
- Legacy-reference audit evidence includes visible skip reasons and the final report lists skipped audit items, why they were skipped, and what risk remains unverified.
- All evidence commands in this plan have been run through `conda run -n cmass_lens ...`, or any missing command is explicitly listed with reason and risk.

Not complete, even if some tests pass:

- Only code was ported, but direct-pipeline tests, docs, configs, or CLI consumption were not ported.
- Implementation changes were landed in the current main worktree without explicit user approval.
- Only a subset of `tests/data_preparation` was run, but the final report claims the direct pipeline is complete.
- `workspace/configs/data_preparation` contains YAML files that no tested CLI/runner path reads.
- The implementation uses a wrapper, import alias, `sys.path` edit, or compatibility re-export to reach old `prepare_dataset` logic.
- The old observation HDF5 remains the core data model for direct builds.
- A skipped real-data audit is described as numeric validation.
- Historical docs mention old commands without labeling them as historical, or current docs recommend old commands.
- The compatibility writer is modified until it looks like a direct-pipeline entrypoint while still taking prepared HDF5 as the real input.
- The subagent plan gate is ignored and the main agent declares completion based only on an implementation summary.

Minimum final evidence bundle:

```bash
conda run -n cmass_lens python -m pytest tests/data_preparation -q -rs
conda run -n cmass_lens python -m pytest tests/data_preparation/test_direct_pipeline_legacy_reference.py -q -rs
conda run -n cmass_lens python -m pytest tests -q
conda run -n cmass_lens python -m statistical_sl.cli prepare-dataset --help
conda run -n cmass_lens python -m statistical_sl.cli prepare-dataset --build-canonical-direct --config workspace/configs/data_preparation/cmass/devauc_direct_hunits.yaml --dry-run
conda run -n cmass_lens sh -c 'rg -n "prepare_dataset|Bayesian_inference|Posterior_predictive_test|legacy/|\\.worktrees/prepare-dataset-direct-canonical-pipeline" src tests workspace pyproject.toml; status=$?; test "$status" -eq 1'
conda run -n cmass_lens sh -c 'rg -n "python -m prepare_dataset|prepare_dataset/examples|\\.worktrees/prepare-dataset-direct-canonical-pipeline" README.md docs/contracts docs/data-preparation workspace; status=$?; test "$status" -eq 1'
conda run -n cmass_lens git diff --check
conda run -n cmass_lens git status --short --untracked-files=all
```

The two negative `rg` gates are expected to produce no output. The wrapper deliberately treats raw `rg` exit code 1 as success and raw exit code 0 as failure.

If the final direct CLI does not support `--dry-run`, replace that command with the smallest fixture-backed CLI invocation that writes only to a temporary test directory and records the exact output path. That replacement must still prove that the public CLI reaches the direct runner and consumes the direct config shape; a Python import, parser-only unit test, or direct function call is not an acceptable substitute. Workspace config consumption and fixture-backed temporary output are separate evidence classes: at least one `workspace/configs/data_preparation/...` config must be covered by parser or runner tests, while fixture-backed CLI output may use small temporary paths to avoid real large-data generation. Do not run or interrupt long data-generation jobs to satisfy this evidence gate without explicit user permission.

## Risk Register

| Risk | Why it matters | Mitigation |
|---|---|---|
| Accidentally reviving old package identity | Breaks 5.24 public namespace and boundary tests | Port behavior into `statistical_sl.data_preparation.direct_pipeline`; never copy imports verbatim |
| Treating catalog sigma as trusted by default | Reintroduces the data-contract bug 5.21 was meant to remove | Keep catalog sigma in provenance unless config explicitly selects catalog-column mode |
| Mixing direct pipeline with compatibility HDF5 writer too early | Makes ownership unclear and hides old intermediate dependencies | Keep direct payload writer separate first; unify only after both paths share tests |
| Workspace paths drifting back to root `data/` | Breaks 5.24 workspace contract | Require workspace config tests and explicit path assertions |
| Assuming green tests prove historical numeric identity | 5.21 audit already found real `s2_grid` differences | Keep numeric audit tests and report mismatches separately |
| Letting direct pipeline configs become pipeline recipes | Blurs single-step config and orchestration layers | Keep direct config under `workspace/configs/data_preparation`; add recipes only after runner semantics are explicit |

## Execution Order Summary

1. Preserve the 5.21 dirty worktree as read-only source material.
2. Port tests and contract docs into current names.
3. Port pure data contracts.
4. Port readers, measurement adapters, resolver, and preparer.
5. Port numerical builder wrappers and cross-section providers using current physics/core modules.
6. Port payload, validator, writer, and runner.
7. Wire `statistical-sl prepare-dataset`.
8. Move examples into `workspace/configs/data_preparation`.
9. Run boundary, unit, integration, and optional real-data audit checks.

## Execution Defaults Unless User Overrides

These defaults remove ambiguity for implementation agents. They are not permission to shrink the plan; they are the minimum behavior unless the user explicitly changes scope.

1. First direct-config coverage defaults to CMASS `devauc` and CMASS `sersic` under `workspace/configs/data_preparation/cmass/`. Fixture-level SLACS coverage remains in scope because it protects the trusted-catalog-sigma mode and the Sonnenfeld/fibre cross-section semantics.
2. A manifest sidecar may be deferred to a later artifact-contract task, but an audit JSON or equivalent audit sidecar is required in the first port. The audit evidence must include catalog path, trusted measurement path, ignored catalog sigma fields, rejected rows, and `num_sigma` distribution.
3. Full pipeline recipes that include `data_preparation` do not block the standalone direct CLI migration. This plan is complete when `statistical-sl prepare-dataset --build-canonical-direct --config ...` is verified with current configs, docs, tests, and boundary gates. Full recipes can be added only after standalone runner semantics are clear and post-canonical recipes remain green.
4. A persistent `s2_grid` mismatch must never be hidden behind green functional tests. Whether it blocks merge depends on the declared target of the implementation: if the target claims numeric parity with the old reference, the mismatch blocks completion; if the target is a source-to-canonical workflow port, the mismatch may be carried as an explicitly named open numeric-audit risk with evidence, scope, and follow-up owner. In both cases, skipped or failing real-data audit must be reported plainly.
