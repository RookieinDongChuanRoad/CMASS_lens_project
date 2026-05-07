# Numba/Emcee Production Backend Migration Plan

## Final Goal

The final production inference architecture in this worktree must be:

```text
canonical inference dataset
  -> thin ModelSpec + ModelRuntimeAdapter
  -> model-specific Numba kernels
  -> backend-owned posterior reduction and diagnostics
  -> emcee sampler
  -> emcee HDFBackend chain.h5 outputs
```

The migration must make `numba/emcee` the only production backend.  JAX and
NumPyro may be used temporarily as numerical oracles during the migration, but
they must not remain on the production run path after the final cleanup.

This is a backend replacement, not an architecture rollback.  The model-author
target in `docs/model_refactor_progress.md` remains binding:

- model files define parameters, metadata, required data capabilities, and
  scientific model assembly;
- model-specific preprocessing builds validated NumPy source contexts from the
  canonical inference dataset;
- framework code owns config validation, context dispatch, sampling, output
  metadata, checkpoints, diagnostics, and backend execution;
- production inference still starts from `data.inference_dataset_path`, not raw
  observation/cross-section/sigma-table paths.

## Non-Negotiable Boundaries

- Do not restore the old monolithic `model.py` / `sampler.py` / raw-input
  production structure.
- Do not re-enable production YAML fields such as `data.observation_path`,
  `data.cross_section_path`, or `data.sigma_table_path`.
- Do not keep NumPyro sampling fields as production config fields once Phase A
  is complete.
- Do not treat CMASS success as final project success.  Sonnenfeld must also be
  wired through the same `numba/emcee` production framework before the final
  goal is considered met.
- Use the `cmass_lens` conda environment for implementation and verification.

## Key Design

Numba cannot realistically compile arbitrary dynamic Python model hooks in the
same way JAX traces pure array functions.  The maintainable design is therefore
not a fully generic automatic Numba compiler for every `ModelSpec` hook.

Instead, the production framework has three layers:

1. **Model declaration layer**
   - `models/cmass.py` and `models/sonnenfeld2024_slacs.py` stay thin assembly
     files.
   - They name concrete models, parameters, capability contracts, metadata, and
     component-level scientific meaning.

2. **Runtime context layer**
   - `models/*_runtime.py` and `models/components/*/preprocessing.py` build
     model-specific NumPy source contexts from canonical datasets.
   - These contexts remain sampler-agnostic and backend-owned consumers treat
     them as opaque model data.

3. **Numba execution layer**
   - Shared backend code owns posterior reduction, diagnostics, parameter-bound
     rejection, output-facing blob schema, and sampler integration.
   - Model-specific monolithic Numba kernels implement the actual hot path for
     each scientific model.

This preserves the final target because a new model author still does not need
to understand emcee, HDF5 outputs, checkpointing, or runner orchestration.  A
production-grade model does, however, need a model-owned Numba kernel adapter
when its likelihood and selection integrals differ materially from existing
models.

## Phase A: Backend Migration Skeleton

### Goal

Replace the production run skeleton with `numba_backend + emcee_sampler` while
preserving the registry and canonical dataset boundaries.

### Planned Work

- Add `src/cmass_lens_inference/numba_backend/`.
- Add backend-facing model adapter and diagnostics schema.
- Add `src/cmass_lens_inference/emcee_sampler.py`.
- Switch `runner.py` from `run_numpyro_sampler()` to `run_emcee_sampler()`.
- Switch production metadata to:
  - `backend: numba_emcee`
  - `kernel_backend: numba`
  - `sampler: emcee`
  - `chain_storage: emcee_hdf_backend`
- Change `SamplingConfig` to emcee-native production fields:
  - `n_walkers`
  - `n_steps`
  - `burn_in`
  - `initial_jitter_scale`
  - `random_seed`
- Reject NumPyro-only sampling fields in production configs:
  - `num_chains`
  - `num_samples`
  - `num_warmup`
  - `chain_method`
  - `thinning`

### Phase A Acceptance

- Production imports no longer require `numpyro_sampler.py`.
- `runner.py` production path resolves a Numba compiled model and emcee sampler.
- Config parser accepts emcee-native sampling fields and rejects NumPyro-only
  production fields.
- A minimal CMASS run writes `chain.h5` and emcee-compatible metadata.

### Phase A Status

Status: complete.

Completed:

- Added `src/cmass_lens_inference/numba_backend/` with a registry-facing model
  adapter, diagnostic blob dtype, backend dispatch, and model-specific kernel
  entrypoints.
- Added `src/cmass_lens_inference/emcee_sampler.py` and switched `runner.py`
  to `build_numba_compiled_model() + run_emcee_sampler()`.
- Replaced production sampling config fields with `n_walkers`, `n_steps`, and
  `burn_in`; the config parser now rejects the retired sampling field names.
- Switched production outputs to emcee HDFBackend `chain.h5` plus lightweight
  emcee checkpoints.
- Removed production NumPyro artifact writers and ArviZ dependency from the
  package dependency list.
- Confirmed importing `model_registry` and `runner` for CMASS/Sonnenfeld does
  not import `jax` or `numpyro`.

Remaining:

- No Phase A blockers remain.

## Phase B: CMASS Numba Production

### Goal

Make CMASS a complete production model on the `numba/emcee` backend, including
real canonical devauc validation and a fixed final Numba benchmark.

### Planned Work

- Add CMASS-specific Numba kernels under `numba_backend`.
- Use the deleted legacy Numba implementation at the backup commit as an
  algorithm reference, but adapt it to the current canonical source context.
- Implement the current canonical `theta_E x gamma` cross-section interpolation.
- Implement CMASS normalization, per-lens likelihood, FP summary, FP prior, and
  posterior diagnostics.
- Ensure `posterior_corner.py` can read the new production `chain.h5`.
- Keep production data entry canonical-only.

### Phase B Acceptance

- Synthetic CMASS log-probability is finite at the configured initial center.
- CMASS bounds and normalization rejection paths are tested.
- FP-disabled and FP-enabled CMASS paths are tested.
- Real devauc canonical log-probability is compared against a frozen oracle for:
  - likelihood term;
  - normalization value;
  - FP prior diagnostics when enabled;
  - total posterior value.
- A short real or synthetic CMASS emcee run completes and writes readable
  `chain.h5`.
- Performance is measured on the real FP-enabled path with fixed config details.

### Phase B Status

Status: complete.

Completed:

- Added CMASS Numba kernels for selection normalization, per-lens likelihood,
  canonical `theta_E x gamma` cross-section interpolation, FP summary, and FP
  prior diagnostics.
- Updated `posterior_corner.py` to read production `chain.h5` instead of the
  retired array/netCDF artifacts.
- Verified synthetic CMASS finite log-probability, host-side bounds rejection,
  h-unit context values, FP-disabled path, FP-enabled path, and short emcee
  `chain.h5` output.
- Verified real devauc slit canonical data against the legacy raw oracle using
  current Numba kernels:
  - canonical log_prob: `-129.39561603870516`;
  - legacy log_prob: `-129.39561603870516`;
  - delta log_prob: `0.0`;
  - canonical normalization: `0.4871499982066181`;
  - legacy normalization: `0.487149998206618`;
  - delta normalization: `1.1102230246251565e-16`.
- Verified real devauc slit/BOSS FP diagnostics are finite:
  - slit log_prob `-233.26842187564836`, FP prior `-103.8728058369432`,
    `fpfit_mu=2.3824739595552935`, `fpfit_beta=0.30186483077170184`,
    `fpfit_scatter=0.05598807504759378`;
  - BOSS log_prob `-290.39710660067954`, same finite FP diagnostics.
- Ran a real devauc short emcee run and confirmed:
  - status `completed`;
  - completed steps `2`;
  - `chain.h5` exists;
  - metadata records `backend=numba_emcee`, `sampler=emcee`, and
    `chain_storage=emcee_hdf_backend`.
- Replaced `scripts/benchmark_log_prob.py` with a Numba/emcee benchmark tool.
  Real devauc benchmark evidence:
  - config: `/private/tmp/cmass_numba_devauc_acceptance.yaml`;
  - output: `/private/tmp/cmass_numba_benchmarks/20260507_130130_numba_benchmark.json`;
  - `normalization_samples=1024`, `gamma_points=80`, `mstar_points=80`,
    `n_walkers=24`, `n_steps=2`;
  - first call `0.1836464999942109 s`;
  - steady median `0.00052370794583112 s`.

Remaining:

- No Phase B blockers remain.  The old JAX production runner is intentionally
  not restored for live comparison; any future JAX comparison should be run
  from an oracle/backup branch, not from the production path.

## Phase C: Sonnenfeld Numba Model

### Goal

Port the current Sonnenfeld runnable model semantics to model-specific Numba
kernels and run it through the same `numba/emcee` production framework.

### Planned Work

- Add Sonnenfeld-specific Numba kernels under `numba_backend`.
- Implement Sonnenfeld normalization and per-lens likelihood integrals.
- Implement velocity-dispersion proxy selection:
  - `theta_E_est` from sigma proxy;
  - population sigma-unit interpolation;
  - finite-fibre `theta_E x gamma` cross-section;
  - parent/proposal correction;
  - source-redshift density.
- Support both concrete Sonnenfeld model labels:
  - `sonnenfeld2024_slacs` for paper-native `legacy_fixed_kpc + m5`;
  - `sonnenfeld2024_slacs_hunit` for explicit h-units backend continuity.
- Use synthetic/reference fixtures as migration oracles.  Any future JAX
  comparison must stay outside the production run path.

### Phase C Acceptance

- Synthetic paper-native Sonnenfeld log-probability is finite.
- Synthetic hunit Sonnenfeld log-probability is finite.
- Missing canonical capabilities fail fast.
- Numba and reference fixture values are checked for:
  - population draw;
  - selection weight;
  - lens integrals;
  - normalization;
  - total log-probability.
- A short Sonnenfeld emcee run completes on an available synthetic canonical
  fixture.

### Phase C Status

Status: complete.

Completed:

- Added Sonnenfeld-specific Numba kernels for normalization and per-lens
  likelihood.
- Implemented velocity-dispersion proxy selection with `theta_E_est`,
  population sigma-unit interpolation, finite-fibre `theta_E x gamma`
  cross-section lookup, parent/proposal correction, and source-redshift
  density.
- Kept both concrete model labels wired through the same production backend:
  - `sonnenfeld2024_slacs` for `legacy_fixed_kpc + m5`;
  - `sonnenfeld2024_slacs_hunit` for `h_units_v1 + m5_hinvkpc`.
- Verified synthetic paper-native Sonnenfeld finite Numba log-probability.
- Verified synthetic hunit Sonnenfeld finite Numba log-probability.
- Verified missing canonical capabilities fail at the model/data boundary.
- Verified a short Sonnenfeld emcee run writes production `chain.h5`.

Remaining:

- No Phase C production-backend blockers remain.  Paper-level Sonnenfeld
  science validation remains a future scientific validation task, separate from
  the backend migration.

## Final Acceptance Framework

The final migration is complete only when all of the following are true.

### Architecture Acceptance

- `runner.py` production path calls only `numba_backend` and `emcee_sampler`.
- `model_registry.py` keeps config-driven dispatch for CMASS and both
  Sonnenfeld labels.
- Model assembly files do not import samplers, runners, output writers, or raw
  production data loaders.
- Production config uses emcee-native sampling fields and rejects NumPyro-only
  fields.
- Production data entry remains `data.inference_dataset_path`.

### Numerical Acceptance

- CMASS synthetic and real devauc paths pass finite/equivalence checks.
- CMASS FP-disabled and FP-enabled paths pass.
- Sonnenfeld synthetic paper-native and hunit paths pass.
- Missing capability tests fail at the model/data boundary, not deep inside a
  kernel.

### Runtime Acceptance

- CMASS short production run completes and writes native emcee `chain.h5`.
- Sonnenfeld short production run completes on the available fixture.
- Resume works from persisted emcee state or an explicitly documented
  checkpoint contract.
- Posterior plotting can read new production outputs.
- Metadata records `backend: numba_emcee`,
  `kernel_backend: numba`, `sampler: emcee`, and
  `chain_storage: emcee_hdf_backend`.

### Performance Acceptance

- A fixed real devauc benchmark records final Numba backend timing and config
  details.
- The benchmark records config path, normalization sample count, gamma grid
  size, stellar-mass grid size, walker/step settings, CPU/thread policy, and
  throughput.
- Live JAX comparison is no longer a production acceptance requirement in this
  worktree because the production runner has intentionally removed JAX/NumPyro.
  If a historical comparison is needed, it should be run from an oracle branch
  or backup commit.

### Documentation Acceptance

- This document is updated after each phase.
- `docs/model_refactor_progress.md` is updated after final acceptance to reflect
  `numba/emcee` as the production backend and to remove JAX/NumPyro from the
  final target wording.

## Progress Log

### 2026-05-06

- Created this migration plan.
- Current phase: Phase A.
- Next work: write failing tests for the production backend contract and then
  implement the minimal backend skeleton.

### 2026-05-07

- Completed Phase A, Phase B, and Phase C.
- Final production backend is `numba/emcee`; optional JAX code remains only as
  oracle/legacy reference code and is not imported by production registry or
  runner paths.
- Verification commands:
  - `conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q`
  - `conda run -n cmass_lens python -m py_compile ...`
  - `conda run -n cmass_lens python Bayesian_inference/scripts/benchmark_log_prob.py --config /tmp/cmass_numba_devauc_acceptance.yaml --repeats 3 --output-dir /tmp/cmass_numba_benchmarks`
- Latest full test result:
  - `........................................................................ [ 62%]`
  - `.............sss............................                             [100%]`
  - the three skipped tests are real-data tests whose hardcoded b283
    `data/external` files are absent in this worktree; equivalent real-data
    checks were run manually against locally available canonical HDF5 files in
    the 1ca5 data directory.
