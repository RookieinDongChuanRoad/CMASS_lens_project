# Prepare Dataset Direct Canonical Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild `prepare_dataset` around a direct, source-to-canonical pipeline that can start from simple lens catalogs, attach trusted velocity-dispersion measurements, generate derived grids, and write a validated canonical inference dataset without using intermediate observation HDF5 files as the core data model.

**Architecture:** The new path separates catalog truth, measurement truth, preparation policy, derived numerical builders, canonical payload assembly, HDF5 writing, and validation. Existing physics kernels should be reused where they are already tested; the rewrite target is the orchestration/data-contract layer, not the mass-grid, Jeans, or cross-section mathematics.

**Tech Stack:** Python 3 in the `cmass_lens` conda environment, `numpy`, `h5py`, `astropy`, `PyYAML`, existing `prepare_dataset.physics` kernels, and pytest.

---

## Context And Design Constraints

This plan responds to two real data-preparation modes that must share one architecture:

1. **SLACS-like mode:** a table such as `data/raw/SLACS_table.cat` carries lens-level catalog fields and usable velocity-dispersion measurements. In this mode the sigma resolver may trust the catalog sigma columns.
2. **CMASS/slit-like mode:** a table such as `data/raw/summary_table_deV.txt` carries useful lens-level catalog fields, but its `sigma` and `sigma_err` columns are not trusted. Trusted velocity-dispersion measurements must come from an external measurement source such as `/Users/liurongfu/Desktop/spec_check/outputs/abba_vis_positive_2026-04-02_rest3850_5250_all10/ppxf_results_optimal.csv`. Lenses absent from the trusted measurement source must remain in the lens sample with `num_sigma = 0`.

The current HDF5 reference file `data/raw/observations_deV_with_mass_grids.hdf5` should be treated as a migration/reference artifact, not as the central source model for the new pipeline. A recent inspection showed 23 lens groups with `num_sigma` distribution `{0: 10, 1: 10, 2: 3}`, while `summary_table_deV.txt` has `sigma` and `sigma_err` values on every row. The new design must therefore never infer sigma availability from the presence of catalog sigma columns.

All commands in this project must run in `cmass_lens`, for example:

```bash
conda run -n cmass_lens python -m pytest ...
```

## External Velocity Measurement Contract

The new pipeline should define a stable upstream input format instead of directly coupling itself to the current pPXF CSV column names. The pPXF file remains useful as an adapter target, but upstream measurement production should eventually emit the standard format below.

### `velocity_measurements_v1` CSV

Required columns:

```text
schema_version
lens_id
obs_tag
sigma_kms
sigma_err_kms
sigma_error_kind
measurement_status
use_for_likelihood
source_system
source_file
```

Optional but strongly recommended columns:

```text
z_lens
z_source
extraction_method
spectral_window
sigma_stat_kms
sigma_sys_kms
sigma_total_kms
velocity_offset_kms
chi2
pp_chi2
n_goodpix
template_id
warning_flags
quality_notes
aperture_shape
aperture_width_arcsec
aperture_height_arcsec
aperture_radius_arcsec
seeing_fwhm_arcsec
```

Column semantics:

- `schema_version`: must be `velocity_measurements_v1`.
- `lens_id`: canonical join key matching catalog lens IDs such as `023817-054555`.
- `obs_tag`: empty for single-measurement lenses; required as stable labels such as `A` and `B` when a lens has multiple spectra.
- `sigma_kms`: observed velocity dispersion in km/s. Must be positive and finite.
- `sigma_err_kms`: uncertainty actually used by the likelihood, also in km/s. This allows upstream to decide whether the likelihood uses statistical-only or total uncertainty.
- `sigma_error_kind`: enum such as `statistical`, `total`, or `custom`; this records how `sigma_err_kms` was chosen.
- `measurement_status`: upstream fit status such as `success`, `warning`, or `failed`.
- `use_for_likelihood`: boolean-like field. Only true rows participate in `num_sigma`; false rows are carried in the audit report but not in the canonical likelihood arrays.
- `source_system`: original spectrum/system identifier, for example `HSCJ023817-054555A`.
- `source_file`: path or logical identifier for the upstream pPXF run.
- aperture columns: may be omitted only when the pipeline config supplies a dataset-level default aperture contract. The resolved `PreparedLensRecord` must always end with explicit aperture metadata.

### pPXF adapter mapping

The current pPXF file has columns such as:

```text
system
base_name
obs_tag
extraction_method
z_lens
z_source
primary_status
sigma_primary_kms
sigma_stat_kms
sigma_sys_window_kms
sigma_total_kms
warnings
```

The adapter should map it into `velocity_measurements_v1` as:

```text
lens_id              <- base_name
obs_tag              <- obs_tag
sigma_kms            <- sigma_primary_kms
sigma_err_kms        <- configurable: sigma_stat_kms by default, sigma_total_kms if requested
sigma_error_kind     <- statistical or total, from config
measurement_status   <- primary_status
use_for_likelihood   <- primary_status == SUCCESS and sigma/err are finite positive
source_system        <- system
source_file          <- absolute pPXF CSV path
warning_flags        <- warnings
```

The adapter must not silently accept failed pPXF rows. It should report rejected rows with reasons.

## Target Module Layout

Create a new package under:

```text
prepare_dataset/prepare_dataset/canonical_pipeline/
```

Planned modules:

```text
records.py              # BaseLensRecord, SigmaObservation, PreparedLensRecord, CanonicalDatasetPayload
policies.py             # UnitPolicy, ProfilePolicy, AperturePolicyRef, MassDefinitionPolicy, SigmaPolicy
provenance.py           # source paths, file hashes, ignored columns, resolver decisions
catalog_sources.py      # SLACS and CMASS summary-table catalog readers
sigma_sources.py        # standard measurement CSV reader and pPXF adapter
sigma_resolver.py       # joins catalog records with trusted sigma observations
lens_preparer.py        # unit normalization, Sigma_crit, profile/aperture resolution
grid_builders.py        # in-memory mass grid and per-lens s2 grid builders
cross_sections.py       # cross-section provider abstraction
payload_builder.py      # assembles CanonicalDatasetPayload
writer.py               # writes payload to canonical HDF5
validator.py            # validates records, payload, and written HDF5
config.py               # YAML config parser for direct canonical builds
cli.py                  # direct-pipeline CLI entrypoint, called from existing prepare_dataset CLI
```

Existing modules such as `prepare_dataset.physics.m5`, `prepare_dataset.physics.jeans`, and `prepare_dataset.physics.lensing_cross_section` should be reused. Do not duplicate the numerical formulas unless tests prove a new wrapper is needed.

## Configuration Shape

The new direct path should be config-driven. A CMASS/slit build config should look like:

```yaml
schema_version: prepare_dataset_direct_pipeline_v1
output:
  canonical_hdf5: data/external/inference_dataset_devauc_slit_m5_hunits_v1.hdf5
  audit_json: data/external/inference_dataset_devauc_slit_m5_hunits_v1.audit.json

catalog:
  type: cmass_summary_table
  path: data/raw/summary_table_deV.txt
  profile_name: devauc
  ignored_columns:
    sigma: untrusted_catalog_value
    sigma_err: untrusted_catalog_value

velocity_measurements:
  type: ppxf_results_adapter
  path: /Users/liurongfu/Desktop/spec_check/outputs/abba_vis_positive_2026-04-02_rest3850_5250_all10/ppxf_results_optimal.csv
  error_column: sigma_stat_kms
  missing_policy: num_sigma_zero
  max_observations_per_lens: 2

units:
  unit_convention: h_units_v1
  h_ref: 0.7
  mass_definition_label: m5_hinvkpc
  mass_radius_kpc: 5.0

aperture:
  observation_flavor: slit
  sigma_definition: observed_aperture
  shape: rectangular
  width_arcsec: 1.6
  height_arcsec: 0.9
  seeing_fwhm_arcsec: 0.9

grids:
  gamma_axis:
    min: 1.2
    max: 2.8
    points: 17
  theta_e_axis:
    min: 0.0
    max: 5.0
    points: 256

cross_section:
  type: cmass_power_law
  source_hdf5: data/external/cs_grid_power.h5
```

A SLACS build config should differ mainly in `catalog`, `velocity_measurements`, `units`, `aperture`, and `cross_section`:

```yaml
catalog:
  type: slacs_table
  path: data/raw/SLACS_table.cat
  profile_name: devauc

velocity_measurements:
  type: catalog_columns
  value_column: veldisp
  error_column: veldisp_err
  missing_policy: fail

units:
  unit_convention: legacy_fixed_kpc
  h_ref: 0.7
  mass_definition_label: m5
  mass_radius_kpc: 5.0

aperture:
  observation_flavor: slacs_fibre
  sigma_definition: observed_aperture
  shape: circular
  radius_arcsec: 1.5
  seeing_fwhm_arcsec: 1.5

cross_section:
  type: sonnenfeld_fibre
  source_hdf5: data/external/fibre_crosssect_grid.hdf5
```

## Implementation Tasks

### Task 1: Add domain records and policy objects

**Files:**
- Create: `prepare_dataset/prepare_dataset/canonical_pipeline/__init__.py`
- Create: `prepare_dataset/prepare_dataset/canonical_pipeline/records.py`
- Create: `prepare_dataset/prepare_dataset/canonical_pipeline/policies.py`
- Test: `prepare_dataset/tests/test_direct_pipeline_records.py`

**Steps:**

1. Write tests for:
   - valid `BaseLensRecord`
   - valid `SigmaObservation`
   - rejection of non-positive `sigma_kms` and `sigma_err_kms`
   - `PreparedLensRecord.num_sigma` matching the observation list length
   - rejection of more than two sigma observations
2. Implement dataclasses with explicit docstrings and validation in `__post_init__`.
3. Run:

```bash
conda run -n cmass_lens python -m pytest prepare_dataset/tests/test_direct_pipeline_records.py -q
```

Expected: all new tests pass.

### Task 2: Implement catalog readers without trusting catalog sigma by default

**Files:**
- Create: `prepare_dataset/prepare_dataset/canonical_pipeline/catalog_sources.py`
- Test: `prepare_dataset/tests/test_direct_pipeline_catalog_sources.py`

**Steps:**

1. Write fixture tests for `summary_table_deV.txt` style input.
2. Assert that `CmassSummaryCatalogReader` reads lens identity, redshifts, Einstein radius, deV size, stellar mass, and profile fields.
3. Assert that CMASS summary-table `sigma` and `sigma_err` columns are recorded in provenance as ignored/untrusted, not stored as trusted `SigmaObservation`.
4. Write fixture tests for `SLACS_table.cat` style input.
5. Implement readers.
6. Run:

```bash
conda run -n cmass_lens python -m pytest prepare_dataset/tests/test_direct_pipeline_catalog_sources.py -q
```

Expected: catalog readers produce `BaseLensRecord` objects and provenance without sigma coupling.

### Task 3: Implement standard velocity measurement CSV reader and pPXF adapter

**Files:**
- Create: `prepare_dataset/prepare_dataset/canonical_pipeline/sigma_sources.py`
- Test: `prepare_dataset/tests/test_direct_pipeline_sigma_sources.py`

**Steps:**

1. Write tests for `velocity_measurements_v1` parsing.
2. Assert units are km/s by contract and values must be positive finite numbers.
3. Assert `use_for_likelihood=false` rows are returned as rejected/audit rows, not as `SigmaObservation`.
4. Write tests for pPXF adapter mapping from `base_name`, `obs_tag`, `sigma_primary_kms`, `sigma_stat_kms`, `sigma_total_kms`, and `primary_status`.
5. Assert `primary_status != SUCCESS` is rejected by default.
6. Assert the adapter can choose `sigma_stat_kms` or `sigma_total_kms` for `sigma_err_kms`.
7. Run:

```bash
conda run -n cmass_lens python -m pytest prepare_dataset/tests/test_direct_pipeline_sigma_sources.py -q
```

Expected: standard and pPXF formats both map into one measurement model.

### Task 4: Implement sigma resolver and missing-measurement policy

**Files:**
- Create: `prepare_dataset/prepare_dataset/canonical_pipeline/sigma_resolver.py`
- Test: `prepare_dataset/tests/test_direct_pipeline_sigma_resolver.py`

**Steps:**

1. Write tests for SLACS-like `catalog_columns` mode where every lens must receive one sigma observation.
2. Write tests for CMASS-like external mode:
   - lens with no external row becomes `num_sigma = 0`
   - lens with one untagged row becomes `num_sigma = 1`
   - lens with exactly `A` and `B` rows becomes `num_sigma = 2`
   - lens with duplicate tags fails
   - lens with more than two accepted measurements fails
3. Implement deterministic ordering: untagged single first, otherwise `A`, `B`.
4. Run:

```bash
conda run -n cmass_lens python -m pytest prepare_dataset/tests/test_direct_pipeline_sigma_resolver.py -q
```

Expected: missing sigma is a legal business state only when config says `missing_policy: num_sigma_zero`.

### Task 5: Implement lens preparation and unit normalization

**Files:**
- Create: `prepare_dataset/prepare_dataset/canonical_pipeline/lens_preparer.py`
- Test: `prepare_dataset/tests/test_direct_pipeline_lens_preparer.py`

**Steps:**

1. Write tests that fixed-kpc records preserve physical `log_mstar` and `log_re_kpc`.
2. Write tests that h-units records shift stellar mass and size consistently with the existing unit convention helpers.
3. Assert `Sigma_crit` is computed once and carried as prepared state.
4. Assert aperture metadata is explicit after preparation, either from measurement rows or dataset-level config.
5. Run:

```bash
conda run -n cmass_lens python -m pytest prepare_dataset/tests/test_direct_pipeline_lens_preparer.py -q
```

Expected: `PreparedLensRecord` is the only object passed to derived builders.

### Task 6: Add in-memory mass and per-lens velocity grid builders

**Files:**
- Create: `prepare_dataset/prepare_dataset/canonical_pipeline/grid_builders.py`
- Test: `prepare_dataset/tests/test_direct_pipeline_grid_builders.py`

**Steps:**

1. Write tests with small gamma axes and monkeypatched Jeans functions to avoid slow solves.
2. Assert mass grids are generated for all lenses.
3. Assert `s2_grid` is generated only for `num_sigma > 0`.
4. Assert `num_sigma = 0` lenses carry `has_s2 = false` and do not require a physical `s2_grid`.
5. Assert the builder reuses `prepare_dataset.physics.m5.compute_mass_grid`, `compute_dmass_dthetaein_grid`, and `prepare_dataset.physics.jeans.compute_sigma_unit_grid`.
6. Run:

```bash
conda run -n cmass_lens python -m pytest prepare_dataset/tests/test_direct_pipeline_grid_builders.py -q
```

Expected: no intermediate observation HDF5 is needed to construct mass and s2 payload blocks.

### Task 7: Add cross-section provider abstraction

**Files:**
- Create: `prepare_dataset/prepare_dataset/canonical_pipeline/cross_sections.py`
- Test: `prepare_dataset/tests/test_direct_pipeline_cross_sections.py`

**Steps:**

1. Write tests for CMASS power-law provider reading `compressed_grids/cs_over_theta_ein` and producing canonical `theta_E x gamma` area grids.
2. Write tests for Sonnenfeld fibre provider reading `tein_grid`, `gamma_grid`, and `mufibre3_cs_grid` without applying the CMASS area formula.
3. Assert provenance records the source HDF5 path and source mode.
4. Run:

```bash
conda run -n cmass_lens python -m pytest prepare_dataset/tests/test_direct_pipeline_cross_sections.py -q
```

Expected: cross-section semantics are explicit and mode-specific.

### Task 8: Build `CanonicalDatasetPayload`

**Files:**
- Create: `prepare_dataset/prepare_dataset/canonical_pipeline/payload_builder.py`
- Create: `prepare_dataset/prepare_dataset/canonical_pipeline/provenance.py`
- Test: `prepare_dataset/tests/test_direct_pipeline_payload_builder.py`

**Steps:**

1. Write tests assembling a tiny payload from two records, one with sigma and one without.
2. Assert payload blocks match canonical schema names:
   - `metadata`
   - `lenses`
   - `lensing_mass_grids`
   - `lensing_cross_section`
   - `velocity_dispersion_grids`
3. Assert capabilities match actual payload content.
4. Assert audit provenance includes:
   - catalog path
   - measurement path
   - ignored catalog sigma columns
   - rejected measurement rows
   - `num_sigma` distribution
5. Run:

```bash
conda run -n cmass_lens python -m pytest prepare_dataset/tests/test_direct_pipeline_payload_builder.py -q
```

Expected: payload creation is independent of HDF5 writer details.

### Task 9: Write canonical HDF5 from payload and validate it

**Files:**
- Create: `prepare_dataset/prepare_dataset/canonical_pipeline/writer.py`
- Create: `prepare_dataset/prepare_dataset/canonical_pipeline/validator.py`
- Test: `prepare_dataset/tests/test_direct_pipeline_writer.py`

**Steps:**

1. Write tests that a tiny payload writes a complete HDF5 file.
2. Assert atomic write behavior: failed validation leaves no partial output.
3. Assert HDF5 read-back validation catches:
   - missing capability block
   - non-finite arrays
   - `num_sigma > 0` with missing `s2_grid`
   - mismatched lens and grid dimensions
4. Run:

```bash
conda run -n cmass_lens python -m pytest prepare_dataset/tests/test_direct_pipeline_writer.py -q
```

Expected: canonical HDF5 writing is a pure serialization boundary over `CanonicalDatasetPayload`.

### Task 10: Add YAML config parser and direct-pipeline CLI

**Files:**
- Create: `prepare_dataset/prepare_dataset/canonical_pipeline/config.py`
- Create: `prepare_dataset/prepare_dataset/canonical_pipeline/cli.py`
- Modify: `prepare_dataset/prepare_dataset/cli.py`
- Test: `prepare_dataset/tests/test_direct_pipeline_cli.py`

**Steps:**

1. Write tests for YAML parsing.
2. Write tests for invalid configs:
   - CMASS catalog with `velocity_measurements.type: catalog_columns` should fail unless explicitly allowed.
   - external sigma source missing with `missing_policy: fail` should fail.
   - missing aperture metadata and no default aperture should fail.
3. Add an existing CLI flag:

```text
--build-canonical-direct
--config <path>
```

4. Ensure the existing CLI remains backward compatible.
5. Run:

```bash
conda run -n cmass_lens python -m pytest prepare_dataset/tests/test_direct_pipeline_cli.py -q
conda run -n cmass_lens python -m prepare_dataset --help
```

Expected: new direct build mode is available without breaking existing modes.

### Task 11: Add integration tests for CMASS/slit and SLACS modes

**Files:**
- Create: `prepare_dataset/tests/test_direct_pipeline_integration.py`
- Create: `prepare_dataset/tests/fixtures/direct_pipeline/`

**Steps:**

1. Build compact fixture files:
   - CMASS summary table with three lenses.
   - pPXF-like CSV with one untagged measurement, one A/B pair, and one missing lens.
   - SLACS-like table with two lenses and catalog sigma.
2. Run the full direct pipeline on fixtures with monkeypatched slow builders where needed.
3. Assert CMASS fixture produces `num_sigma = [1, 2, 0]`.
4. Assert SLACS fixture produces `num_sigma = [1, 1]`.
5. Assert both output HDF5 files pass validator.
6. Run:

```bash
conda run -n cmass_lens python -m pytest prepare_dataset/tests/test_direct_pipeline_integration.py -q
```

Expected: both real-world modes share the same payload/writer path.

### Task 12: Add migration comparison against existing HDF5 reference

**Files:**
- Create: `prepare_dataset/tests/test_direct_pipeline_legacy_reference.py`
- Optional: `prepare_dataset/prepare_dataset/canonical_pipeline/legacy_reference.py`

**Steps:**

1. Add a test that reads a small subset of `observations_deV_with_mass_grids.hdf5` when present.
2. Compare catalog-derived lens fields against `summary_table_deV.txt` for matching lens IDs.
3. Verify the new resolver does not adopt `summary_table_deV.txt` sigma values.
4. Verify the new resolver can reproduce the reference `num_sigma` distribution when pointed at a matching external measurement source, but does not require the old HDF5 file as input.
5. Mark tests that need local real data as skipped when the files are absent.
6. Run:

```bash
conda run -n cmass_lens python -m pytest prepare_dataset/tests/test_direct_pipeline_legacy_reference.py -q
```

Expected: old HDF5 is a reference, not a pipeline dependency.

### Task 13: Documentation and examples

**Files:**
- Create: `prepare_dataset/docs/direct_canonical_pipeline.md`
- Create: `prepare_dataset/examples/direct_canonical_cmass_slit.yaml`
- Create: `prepare_dataset/examples/direct_canonical_slacs.yaml`
- Modify: `prepare_dataset/README.md`

**Steps:**

1. Document the difference between catalog source, measurement source, and legacy HDF5 reference.
2. Document `velocity_measurements_v1` as the upstream contract.
3. Explain that pPXF CSV is adapted into that contract, not treated as the permanent interface.
4. Add example commands:

```bash
conda run -n cmass_lens python -m prepare_dataset --build-canonical-direct --config prepare_dataset/examples/direct_canonical_cmass_slit.yaml
conda run -n cmass_lens python -m prepare_dataset --build-canonical-direct --config prepare_dataset/examples/direct_canonical_slacs.yaml
```

5. Run:

```bash
conda run -n cmass_lens python -m pytest prepare_dataset/tests/test_docs_and_env.py -q
```

Expected: docs mention the standard environment and direct pipeline entrypoint.

### Task 14: Final verification

**Files:**
- No new files unless failures require fixes.

**Steps:**

1. Run targeted direct-pipeline tests:

```bash
conda run -n cmass_lens python -m pytest \
  prepare_dataset/tests/test_direct_pipeline_records.py \
  prepare_dataset/tests/test_direct_pipeline_catalog_sources.py \
  prepare_dataset/tests/test_direct_pipeline_sigma_sources.py \
  prepare_dataset/tests/test_direct_pipeline_sigma_resolver.py \
  prepare_dataset/tests/test_direct_pipeline_lens_preparer.py \
  prepare_dataset/tests/test_direct_pipeline_grid_builders.py \
  prepare_dataset/tests/test_direct_pipeline_cross_sections.py \
  prepare_dataset/tests/test_direct_pipeline_payload_builder.py \
  prepare_dataset/tests/test_direct_pipeline_writer.py \
  prepare_dataset/tests/test_direct_pipeline_cli.py \
  prepare_dataset/tests/test_direct_pipeline_integration.py \
  -q
```

2. Run the existing prepare_dataset suite:

```bash
conda run -n cmass_lens python -m pytest prepare_dataset/tests -q
```

3. Run environment check:

```bash
conda run -n cmass_lens python -m prepare_dataset.env_check
```

4. Run whitespace check:

```bash
git diff --check
```

Expected: all tests pass, environment check passes, and `git diff --check` reports no whitespace errors.

## Open Questions Before Implementation

1. Should `sigma_err_kms` default to `sigma_stat_kms` or `sigma_total_kms` for the current pPXF adapter? The current legacy updater used statistical uncertainty, but this should be an explicit config choice.
2. Should rejected pPXF rows be written only to the audit JSON, or also copied into a non-likelihood HDF5 provenance block?
3. Should the first production direct build target be CMASS/slit h-units only, or should SLACS fixed-kpc be implemented in the same PR for symmetry?
4. Should the new direct pipeline eventually replace existing `--build-canonical-inference-dataset`, or should both remain supported until all old run scripts are migrated?

## Non-Goals

- Do not rewrite `spherical_jeans` or the tested physics kernels.
- Do not make the canonical writer infer scientific model semantics from filenames.
- Do not require intermediate observation HDF5 files for the direct path.
- Do not treat catalog sigma columns as trusted unless a config explicitly selects `velocity_measurements.type: catalog_columns`.
- Do not stop or kill any long-running data-generation job without explicit user permission.

