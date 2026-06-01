# Direct Canonical Data-Preparation Pipeline

The direct pipeline builds a canonical inference HDF5 from source catalogs and
trusted velocity-dispersion measurements. It lives under
`statistical_sl.data_preparation.direct_pipeline` and is exposed through:

```bash
conda run -n cmass_lens statistical-sl prepare-dataset \
  --build-canonical-direct \
  --config workspace/configs/data_preparation/cmass/devauc_direct_hunits.yaml
```

The direct path does not use prepared observation HDF5 files as its source
model. Those files may still be used in migration-reference tests, but they are
not runtime inputs for a direct build.

## Input Boundary

- Catalog source: redshifts, Einstein radius, effective radius, stellar mass,
  profile branch, and catalog provenance.
- Measurement source: trusted likelihood velocity dispersions in `km/s`.
- Aperture contract: accepted external measurement rows carry aperture geometry
  and seeing; the canonical output stores aperture fields per lens.
- Cross-section source: a CMASS power-law ratio product or a Sonnenfeld
  finite-fibre area grid.

For CMASS, catalog `sigma` and `sigma_err` columns are provenance only unless a
config explicitly chooses trusted catalog columns. A lens missing from the
external measurement source remains in the sample with `num_sigma = 0` when the
missing policy is `num_sigma_zero`.

For SLACS/Sonnenfeld compatibility, catalog sigma columns may be used only with:

```yaml
velocity_measurements:
  type: catalog_columns
  trust_catalog_sigma: true
```

## Module Flow

```text
catalogs.py        -> source lens records plus catalog provenance
measurements.py    -> trusted sigma rows plus rejected-row audit
sigma_resolver.py  -> per-lens num_sigma decisions
lens_preparer.py   -> sigma_crit, physical scales, aperture policy, unit policy
grid_builders.py   -> in-memory mass grids and per-lens s2 grids
cross_sections.py  -> theta_E x gamma cross-section blocks
payload.py         -> CanonicalDatasetPayload
writer.py          -> validated canonical HDF5
validator.py       -> payload and HDF5 read-back checks
config.py          -> schema-checked YAML config
runner.py          -> end-to-end orchestration
```

Physics formulas remain in the existing data-preparation physics layer. This
pipeline is responsible for source contracts, orchestration, provenance, and
validation.

## YAML Contract

The current direct-pipeline schema version is:

```yaml
schema_version: statistical_sl_direct_data_preparation_v1
```

All relative paths are resolved relative to the YAML file location. Repository
examples live under `workspace/configs/data_preparation/` and should point into
`workspace/data`, not root-level `data` or `outputs`.

Supported catalog types:

- `cmass_summary_table`
- `slacs_table`

Supported measurement modes:

- `velocity_measurements_v1`
- `ppxf_results_adapter`
- `catalog_columns`

Supported cross-section modes:

- `cmass_power_law`: converts the compressed CMASS ratio to a
  `theta_E x gamma` source-plane area grid.
- `sonnenfeld_fibre`: preserves `mufibre3_cs_grid` as an already-integrated
  finite-fibre area grid.

Axis specs may be explicit lists or `{min, max, points}` range mappings.

## Validation

The writer validates both the in-memory payload and the HDF5 read-back. The
checks include:

- required canonical top-level blocks exist
- declared capabilities are present
- lens rows align with mass-grid and `s2_grid` rows
- `num_sigma > 0` implies an available per-lens `s2_grid`
- root metadata declares `aperture_contract = per_lens`
- sigma-bearing lens rows have complete aperture and seeing metadata
- sigma-free rows may leave aperture and seeing metadata empty
- numeric datasets are finite
- cross-section grid shape matches its axes

Focused regression command:

```bash
conda run -n cmass_lens python -m pytest tests/data_preparation/test_direct_pipeline_*.py -q
```
