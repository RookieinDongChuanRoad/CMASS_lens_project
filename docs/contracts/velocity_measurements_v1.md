# velocity_measurements_v1 CSV Contract

`velocity_measurements_v1` is the handoff contract between an upstream
velocity-dispersion measurement workflow and
`statistical_sl.data_preparation.direct_pipeline`.

Source catalogs such as `summary_table_deV.txt` and `SLACS_table.cat` remain
catalog sources. They are not trusted velocity-dispersion likelihood sources
unless a direct-pipeline config explicitly selects `catalog_columns` with
`trust_catalog_sigma: true`.

## Required Purpose

Each row describes one attempted velocity-dispersion measurement for one lens.
Rows with `use_for_likelihood=true` become accepted sigma observations and
contribute to `num_sigma`. Rows with `use_for_likelihood=false` are preserved
in the audit trail and do not contribute to `num_sigma`.

Accepted velocity-dispersion values are always in `km/s`:

- `sigma_kms`: measured velocity dispersion.
- `sigma_err_kms`: likelihood uncertainty.

## Standard Columns

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
aperture_shape
aperture_width_arcsec
aperture_height_arcsec
aperture_radius_arcsec
seeing_fwhm_arcsec
```

Additional audit columns are allowed. Common examples include `z_lens`,
`z_source`, `extraction_method`, `spectral_window`, `sigma_stat_kms`,
`sigma_sys_kms`, `sigma_total_kms`, `chi2`, `template_id`, and
`quality_notes`.

## Accepted-Row Rules

- `schema_version` must be exactly `velocity_measurements_v1`.
- `lens_id` must match the source catalog join key.
- `sigma_kms` and `sigma_err_kms` must be positive finite numbers.
- `aperture_shape` must be `rectangular` or `circular`.
- `seeing_fwhm_arcsec` must be a positive finite number.
- Rectangular rows require positive `aperture_width_arcsec` and
  `aperture_height_arcsec`, with blank `aperture_radius_arcsec`.
- Circular rows require positive `aperture_radius_arcsec`, with blank
  `aperture_width_arcsec` and `aperture_height_arcsec`.
- A lens with two accepted rows must use one shared aperture geometry and
  seeing value across those rows.

Heterogeneous apertures across different lenses are valid. Heterogeneous
apertures within one lens are rejected because one lens-level `s2_grid` row
cannot represent two incompatible aperture contracts.

## Rejected-Row Rules

Rows with `use_for_likelihood=false` may leave aperture and seeing fields blank.
Their raw row payload is still preserved so the audit JSON can explain why the
measurement was excluded.

## Minimal Example

```csv
schema_version,lens_id,obs_tag,sigma_kms,sigma_err_kms,sigma_error_kind,measurement_status,use_for_likelihood,source_system,source_file,aperture_shape,aperture_width_arcsec,aperture_height_arcsec,aperture_radius_arcsec,seeing_fwhm_arcsec
velocity_measurements_v1,023817-054555,A,262.4,18.7,statistical,success,true,abba_vis,velocity_measurements_v1.csv,rectangular,1.6,0.9,,0.7
velocity_measurements_v1,090315+411609,A,301.2,22.1,statistical,success,true,sdss_fibre,velocity_measurements_v1.csv,circular,,,1.5,1.2
velocity_measurements_v1,091205+002901,,0,0,statistical,failed,false,abba_vis,velocity_measurements_v1.csv,,,,,
```

## Compatibility Adapter

`ppxf_results_adapter` remains available for older local CSV exports. It is a
compatibility adapter, not the preferred upstream handoff. New upstream work
should emit `velocity_measurements_v1.csv` directly.

Successful adapter rows must still provide the aperture and seeing columns
listed above. The direct pipeline does not silently fill accepted external
measurements from a dataset-level aperture default.
