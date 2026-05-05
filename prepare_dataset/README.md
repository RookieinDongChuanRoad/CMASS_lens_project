# Prepare Dataset

This project prepares the numerical data products used by the CMASS lensing
workflow. It owns interpolation-grid generation, sigma-table generation, and
canonical inference dataset construction. The only supported runtime and
verification environment is the conda environment `cmass_lens`.

## Standard Environment

Use `cmass_lens` for every command in this project. The repository includes
[environment.yml](/Users/liurongfu/Work/CMASS_lens_project/prepare_dataset/environment.yml)
to describe the baseline package set expected in that environment.

Update an existing environment with:

```bash
conda env update -n cmass_lens -f environment.yml
```

The project also depends on `spherical_jeans`, but that package is not vendored
into this repository. The reason is deliberate: this repository is responsible
for grid orchestration and compatibility tests, while `spherical_jeans`
continues to be maintained as a separate scientific dependency. The standard
environment must therefore be able to import `spherical_jeans`.

## Environment Check

Run this before processing data or claiming the environment is ready:

```bash
conda run -n cmass_lens python -m prepare_dataset.env_check
```

The check validates:
- the active interpreter belongs to `cmass_lens`
- runtime dependencies such as `numpy`, `scipy`, `astropy`, and `h5py` import
- the external dependency `spherical_jeans` imports successfully
- `pytest` is present so the verification workflow can run in the same standard environment

## Running the Tool

Process one file:

```bash
conda run -n cmass_lens python -m prepare_dataset --input /Users/liurongfu/Work/CMASS_lens_project/data/raw/observations_with_mass_grids_all.hdf5
```

Process one galaxy group for debugging:

```bash
conda run -n cmass_lens python -m prepare_dataset --input /Users/liurongfu/Work/CMASS_lens_project/data/raw/observations_with_mass_grids_all.hdf5 --group 023817-054555
```

Process both standard input files:

```bash
conda run -n cmass_lens python -m prepare_dataset --all-default-inputs
```

Use `--overwrite-in-place` only when you explicitly want the input file to be
replaced after successful processing. The default behavior writes a new file so
the original input remains untouched.

## Build Lensing Cross-section HDF5 Files

`prepare_dataset` now owns the two cross-section generators used by the project.
They should be treated as different scientific products, even though both are
tabulated on power-law lens parameters.

### CMASS power-law table

This mode migrates the historical
`/Users/liurongfu/Desktop/Spectrum_reduction/make_lenscrosect_grid.py` logic
into the package.  It writes the old `cs_grid_power.h5` schema:

- `/full_grids/gamma_grids`
- `/full_grids/theta_ein_grids`
- `/full_grids/cs_grid`
- `/compressed_grids/gamma_grids`
- `/compressed_grids/cs_over_theta_ein_grid`

Important convention: `full_grids/cs_grid` is the source-plane radius
`beta_max`, not an area.  Downstream code converts the compressed ratio into an
area with:

```text
cross_section(theta_E, gamma) = pi * (cs_over_theta_ein(gamma) * theta_E)**2
```

Standard build command:

```bash
conda run -n cmass_lens python -m prepare_dataset \
  --build-power-law-cross-section-hdf5 \
  --output /Users/liurongfu/Work/CMASS_lens_project/data/external/cs_grid_power.h5 \
  --overwrite-in-place
```

The production defaults are `gamma = linspace(1.2, 2.8, 81)` and
`theta_E = linspace(0.1, 5.0, 51)`.

### Sonnenfeld finite-fibre table

This mode migrates the official Sonnenfeld reference script
`astrosonnen/strong_lensing_tools:papers/slacs_selection/scripts/make_crosssect_grid.py`.
It writes `fibre_crosssect_grid.hdf5` with:

- `tein_grid`
- `gamma_grid`
- `mufibre2_cs_grid`
- `mufibre3_cs_grid`
- `ycaust_grid`

Important convention: `mufibre2_cs_grid` and `mufibre3_cs_grid` are already
source-plane area cross-sections.  They include the finite SDSS fibre aperture,
Gaussian seeing convolution, and flux-threshold cuts.  They must not be
converted again with the CMASS separable formula.

Standard build command:

```bash
conda run -n cmass_lens python -m prepare_dataset \
  --build-fibre-cross-section-hdf5 \
  --output /Users/liurongfu/Work/CMASS_lens_project/data/external/fibre_crosssect_grid.hdf5 \
  --overwrite-in-place
```

The production defaults are `gamma = linspace(1.2, 2.8, 81)`,
`theta_E = linspace(0.0, 5.0, 51)`, `nbeta = 1001`, `nr = 16`,
`fibre_arcsec = 1.5`, `seeing_arcsec = 1.5`, and `muB_min = 1.0`.  The full
fibre grid is intentionally slow because it follows the reference numerical
integration path instead of replacing the fibre/PSF calculation with an
analytic approximation.

## Build Canonical Inference Dataset

The canonical dataset writer combines a prepared observation HDF5 file, a
legacy CMASS cross-section HDF5 file, and optional sigma-bundle input into one
`inference_dataset.hdf5` product.  The writer is part of data preparation only;
Bayesian inference does not read this product in the current migration step.

Example:

```bash
conda run -n cmass_lens python -m prepare_dataset \
  --build-canonical-inference-dataset \
  --observation-hdf5 /Users/liurongfu/Work/CMASS_lens_project/data/raw/observations_with_mass_grids_all.hdf5 \
  --cross-section-hdf5 /Users/liurongfu/Work/CMASS_lens_project/data/external/cs_grid_power.h5 \
  --sigma-bundle-hdf5 /Users/liurongfu/Work/CMASS_lens_project/data/external/jeans_sers_sigma_bundle.h5 \
  --profile sersic \
  --mass-definition-label m5_hinvkpc \
  --output /Users/liurongfu/Work/CMASS_lens_project/data/external/inference_dataset_sersic.hdf5 \
  --theta-e-min 0.0 \
  --theta-e-max 5.0 \
  --theta-e-points 256
```

The canonical writer emits these top-level HDF5 blocks:

- `metadata`
- `lenses`
- `lensing_mass_grids`
- `lensing_cross_section`
- `velocity_dispersion_grids`

The old one-dimensional `cs_over_theta_ein(gamma)` input is converted during
data preparation into the canonical two-dimensional `theta_E x gamma`
`lensing_cross_section.cross_section_grid`.  Lenses with `num_sigma = 0` may
carry placeholder `s2_grid` values in the canonical product; inference logic
must rely on `num_sigma` and `has_s2`, not the placeholder value.

When the cross-section input is a Sonnenfeld `fibre_crosssect_grid.hdf5`, the
canonical writer preserves `mufibre3_cs_grid` directly as the two-dimensional
area grid and uses the file's own `tein_grid` axis.  The optional
`--theta-e-*` arguments only affect legacy CMASS separable inputs.

## Build PPT Sigma HDF5 Tables

The posterior predictive test now treats two bundle files as canonical:

- `/Users/liurongfu/Work/CMASS_lens_project/data/external/jeans_deV_sigma_bundle.h5`
- `/Users/liurongfu/Work/CMASS_lens_project/data/external/jeans_sers_sigma_bundle.h5`

These tables are not derived from the raw observation HDF5 files above. They
are full Jeans interpolation tables for arbitrary replicated lenses and must be
generated by direct evaluation of the current production kernel. Do not rebuild
them by resampling any previous external table.

The target bundle schema can contain six canonical leaves:

- `/slit/m5`
- `/slit/m10`
- `/boss/m5`
- `/boss/m10`
- `/within_re/m5`
- `/within_re/m10`

The bundle root groups now mix two different concepts on purpose:

- `slit` and `boss` are observation-flavor leaves using fixed angular apertures
- `within_re` is a separate sigma definition, not a third observation flavor

The supported production definitions are:

- `slit`: rectangular `1.6 x 0.9 arcsec`, `seeing = 0.9 arcsec`
- `boss`: circular radius `1.0 arcsec`, `seeing = 1.5 arcsec`
- `within_re`: circular aperture with radius equal to each galaxy's `R_e`, with no seeing term

The CLI exposes a dedicated build mode for these tables:

```bash
conda run -n cmass_lens python -m prepare_dataset --build-sigma-unit-hdf5 --profile all --observation-flavor all --sigma-definition all --workers 14
```

The legacy-repack mode is still available when you only need to reorganize the
historical slit tables without recomputing them:

```bash
conda run -n cmass_lens python -m prepare_dataset --repack-legacy-sigma-unit-hdf5 --profile all
```

Important options:

- `--build-sigma-unit-hdf5`: switch from raw-file updates to PPT table generation
- `--profile devauc|sersic|all`: build one profile or both
- `--observation-flavor slit|boss|all`: build one observed-aperture flavor or refresh both observed-aperture leaves
- `--sigma-definition observed_aperture|within_re|all`: choose whether to build fixed-aperture leaves, within-Re leaves, or both
- `--workers <N>`: worker-process count; defaults to `os.cpu_count()`
- `--output-dir <DIR>`: optional override for the output directory; defaults to `/Users/liurongfu/Work/CMASS_lens_project/data/external`
- `--repack-legacy-sigma-unit-hdf5`: reorganize the existing legacy slit files into the bundle schema without recomputing
- `--legacy-sigma-input-dir <DIR>`: optional override for the location of the old flat slit files during repack

Constraint:

- `--sigma-definition within_re` must not be combined with `--observation-flavor slit` or `--observation-flavor boss`, because `within_re` is not an observation flavor

Standard rebuild commands:

Rebuild only the deVaucouleurs BOSS leaves:

```bash
conda run -n cmass_lens python -m prepare_dataset --build-sigma-unit-hdf5 --profile devauc --observation-flavor boss --workers 14
```

Rebuild only the Sersic slit leaves:

```bash
conda run -n cmass_lens python -m prepare_dataset --build-sigma-unit-hdf5 --profile sersic --observation-flavor slit --workers 14
```

Rebuild both bundle files while saturating the local workstation:

```bash
conda run -n cmass_lens python -m prepare_dataset --build-sigma-unit-hdf5 --profile all --observation-flavor all --sigma-definition all --workers 14
```

Rebuild only the within-Re leaves:

```bash
conda run -n cmass_lens python -m prepare_dataset --build-sigma-unit-hdf5 --profile all --sigma-definition within_re --workers 14
```

The Sersic build is much slower than the deV table. That is expected: the
Sersic table spans one extra interpolation axis and therefore requires
substantially more direct Jeans solves.

The canonical production bundle files are expected to contain populated leaves
for both observed-aperture flavors plus the within-Re definition:

- `/slit/m5` and `/slit/m10`
- `/boss/m5` and `/boss/m10`
- `/within_re/m5` and `/within_re/m10`

### HDF5 Contract

Each output bundle uses an explicit schema consumed by the PPC code:

- root dataset `profile_name`
- root attr `schema_version`
- root attr `quantity_name`
- root groups `slit`, `boss`, and `within_re`

Observed-aperture leaves under `/<observation flavor>/<mass label>` contain:

- dataset `gamma_axis`
- dataset `zd_axis`
- dataset `log_re_kpc_axis`
- dataset `n_axis` for `sersic` only
- dataset `s_unit_grid`
- attr `mass_definition_label`
- attr `mass_radius_kpc`
- attr `units`
- attr `observation_flavor`
- attr `aperture_shape`
- attr `aperture_radius_arcsec` or `aperture_width_arcsec` plus `aperture_height_arcsec`
- attr `seeing_fwhm_arcsec`

Within-Re leaves under `/within_re/<mass label>` contain:

- dataset `gamma_axis`
- dataset `log_re_kpc_axis`
- dataset `n_axis` for `sersic` only
- dataset `s_unit_grid`
- attr `mass_definition_label`
- attr `mass_radius_kpc`
- attr `units`
- attr `sigma_definition="within_re"`
- attr `aperture_shape="circular"`
- attr `aperture_radius_mode="effective_radius"`
- attr `seeing_mode="none"`

The tabulated quantity is:

```text
S_unit = sigma^2 / 10**m_R
```

The current supported enclosed-mass radii are `R = 5 kpc` and `R = 10 kpc`.
The `units` attribute therefore becomes either `km2 s-2 per 10**m5` or
`km2 s-2 per 10**m10`.

The axis order is fixed:

- `devauc`: `(gamma, zd, log_re_kpc)`
- `sersic`: `(gamma, zd, log_re_kpc, n)`

For the within-Re definition, the axis order is lower-dimensional because redshift
is no longer part of the aperture contract:

- `devauc`: `(gamma, log_re_kpc)`
- `sersic`: `(gamma, log_re_kpc, n)`

The legacy one-file-per-leaf HDF5 tables remain readable for backward
compatibility, but bundle files are now the canonical assets for both slit and
BOSS workflows.

### Verification

Verify the upstream grid-generation code:

```bash
conda run -n cmass_lens pytest -q tests/test_jeans_regression.py tests/test_hdf5_processing.py tests/test_sigma_unit_tables.py
```

Verify the downstream PPC loader and consumer:

```bash
cd /Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test
conda run -n cmass_lens pytest -q tests/test_posterior_predictive.py
```

Smoke-check that the real external files can be loaded by PPC:

```bash
cd /Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference
conda run -n cmass_lens env PYTHONPATH="/Users/liurongfu/Work/CMASS_lens_project/Bayesian_inference/src:/Users/liurongfu/Work/CMASS_lens_project/Posterior_predictive_test/src" \
  python -c "from cmass_lens_inference.mass_definition import get_mass_definition; from cmass_posterior_predictive.predictive import SigmaUnitTable; jobs=[('/Users/liurongfu/Work/CMASS_lens_project/data/external/jeans_deV_sigma_bundle.h5','boss','devauc'),('/Users/liurongfu/Work/CMASS_lens_project/data/external/jeans_sers_sigma_bundle.h5','slit','sersic')]; \
for path, flavor, profile in jobs: \
    table = SigmaUnitTable.from_path(path, mass_definition=get_mass_definition(5), observation_flavor=flavor); \
    print(path, flavor, profile, table.mass_definition_label, table.bundle_leaf_path, table.values.shape, float(table.values.min()), float(table.values.max()))"
```

## Testing

All verification commands are defined in the same standard environment:

```bash
conda run -n cmass_lens pytest -q
```

This project treats testability in `cmass_lens` as part of the environment
contract, not as an optional developer convenience.
