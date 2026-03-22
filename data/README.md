# Data Layout

- `raw/`: immutable observation files.
- `external/`: externally generated grids.
- `derived/`: reproducible transformed tables.
- `caches/`: disposable performance caches.

Current files:

- `raw/observations_with_mass_grids_all.hdf5`
- `raw/observations_deV_with_mass_grids.hdf5`
- `raw/observations_with_BOSS_mass_grids_all.hdf5`
- `raw/observations_deV_with_BOSS_mass_grids.hdf5`
- `external/jeans_deV_sigma_bundle.h5`
- `external/jeans_sers_sigma_bundle.h5`
- `external/cs_grid_power.h5`

Canonical external sigma assets are now the two bundle files above. Each bundle
uses grouped HDF5 leaves for observation flavor and enclosed-mass definition.
During the current staged rollout, the migrated bundles are expected to contain
only the existing legacy slit leaves under `/slit/m5` and `/slit/m10`. The
`/boss` group is reserved for a later dedicated build and should not be assumed
to contain populated leaves yet.
