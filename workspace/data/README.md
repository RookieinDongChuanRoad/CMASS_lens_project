# Workspace Data

Place local data products under this directory.  The expected subdirectories
are:

- `raw/` for source catalogs and untouched external tables.
- `external/` for externally produced auxiliary products such as sigma tables.
- `canonical/` for canonical inference datasets consumed by inference configs.
- `derived/` for intermediate products that can be regenerated.
- `caches/` for disposable local caches.
- `_staging/` for temporary migration or validation inputs that are not yet a
  stable public dataset contract.

Large scientific data files are ignored by git.  Keep provenance in manifests
or reports rather than committing binary data products directly.

The small source catalogs required by the example workspace recipe are tracked
under `raw/`; generated HDF5, NPZ, and cache products remain local.
