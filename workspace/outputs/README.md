# Workspace Outputs

Generated runs belong here.  The canonical layout is one directory per run:

```text
outputs/
  <analysis_family>/
    <run_id>/
      run_manifest.json
      config_snapshots/
      data_preparation/
      inference/
      posterior_predictive/
        diagnostics/
```

The repository tracks this README only.  Run directories, HDF5 chains, NPZ
bundles, figures, logs, and temporary sampler state remain local by default.
