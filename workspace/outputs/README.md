# Workspace Outputs

Generated runs belong here.  The canonical layout is one directory per run:

```text
outputs/
  <profile>/
    <run_id>/
      run_manifest.json
      config_snapshot.yaml
      metadata.json
      run_result.json
      chain.h5
      posterior_corner.png
      posterior_corner_result.json
      checkpoints/
      logs/
      posterior_predictive/
        diagnostics/
          <diagnostic_run_id>/
            run_manifest.json
            ppc_summary.json
            replicated_statistics.npz
            fig8_like.png
            fig8_like_summary.json
            fig8_like_curves.npz
            gamma_vs_*.png
            gamma_vs_*_summary.json
            gamma_vs_*_curves.npz
```

Inference artifacts currently live at the run root because `posterior_corner`
and posterior diagnostics read `config_snapshot.yaml` and `chain.h5` from that
location. A future artifact-layout migration may introduce an `inference/`
subdirectory, but that requires compatibility handling for existing readers and
saved runs.

The repository tracks this README only.  Run directories, HDF5 chains, NPZ
bundles, figures, logs, and temporary sampler state remain local by default.
