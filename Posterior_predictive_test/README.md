# Posterior Predictive Test

This package runs model-aware posterior-predictive diagnostics for completed
inference runs.

## Architecture

- Generic workflow code reads run directories, config snapshots, posterior
  samples, and writes JSON/NPZ/PNG artifacts.
- Model-specific predictive logic is registered through
  `lensing_posterior_predictive.registry.get_predictive_definition(model_name)`.
- CMASS diagnostics use `adapters/cmass.py` and require the model-declared
  external input `sigma_table`.
- Sonnenfeld/SLACS diagnostics use `adapters/sonnenfeld.py` and read the
  population sigma-unit grid from canonical input instead of requiring the
  CMASS observed-aperture sigma table.
- Pre-registry raw observation/cross-section snapshots are quarantined in
  `legacy.py` and are CMASS-only compatibility inputs.

## Diagnostics Parallelism

Posterior diagnostics use adapter-owned Numba kernels by default.  The generic
PPC workflow resolves one `DiagnosticsExecution` object, records it in the JSON
artifacts, and passes it to the model adapter.  It does not wrap model-specific
prediction logic in a generic Python process pool.

The current default execution strategy is `kernel_only`:

- `requested_worker_processes` records the user-facing `--worker-processes`
  request when one was provided.
- `worker_processes` is `0`, meaning no Python process pool is used.
- `kernel_threads_per_process` is the resolved Numba thread budget consumed by
  the adapter kernel, capped by the runtime CPU budget and the number of
  posterior draws.

This keeps the scientific prediction boundary inside adapters while making the
runtime metadata explicit.  A future process-pool strategy should be added only
after benchmark evidence shows that chunking compiled adapter kernels across
processes improves a specific model without CPU oversubscription.

## Adding A Model

Add a model-specific adapter that exposes a `PredictiveDefinition` with:

- `model_name`
- `backend`
- `supported_diagnostics`
- `required_external_inputs`
- `artifact_schema_version`
- `build_context`
- `run_diagnostics`
- `trend_category_names`
- `build_trend_panel_order`

`run_diagnostics` must accept a keyword-only `execution` argument.  New model
adapters should follow the CMASS and Sonnenfeld pattern: keep posterior
predictive semantics in the adapter, move the expensive parent-population loop
into a Numba-backed implementation, and apply
`execution.kernel_threads_per_process` before entering the compiled kernel.

The generic orchestration layer should not import concrete model posterior
helpers directly or branch on model names.
