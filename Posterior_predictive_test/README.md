# Posterior Predictive Test

This package runs model-aware posterior-predictive diagnostics for completed
inference runs.

## Architecture

- Generic workflow code reads run directories, config snapshots, posterior
  samples, and writes JSON/NPZ/PNG artifacts.
- Model-specific predictive logic is registered through
  `cmass_posterior_predictive.registry.get_predictive_definition(model_name)`.
- CMASS diagnostics use `adapters/cmass.py` and require the model-declared
  external input `sigma_table`.
- Sonnenfeld/SLACS diagnostics use `adapters/sonnenfeld.py` and read the
  population sigma-unit grid from canonical input instead of requiring the
  CMASS observed-aperture sigma table.
- Pre-registry raw observation/cross-section snapshots are quarantined in
  `legacy.py` and are CMASS-only compatibility inputs.

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

The generic orchestration layer should not import concrete model posterior
helpers directly.
