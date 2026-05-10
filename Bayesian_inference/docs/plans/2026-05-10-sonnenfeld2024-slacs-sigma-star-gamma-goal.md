# Sonnenfeld 2024 SLACS Sigma-Star Gamma Model Goal

> This is the read-only goal document for the planned model addition. After creation, the file must remain read-only unless the scientific target itself is deliberately renegotiated.

**Goal:** Add a new production Bayesian-inference model family named `sonnenfeld2024_slacs_sigma_star_gamma` that is parallel to `cmass` and `sonnenfeld2024_slacs`, while changing only the Sonnenfeld gamma-distribution relation to the CMASS-style sigma-star-dependent form.

**Architecture:** The new model lives in its own implementation package under `Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs_sigma_star_gamma/`. Like the existing Sonnenfeld implementation, one package exposes two concrete registry model names: a paper-native fixed-kpc model and an explicit h-unit model. Runtime preprocessing, canonical dataset requirements, finite-fibre selection, source-redshift treatment, FP prior semantics, and output contracts remain Sonnenfeld-native.

**Tech Stack:** Python, YAML, Numba/emcee production backend, canonical HDF5 inference datasets, pytest, existing `ModelSpec` / `ModelRuntimeAdapter` registry contract.

---

## Final Target

### New Implementation Package

Create a package parallel to the existing production model packages:

```text
Bayesian_inference/src/cmass_lens_inference/models/
  cmass/
  sonnenfeld2024_slacs/
  sonnenfeld2024_slacs_sigma_star_gamma/
    __init__.py
    assembly.py
    posterior.py
    runtime.py
```

The new package must not replace or mutate the scientific contract of `sonnenfeld2024_slacs`. The original Sonnenfeld package remains the paper-faithful mass-and-size-residual gamma model.

### Registry Model Names

Expose two concrete model names through `Bayesian_inference/src/cmass_lens_inference/model_registry.py`:

- `sonnenfeld2024_slacs_sigma_star_gamma`
  - `unit_convention: legacy_fixed_kpc`
  - `mass_definition_label: m5`
  - `mass_coordinate: physical_fixed_5kpc`
- `sonnenfeld2024_slacs_sigma_star_gamma_hunit`
  - `unit_convention: h_units_v1`
  - `mass_definition_label: m5_hinvkpc`
  - `mass_coordinate: h_units_v1_m5_hinvkpc`

The unit-convention boundary must match the current `sonnenfeld2024_slacs` pattern: one implementation package supports both unit semantics, but the registry exposes them as separate concrete model names and YAML configs must choose one explicitly.

### Configuration Files

Create two configs:

```text
Bayesian_inference/configs/sonnenfeld2024_slacs_sigma_star_gamma.yaml
Bayesian_inference/configs/sonnenfeld2024_slacs_sigma_star_gamma_hunit.yaml
```

The paper-native config must point to:

```text
/Users/liurongfu/Work/CMASS_lens_project/data/external/inference_dataset_sonnenfeld2024_slacs_m5_fixed_v1.hdf5
```

The h-unit config must point to:

```text
/Users/liurongfu/Work/CMASS_lens_project/data/external/inference_dataset_sonnenfeld2024_slacs_m5_hunits_v1.hdf5
```

Both configs should explicitly set:

```yaml
fp_prior:
  enabled: true
```

unless a later experiment intentionally defines a no-FP comparison config with a separate name.

## Scientific Contract

### What Stays Unchanged

The new model keeps the current Sonnenfeld model's scientific structure except for the gamma distribution:

- Same SLACS/Sonnenfeld parent population.
- Same `legacy_fixed_kpc` and `h_units_v1` unit-convention handling as `sonnenfeld2024_slacs`.
- Same fixed 5 kpc / `m5` paper-native mass definition.
- Same `m5_hinvkpc` h-unit mass definition.
- Same finite-fibre `theta_E x gamma` cross-section lookup.
- Same velocity-dispersion proxy selection.
- Same source-redshift population and per-lens source-redshift likelihood.
- Same aperture mass relation.
- Same quadratic size relation.
- Same canonical capability requirements, including `velocity_dispersion.population_sigma_unit.v1`.
- Same FP-prior machinery and h-unit mass-location shifts for FP cut and pivot.

### Only Intended Scientific Change

Current `sonnenfeld2024_slacs` gamma mean:

```text
mu_gamma = mu_gamma_0 + beta_gamma * (mstar - mstar_pivot) + xi_gamma * delta_r
```

New `sonnenfeld2024_slacs_sigma_star_gamma` gamma mean:

```text
sigma_star_shift9p0 = mstar - log10(2 pi) - 2 * log_re - 9
mu_gamma = mu_gamma_0 + beta_sigma_star_gamma * sigma_star_shift9p0
```

This mirrors the CMASS sigma-star-dependent gamma mode while keeping Sonnenfeld's parent population, selection, and likelihood structure.

### Unit-Convention Detail

The sigma-star combination is invariant under the existing h-unit coordinate transformation:

```text
mstar_hunit = mstar_physical + 2 log10(h_ref)
log_re_hunit = log_re_physical + log10(h_ref)
mstar_hunit - 2 log_re_hunit = mstar_physical - 2 log_re_physical
```

Therefore the same formula can be evaluated in the active coordinate for both registry models, as long as the h-unit model uses the same active-coordinate preprocessing already used by `sonnenfeld2024_slacs_hunit`.

## Parameter Schema

Both new registry models use the same public parameter names:

```text
mu5_0
beta5
xi5
sigma5
mu_gamma_0
beta_sigma_star_gamma
sigma_gamma
mu_zs
sigma_zs
theta0
loga
```

The dimension is 11D.

The original Sonnenfeld models remain 12D:

```text
mu5_0
beta5
xi5
sigma5
mu_gamma_0
beta_gamma
xi_gamma
sigma_gamma
mu_zs
sigma_zs
theta0
loga
```

The new model must not accept `beta_gamma` or `xi_gamma`. The original model must not start accepting `beta_sigma_star_gamma`.

## Implementation Path

### Task 1: Add Failing Registry and Schema Tests

Modify:

```text
Bayesian_inference/tests/test_model_registry_config.py
Bayesian_inference/tests/test_component_specs.py
```

Add tests that fail before implementation:

- `get_model_definition("sonnenfeld2024_slacs_sigma_star_gamma")` resolves.
- `get_model_definition("sonnenfeld2024_slacs_sigma_star_gamma_hunit")` resolves.
- The paper-native model requires `legacy_fixed_kpc` and resolves to mass label `m5`.
- The h-unit model requires `h_units_v1` and resolves to mass label `m5_hinvkpc`.
- The parameter schema is 11D and includes `beta_sigma_star_gamma`.
- The schema excludes `beta_gamma` and `xi_gamma`.
- The original `sonnenfeld2024_slacs` and `sonnenfeld2024_slacs_hunit` schemas remain 12D.

Run:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_model_registry_config.py Bayesian_inference/tests/test_component_specs.py -q
```

Expected before implementation: failure because the new registry model names do not exist.

### Task 2: Add the New Model Assembly Package

Create:

```text
Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs_sigma_star_gamma/__init__.py
Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs_sigma_star_gamma/assembly.py
```

`assembly.py` should:

- Define `MODEL_NAME = "sonnenfeld2024_slacs_sigma_star_gamma"`.
- Define `HUNIT_MODEL_NAME = "sonnenfeld2024_slacs_sigma_star_gamma_hunit"`.
- Reuse the same Sonnenfeld required capabilities.
- Reuse the same mass, source-redshift, discovery, lensing, selection, and FP-related metadata semantics.
- Use `sigma_star_linear_gamma_component` for the gamma relation.
- Define `get_model_spec()` for paper-native fixed-kpc.
- Define `get_hunit_model_spec()` for h-unit.

Recommended component key:

```text
table1_velocity_proxy_sigma_star_gamma
```

Recommended metadata:

```python
"gamma_distribution": "sigma_star_dependent"
"foreground_population": "sonnenfeld2024_table1"
"selection": "velocity_dispersion_proxy_theta_e_est"
"cross_section": "theta_gamma_finite_fibre"
```

### Task 3: Add the New Posterior

Create:

```text
Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs_sigma_star_gamma/posterior.py
```

Start from the existing Sonnenfeld posterior structure, but keep the new file model-owned and explicit. The hot path changes are:

- `_unpack_theta(...)` returns the 11D sigma-star-gamma parameter tuple.
- `_draw_population_state(...)` computes:

```python
sigma_star_shift9p0 = mstar - LOG10_2PI - 2.0 * log_re - 9.0
mu_gamma = mu_gamma_0 + beta_sigma_star_gamma * sigma_star_shift9p0
```

- `normalization_mc_numba(...)` checks `theta.shape[0] == 11`.
- `population_summary_mc_numba(...)` checks `theta.shape[0] == 11`.
- `log_likelihood_lenses_numba(...)` checks `theta.shape[0] == 11`.
- Per-lens likelihood must receive `log_re_obs` and compute:

```python
sigma_star_shift9p0 = (
    mstar_grid[lens_index, mstar_index]
    - LOG10_2PI
    - 2.0 * log_re_obs[lens_index]
    - 9.0
)
mu_gamma = mu_gamma_0 + beta_sigma_star_gamma * sigma_star_shift9p0
```

The implementation must not change the original `sonnenfeld2024_slacs/posterior.py` unless a tiny shared helper extraction is genuinely necessary and covered by tests.

### Task 4: Add Runtime Wrapper

Create:

```text
Bayesian_inference/src/cmass_lens_inference/models/sonnenfeld2024_slacs_sigma_star_gamma/runtime.py
```

This file may reuse the existing Sonnenfeld runtime adapter and preprocessing because the data contract and parameter-independent context are unchanged:

- same canonical dataset loader
- same context bundle builder
- same data spec

The wrapper exists so the new package has the same package shape as `cmass` and `sonnenfeld2024_slacs`.

### Task 5: Register and Export the Model

Modify:

```text
Bayesian_inference/src/cmass_lens_inference/model_registry.py
Bayesian_inference/src/cmass_lens_inference/models/__init__.py
```

`model_registry.py` must bind:

- `sonnenfeld2024_slacs_sigma_star_gamma` to `get_model_spec()`
- `sonnenfeld2024_slacs_sigma_star_gamma_hunit` to `get_hunit_model_spec()`
- both to the new package's runtime adapter
- both to the new package's posterior `log_prob`

The unsupported-model error message must include the two new names.

### Task 6: Add Configs and Config Tests

Create:

```text
Bayesian_inference/configs/sonnenfeld2024_slacs_sigma_star_gamma.yaml
Bayesian_inference/configs/sonnenfeld2024_slacs_sigma_star_gamma_hunit.yaml
```

Modify:

```text
Bayesian_inference/tests/test_config_profiles_io.py
```

The tests should load both checked-in YAML files and assert:

- paper-native config uses `model.name: sonnenfeld2024_slacs_sigma_star_gamma`
- paper-native config uses `unit_convention: legacy_fixed_kpc`
- paper-native config resolves mass label `m5`
- h-unit config uses `model.name: sonnenfeld2024_slacs_sigma_star_gamma_hunit`
- h-unit config uses `unit_convention: h_units_v1`
- h-unit config resolves mass label `m5_hinvkpc`
- both configs expose the 11D parameter schema
- both configs explicitly set `fp_prior.enabled: true`
- h-unit `mu5_0` initial center is shifted by `log10(h_ref)` relative to the paper-native `mu5_0`, matching the current Sonnenfeld h-unit config convention

Run:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_config_profiles_io.py -q
```

### Task 7: Add Synthetic Runtime Tests

Create or extend:

```text
Bayesian_inference/tests/test_sonnenfeld_sigma_star_gamma_runtime_model.py
```

Use the same tiny canonical dataset fixtures currently used by `test_sonnenfeld_runtime_model.py`.

Required tests:

- paper-native sigma-star-gamma model builds a context and evaluates finite `log_prob`
- h-unit sigma-star-gamma model builds a context and evaluates finite `log_prob`
- returned chain dimension in a tiny emcee smoke run is 11
- original Sonnenfeld smoke tests still report dimension 12

Run:

```bash
conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_sonnenfeld_runtime_model.py Bayesian_inference/tests/test_sonnenfeld_sigma_star_gamma_runtime_model.py -q
```

### Task 8: Final Verification

Run the focused regression set:

```bash
conda run -n cmass_lens python -m pytest \
  Bayesian_inference/tests/test_model_registry_config.py \
  Bayesian_inference/tests/test_component_specs.py \
  Bayesian_inference/tests/test_config_profiles_io.py \
  Bayesian_inference/tests/test_sonnenfeld_runtime_model.py \
  Bayesian_inference/tests/test_sonnenfeld_sigma_star_gamma_runtime_model.py \
  -q
```

If runtime tests pass but real config execution fails because the stable HDF5 files do not exist under `data/external`, that is a data-availability blocker, not a model-contract failure. The implementation should report that distinction explicitly.

## Non-Goals

Do not do these in this feature:

- Do not change the scientific meaning of `sonnenfeld2024_slacs`.
- Do not change `sonnenfeld2024_slacs_hunit`.
- Do not rewrite canonical dataset generation.
- Do not recompute Sonnenfeld finite-fibre cross-sections in inference runtime.
- Do not replace the production Numba/emcee backend.
- Do not introduce a YAML component switch such as `gamma_model`.
- Do not make one config silently switch between unit conventions.
- Do not alter FP-prior defaults except through explicit config fields.

## Acceptance Criteria

The feature is complete only when all of the following are true:

- The new package exists beside `cmass` and `sonnenfeld2024_slacs`.
- Registry exposes exactly two new concrete model names.
- Paper-native and h-unit configs are separate YAML files.
- Both new models load through `load_runtime_config`.
- Both new models have 11 sampled parameters.
- Existing Sonnenfeld models still have 12 sampled parameters.
- Both new models evaluate finite `log_prob` on synthetic canonical fixtures.
- Focused tests pass in the `cmass_lens` environment.
- Any missing real `data/external` HDF5 files are reported as data-preparation gaps, not hidden by code fallbacks.

## Read-Only Policy

This document is the frozen target for the planned implementation. After this file is created, it should be made read-only on disk. If the target changes, create a new dated goal document or deliberately change permissions and record the reason in the commit or handoff notes.
