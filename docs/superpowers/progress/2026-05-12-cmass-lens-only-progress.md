# CMASS Lens-Only Implementation Progress

Source plan: `docs/superpowers/plans/2026-05-12-cmass-lens-only-model.md`

Execution branch: `cmass-lens-only`

Rules for this progress log:

- The source plan is treated as read-only during implementation.
- Progress is appended after each completed task.
- Test commands are run through the `cmass_lens` conda environment.

## Task 1: Add Lens-Only Registry Contract Tests

Status: completed

Changes:

- Added lens-only test fixture helpers and synthetic config fixture in `Bayesian_inference/tests/conftest.py`.
- Added `Bayesian_inference/tests/test_cmass_lens_only_model.py` with registry, schema, capability, and FP-prior rejection contract tests.

Verification:

- Ran `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_cmass_lens_only_model.py -q`.
- Expected failure observed: tests fail because `cmass_lens_only` is not yet registered.

Commit:

- `a908b2f test: define cmass lens-only model contract`

## Task 2: Add ModelSpec, Context, Runtime, and Registry Entry

Status: completed

Changes:

- Added the reusable observed-sample Gaussian stellar-mass component declaration.
- Added `cmass_lens_only` package with assembly, context, preprocessing, and runtime adapter.
- Registered `cmass_lens_only` in the model package exports and registry.
- Declared required capabilities as lens observations, lensing mass grids, and per-lens S2 velocity-dispersion grids.

Verification:

- Ran the targeted registry/schema tests through `cmass_lens`.
- Expected failure observed: test collection fails because `cmass_lens_only.posterior` does not exist yet.

Commit:

- `6aae3dd feat: add cmass lens-only model shell`

## Task 3: Implement the Lens-Only Posterior Kernel

Status: completed

Changes:

- Added `cmass_lens_only/posterior.py`.
- Implemented the 9-parameter theta unpacking helper.
- Implemented the lens-only Numba likelihood without source-redshift, discovery probability, cross-section weighting, selection normalization, or FP prior.
- Kept observed velocity-dispersion likelihood assembly inside the posterior kernel.
- Reused shared kernels directly, including `normal_pdf`, `trapezoid_1d`, `gaussian_linear_mass_mean`, `sigma_star_linear_gamma_mean`, `sigma_model_from_s2`, and `observed_sigma_likelihood`.

Verification:

- Ran `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_cmass_lens_only_model.py -q`.
- Result: passed.

Commit:

- `36b5c86 feat: implement cmass lens-only posterior`

## Task 4: Prove the Model Ignores Selection and Cross-Section Weights

Status: completed

Changes:

- Added a finite synthetic log-prob test for the lens-only Numba path.
- Added a cross-section invariance test that mutates the canonical cross-section grid by a large factor and verifies unchanged log-probability.
- Added a config rejection test proving removed selection parameters such as `theta0` are not accepted by the lens-only schema.

Verification:

- Ran `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_cmass_lens_only_model.py -q`.
- Result: 6 passed.
- During implementation, the rejection test was aligned with the existing `ParameterSchema` contract: extra public parameters raise `ValueError`, not `KeyError`.

Commit:

- `3213767 test: prove cmass lens-only ignores selection weights`

## Task 5: Add Production Config

Status: completed

Changes:

- Added `Bayesian_inference/configs/cmass_lens_only.yaml`.
- The config uses `model.name: cmass_lens_only`, h-units, the 9D lens-only parameter schema, and a production sampling/output block.

Verification:

- Parsed the config through the current checkout by setting `PYTHONPATH=Bayesian_inference/src:prepare_dataset`.
- Confirmed `cfg.model.name == "cmass_lens_only"`.
- Confirmed public parameter order is `('mu_mstar_lens', 'sigma_mstar_lens', 'mu5h_0', 'beta5h', 'xi5h', 'sigma5h', 'mu_gamma_0', 'beta_sigma_star_gamma', 'sigma_gamma')`.
- Note: a bare `python -c` in this conda env imports an editable package from an older worktree, so non-pytest verification commands must pin `PYTHONPATH` to the current checkout.

Commit:

- `de1b0fb config: add cmass lens-only inference preset`

## Task 6: Add Documentation Near the Model

Status: completed

Changes:

- Added `Bayesian_inference/src/cmass_lens_inference/models/cmass_lens_only/README.md`.
- Appended a `CMASS Lens-Only Model` section to `Bayesian_inference/docs/model_refactor_progress.md`.
- Documented that velocity-dispersion likelihood is assembled directly in the posterior and that shared kernels are imported directly instead of using default-model posterior helpers.

Verification:

- Documentation-only task; no tests required by the plan at this step.

Commit:

- `7146d47 docs: document cmass lens-only semantics`

## Task 7: End-to-End Verification

Status: completed

Verification:

- Ran `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_cmass_lens_only_model.py -q`.
  - Result: 6 passed.
- Ran `conda run -n cmass_lens python -m pytest Bayesian_inference/tests/test_model_registry_config.py Bayesian_inference/tests/test_numba_emcee_inference.py -q`.
  - Result: 22 passed.
- Ran `conda run -n cmass_lens python -m pytest Bayesian_inference/tests -q`.
  - Result: full suite passed with one skipped test.
- Ran real-config smoke with `PYTHONPATH=Bayesian_inference/src:prepare_dataset`.
  - First attempt exposed that the originally planned sersic canonical file is absent in this checkout.
  - Verified the existing devauc h-units canonical dataset has `lens_observations.v1`, `lensing_mass_grids.v1`, `lensing_cross_section.theta_gamma_grid.v1`, `velocity_dispersion.per_lens_s2.v1`, and `velocity_dispersion.fp_within_re.v1`.
  - Updated `Bayesian_inference/configs/cmass_lens_only.yaml` to use the existing devauc canonical dataset.
  - Final smoke result: `log_prob -90.90056582560763`, `kernel cmass_lens_only`, `normalization_value 1.0`.
- Ran `git diff --check`.
  - Result: no output.
- Ran a static grep over lens-only Python files for `from ..cmass.posterior`, `cmass_gamma_population_mean`, and `observed_velocity_dispersion_likelihood`.
  - Result: no matches.

Commit:

- `bfad034 fix: stabilize cmass lens-only verification`

