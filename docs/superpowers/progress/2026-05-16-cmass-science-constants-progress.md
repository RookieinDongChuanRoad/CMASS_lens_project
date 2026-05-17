# CMASS Model Scientific Constants Refactor Progress

Target document:

`docs/superpowers/plans/2026-05-16-cmass-science-constants-target.md`

## Status Summary

| Step | Status | Notes |
| --- | --- | --- |
| Step 1: Create CMASS constants source | complete | Added `models/cmass/constants.py` with profile constants, CMASS scalar constants, and 2026-04-29 FP defaults. |
| Step 2: Keep `profiles.py` as compatibility wrapper | complete | `profiles.py` now builds `ProfileSpec` from `models/cmass/constants.py`; it no longer owns the literal devauc/sersic constants. |
| Step 3: Make CMASS FP defaults model-owned | complete | `ModelSpec`/`ModelDefinition` now carry FP defaults; config parsing applies active-model defaults before YAML overrides. |
| Step 4: Route CMASS preprocessing through constants | complete | `models/cmass/preprocessing.py` now reads CMASS scalar constants from `models/cmass/constants.py`. |
| Step 5: Route legacy oracle context through constants | complete | `compiled_context.py` now reads the same CMASS scalar constants. |
| Step 6: Regression verification | complete | Targeted config/profile, Sonnenfeld FP, legacy/canonical, runner, numba FP, compile, and diff checks passed. |

## Session Log

### 2026-05-16

- Created the immutable target document and this progress document.
- Verified the repository worktree was clean before starting implementation.
- Target freeze is now active: do not edit
  `docs/superpowers/plans/2026-05-16-cmass-science-constants-target.md`.
- Completed Step 1 and Step 2:
  - Added `Bayesian_inference/src/cmass_lens_inference/models/cmass/constants.py`.
  - Moved the literal devauc/sersic CMASS constants out of `profiles.py`.
  - Kept `build_profile_spec(profile_name)` as the compatibility entrypoint.
- Completed Step 3:
  - Added `fp_prior_defaults` to the model registry contract.
  - Set `cmass` FP defaults to the 2026-04-29 run constants.
  - Set Sonnenfeld FP defaults explicitly from `paper_constants.py`.
  - Updated config parsing so YAML numeric fields override model defaults.
- Completed Step 4 and Step 5:
  - Replaced CMASS scalar constant literals in production preprocessing.
  - Replaced the matching duplicated literals in the legacy raw oracle context.
  - Kept `CMASSModelContext` and `models/cmass/posterior.py` unchanged.
- Completed Step 6:
  - Added regression coverage for CMASS model-owned FP defaults and explicit YAML override precedence.
  - Extended profile compatibility coverage so `build_profile_spec()` is checked against the CMASS model-owned constants.
  - Verified the old hard-coded CMASS scalar constants no longer appear in `models/cmass/preprocessing.py` or `compiled_context.py`.
  - Verified profile population constants now appear only in `models/cmass/constants.py` within the checked source scope.

## Evidence And Decisions

- Use `models/cmass/constants.py` as the single CMASS-owned source of fixed
  scientific constants.
- Keep `profiles.py` as a compatibility wrapper in this pass to avoid changing
  public call sites and canonical dataset validation.
- Keep `compiled_context.py` in this pass because legacy oracle and real-data
  equivalence tests still reference it.
- Verification commands passed:
  - `conda run -n cmass_lens --no-capture-output python -m compileall -q Bayesian_inference/src/cmass_lens_inference`
  - `conda run -n cmass_lens --no-capture-output python -m pytest Bayesian_inference/tests/test_config_profiles_io.py::test_load_runtime_config_builds_fp_prior_config_when_enabled Bayesian_inference/tests/test_config_profiles_io.py::test_load_runtime_config_fp_prior_overrides_model_defaults Bayesian_inference/tests/test_config_profiles_io.py::test_build_profile_spec_exposes_profile_specific_rules -q`
  - `conda run -n cmass_lens --no-capture-output python -m pytest Bayesian_inference/tests/test_sonnenfeld_runtime_model.py::test_sonnenfeld_fp_prior_defaults_match_reference_fitpars -q`
  - `conda run -n cmass_lens --no-capture-output python -m pytest Bayesian_inference/tests/test_real_data_canonical_equivalence.py::test_real_devauc_canonical_log_prob_matches_legacy_raw_oracle Bayesian_inference/tests/test_real_data_canonical_equivalence.py::test_real_devauc_canonical_fp_prior_diagnostics_are_finite -q`
  - `conda run -n cmass_lens --no-capture-output python -m pytest Bayesian_inference/tests/test_config_profiles_io.py -q`
  - `conda run -n cmass_lens --no-capture-output python -m pytest Bayesian_inference/tests/test_model_registry_config.py -q`
  - `conda run -n cmass_lens --no-capture-output python -m pytest Bayesian_inference/tests/test_numba_emcee_inference.py::test_numba_fp_prior_log_prob_is_finite Bayesian_inference/tests/test_runner_cli.py::test_run_inference_serializes_fp_prior_metadata Bayesian_inference/tests/test_cmass_lens_only_model.py::test_cmass_lens_only_rejects_fp_prior -q`
  - `conda run -n cmass_lens --no-capture-output git diff --check`

## Errors Encountered

| Time | Error | Resolution |
| --- | --- | --- |
| 2026-05-16 | Initial `rg` static check returned exit code 1 for no matches, which `conda run` reported as an error. | Re-ran with an explicit empty-result-tolerant shell wrapper and confirmed no hard-coded CMASS scalar constants remained in the checked files. |
| 2026-05-16 | Initial real-data pytest command used an outdated test node name. | Queried actual test names and reran the correct `test_real_devauc_canonical_log_prob_matches_legacy_raw_oracle` target. |

## Full Inference Check

### 2026-05-16: CMASS devauc with model-owned 2026-04-29 FP defaults

- Ran a full `cmass` + `devauc` + `h_units_v1` inference using a YAML config
  that only set `fp_prior.enabled: true`, so the numerical FP prior defaults
  were supplied by `models/cmass/constants.py`.
- New run directory:
  `outputs/devauc/20260516_153547_devauc_cmass-fp-locked-20260516`.
- Reference run directory:
  `outputs/devauc/20260429_132358_devauc_m5h_sigma_star_fp_prior_slit_rebuilt_obs_within_re_hunits_v1_fpfix_20260429`.
- The new run completed 10000 / 10000 steps with mean acceptance fraction
  `0.3003625`; the 2026-04-29 reference run completed 10000 / 10000 steps
  with mean acceptance fraction `0.2907583333333333`.
- The new run wrote:
  - `posterior_corner.png`
  - `posterior_corner_result.json`
- Comparison artifacts were written under
  `outputs/devauc/comparisons/20260516_fp_locked_vs_20260429/`:
  - `posterior_summary_comparison.json`
  - `posterior_marginal_overlay.png`
- The FP prior metadata values matched the 2026-04-29 reference run exactly
  for `fit_mstar_min`, `pivot_mstar`, `fiducial_scatter`, `scatter_error`,
  `mu_v_prior`, `mu_v_error`, `beta_v_prior`, and `beta_v_error`.
- The old raw observation HDF5 and current canonical HDF5 matched exactly for
  lens order, `log_mstar_obs`, `log_re_obs`, `theta_e_obs`, `z_d`, `z_s`,
  `num_sigma`, `gamma_grid`, `log_enclosed_mass_grid`,
  `dmass_dthetaein_grid`, per-lens `s2_grid`, and the FP
  `within_re/m5_hinvkpc/s_unit_grid`.
- The posterior did not reproduce the 2026-04-29 result. The largest visible
  mismatches were:
  - `mu5h_0`: median `11.4101` in the reference run versus `11.3739` in the
    new run; 68 percent intervals did not overlap.
  - `mu_gamma_0`: median `1.89365` in the reference run versus `1.79376` in
    the new run; 68 percent intervals did not overlap.
- Replayed 500 post-burn-in samples from the 2026-04-29 chain through the
  current CMASS FP-locked log-probability path. The replay artifact is
  `outputs/devauc/comparisons/20260516_fp_locked_vs_20260429/fp_prior_replay_on_0429_samples.json`.
- The replay shows a direct FP calculation mismatch rather than only a sampler
  initialization issue:
  - Stored 2026-04-29 `fp_prior_log_term` median: `-1.3095`.
  - Current-code replay `fp_prior_log_term` median on the same samples:
    `-52.9686`.
  - Stored 2026-04-29 `fpfit_mu` median: `2.3426`.
  - Current-code replay `fpfit_mu` median on the same samples: `2.4014`.
  - Stored 2026-04-29 `fpfit_beta` median: `0.1752`.
  - Current-code replay `fpfit_beta` median on the same samples: `0.2206`.
- Immediate interpretation: locking the FP prior constants is necessary but
  not sufficient to recover the 2026-04-29 posterior. Since the checked
  observation/canonical data products and FP grid are numerically identical,
  the remaining difference is in the FP prior calculation path/definition or
  another runtime posterior detail that changes the FP fitted summary from the
  same theta samples.

### 2026-05-16: h-unit FP mass-coordinate fix and corrected full inference

- Fixed `models/cmass/preprocessing.py` so the CMASS FP prior fit threshold and
  pivot are translated into the active h-unit stellar-mass coordinate before
  posterior kernels use them:
  - public/historical constants remain `fit_mstar_min = 11.0` and
    `pivot_mstar = 11.3`;
  - internal h-unit locations become `11.0 + 2 log10(h_ref)` and
    `11.3 + 2 log10(h_ref)`.
- Added `test_real_devauc_canonical_fp_prior_matches_legacy_raw_oracle` to
  protect this conversion by comparing FP-enabled canonical log probability and
  FP diagnostics against the legacy raw oracle.
- Replayed 500 post-burn-in samples from the 2026-04-29 chain through the
  corrected current code. The current FP diagnostics now match the 2026-04-29
  blob values to floating-point precision:
  - `fp_prior_log_term` max absolute replay delta: `6.854e-12`.
  - `fpfit_mu` max absolute replay delta: `4.441e-16`.
  - `fpfit_beta` max absolute replay delta: `8.049e-16`.
  - `fpfit_scatter` max absolute replay delta: `1.578e-14`.
- Reran the full CMASS devauc FP-enabled inference after the fix.
  - Corrected run directory:
    `outputs/devauc/20260516_160246_devauc_cmass-fp-locked-20260516`.
  - Status: completed 10000 / 10000 steps.
  - Mean acceptance fraction: `0.285325`.
  - Wrote `posterior_corner.png` and `posterior_corner_result.json`.
- Corrected comparison artifacts were written under
  `outputs/devauc/comparisons/20260516_fp_hunit_fixed_vs_20260429/`:
  - `posterior_summary_comparison.json`
  - `posterior_marginal_overlay.png`
- Corrected posterior comparison against the 2026-04-29 run:
  - All checked 68 percent intervals overlap.
  - `mu5h_0` median: `11.4101` in 2026-04-29 versus `11.4119` after the fix.
  - `mu_gamma_0` median: `1.89365` in 2026-04-29 versus `1.88269` after the fix.
  - FP blob medians are also aligned: `fpfit_mu` differs by `-0.00039`,
    `fpfit_beta` by `+0.000153`, and `fpfit_scatter` by `-0.000117`.
- Verification commands passed after the fix:
  - `conda run -n cmass_lens --no-capture-output env PYTHONPATH="$PY_PATH" python -m compileall -q Bayesian_inference/src/cmass_lens_inference`
  - `conda run -n cmass_lens --no-capture-output env PYTHONPATH="$PY_PATH" python -m pytest Bayesian_inference/tests/test_real_data_canonical_equivalence.py::test_real_devauc_canonical_fp_prior_matches_legacy_raw_oracle Bayesian_inference/tests/test_real_data_canonical_equivalence.py::test_real_devauc_canonical_log_prob_matches_legacy_raw_oracle -q`
  - `conda run -n cmass_lens --no-capture-output git diff --check`

### 2026-05-16: residual posterior-median mismatch diagnosis

- Rechecked the statement that the corrected FP-prior run had "returned" to
  the 2026-04-29 solution. The stricter conclusion is: the run returned to the
  same broad posterior region, but it is not an exact numerical replay of the
  2026-04-29 posterior target.
- The corrected comparison still shows nonzero median offsets for several
  parameters even though all checked 68 percent intervals overlap:
  - `beta5h`: median offset `+0.0261`, about `0.434` old 68-percent half-width.
  - `xi5h`: median offset `+0.0703`, about `0.479` old 68-percent half-width.
  - `beta_sigma_star_gamma`: median offset `+0.0498`, about `0.434` old
    68-percent half-width.
- Replayed 2026-04-29 post-burn-in samples under the corrected current code.
  FP diagnostics are now exact to floating-point precision, so the residual
  median mismatch is not caused by the FP prior constants or the h-unit FP
  mass-coordinate conversion.
- Tested the normalization kernel directly on 25 post-burn-in 2026-04-29
  chain samples, using the current context but replacing only the current
  two-dimensional `(theta_E, gamma)` cross-section interpolation with the
  2026-04-29 analytic form:
  `pi * (theta_E * interp1d(gamma, cs_over_theta))**2`.
- The 2026-04-29 analytic normalization reproduces the stored 2026-04-29
  `normalization_value` blobs to floating-point precision:
  - `legacy_theta2_minus_stored` q16/median/q84:
    `[-4.619e-16, 0.0, 3.131e-16]`;
  - max absolute difference: `1.332e-15`.
- The current two-dimensional grid interpolation is lower than the stored
  2026-04-29 normalization on the same samples:
  - `current_minus_stored` q16/median/q84:
    `[-0.01026, -0.005239, -0.000839]`;
  - max absolute difference: `0.02053`.
- Immediate conclusion: after the FP h-unit fix, the remaining target-level
  difference is the CMASS selection-normalization cross-section calculation.
  The 2026-04-29 code interpolated `cs_over_theta(gamma)` and applied the
  analytic `theta_E**2` factor. The current code bilinearly interpolates a
  precomputed `cross_section_grid(theta_E, gamma)`, which is mathematically
  close but not numerically identical because it linearly interpolates a grid
  whose theta dependence is quadratic.
- Next required fix for exact 2026-04-29 recovery: make the CMASS model's
  normalization path use the model-owned 2026-04-29-compatible analytic
  `cs_over_theta(gamma) * theta_E**2` calculation, or add an explicitly named
  compatibility mode plus regression test that checks replayed 2026-04-29
  `normalization_value` blobs.
