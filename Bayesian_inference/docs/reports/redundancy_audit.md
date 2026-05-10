# Redundancy Audit For Component/Kernel Refactor

This audit supports
`Bayesian_inference/docs/plans/2026-05-08-redundancy-main-comparison.md`.

## Scope

Checked active source, tests, and docs for:

- stale module names: `posterior_kernels`, `production.py`,
  `models/components`, `cmass_kernels`, `sonnenfeld_kernels`;
- retired backend names: `jax`, `numpyro`;
- duplicated helper shapes around theta unpacking, population density, size
  relations, and component-to-kernel declarations.

## Findings

| Item | Location | Classification | Decision |
| --- | --- | --- | --- |
| Historical `posterior_kernels.py` references | `docs/component_kernel_refactor_progress.md`, target/plan history | Historical record | Keep. These references describe completed migrations or acceptance constraints, not active implementation. |
| Historical `production.py` references | older planning/progress docs | Historical record | Keep unless the user asks for historical doc rewrite. Active source no longer uses `production.py`. |
| `jax` / `numpyro` references | `tests/test_no_jax_numpyro_backend.py`, backend migration docs | Guard rail and history | Keep. The test prevents retired backend imports from returning. |
| `models/components` references | historical docs | Historical record | Keep. Active source has no Python files under `models/components`. |
| `unpack_cmass_theta` | `models/cmass/posterior.py` | Intentionally model-specific | Keep. It encodes CMASS gamma-mode theta order and belongs in the model posterior. |
| `_unpack_theta` | `models/sonnenfeld2024_slacs/posterior.py` | Intentionally model-specific | Keep. It encodes Sonnenfeld 12D theta order and belongs in the model posterior. |
| `mu_r` | `models/cmass/posterior.py` | Intentionally model-specific wrapper | Keep. It includes the CMASS Sersic-index option and the CMASS active pivot coordinate. |
| `_size_relation_mean` | `models/sonnenfeld2024_slacs/posterior.py` | Redundant with shared kernel | Replace with `numba_backend.kernels.population.quadratic_size_relation_mean`. |
| `_parent_density_for_draw` Schechter math | `models/sonnenfeld2024_slacs/posterior.py` | Partly redundant with shared kernel | Keep the model-specific truncation-threshold wrapper, but use `smooth_truncated_schechter_density` for the reusable density calculation. |

## Cleanup Boundary

The cleanup should remove duplicated generic math only when the shared kernel
already expresses the same first-principles operation.  It should not merge
CMASS and Sonnenfeld posterior loops, theta unpacking, likelihood reductions,
or model-specific constants.

## Planned Low-Risk Cleanup

1. Add a static test that fails while Sonnenfeld posterior defines private
   duplicate generic population helpers.
2. Import the shared population kernels into
   `models/sonnenfeld2024_slacs/posterior.py`.
3. Delete the private `_size_relation_mean` helper.
4. Rewrite `_parent_density_for_draw` to delegate the reusable density math to
   `smooth_truncated_schechter_density`.
5. Update stale active docstrings/comments that still call model posterior
   code “production kernels” when “posterior” is more precise.
6. Run targeted model tests and the full suite.
