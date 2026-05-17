# CMASS Model Scientific Constants Refactor Target

## Freeze Rule

This target document is the fixed contract for the CMASS scientific-constant
refactor. After implementation starts, this file must not be edited. Ongoing
status, deviations, and evidence must be recorded in:

`docs/superpowers/progress/2026-05-16-cmass-science-constants-progress.md`

## Final Goal

Bring the default `cmass` model into the current model-ownership architecture
without changing its posterior numerical formulas.

The final design must make `models/cmass/` the unique owner of CMASS-specific
scientific constants and CMASS-specific fixed population definitions. Generic
configuration and shared helper modules may transport or validate values, but
they must not silently define CMASS scientific defaults.

## Architectural Contract

The desired architecture is:

1. `models/cmass/assembly.py` owns the sampled parameter surface, component
   selection, model metadata, unit convention, and mass-aperture contract.
2. `models/cmass/constants.py` owns CMASS fixed scientific constants.
3. `models/cmass/preprocessing.py` owns parameter-independent CMASS context
   construction from a canonical dataset, using only model-owned constants for
   CMASS scientific defaults.
4. `models/cmass/posterior.py` continues to own the CMASS posterior structure,
   fused loops, optional FP prior contribution, and final log-probability.
5. Shared modules such as `types.py`, `config.py`, `profiles.py`, and
   `compiled_context.py` must not be the primary source of CMASS scientific
   constants.

## Scope

In scope:

- Introduce `models/cmass/constants.py`.
- Move or expose through that file the devauc/sersic CMASS population constants.
- Move CMASS lens-redshift prior constants, gamma truncation limits,
  normalization floor, and stellar-mass pivot into the CMASS model namespace.
- Lock the default CMASS FP-prior constants to the 2026-04-29 run semantics:
  - `fiducial_scatter = 0.075`
  - `scatter_error = 0.003`
  - `mu_v_prior = 2.34548`
  - `mu_v_error = 0.00611`
  - `beta_v_prior = 0.176`
  - `beta_v_error = 0.011`
- Keep YAML-provided FP-prior values as explicit overrides.
- Keep `CMASSModelContext` and `models/cmass/posterior.py` numerical interfaces
  stable unless a verification failure proves a narrow adjustment is required.
- Keep the legacy raw oracle path working, but make it read the same CMASS
  constants rather than carrying duplicated constants.
- Add focused regression coverage for the new ownership boundary and the FP
  default behavior.

Out of scope:

- Rewriting CMASS posterior formulas.
- Changing sampler behavior, output format, chain storage, or corner plotting.
- Removing historical documentation references.
- Deleting `compiled_context.py` in this pass.
- Removing `profile.name` from public run layout or canonical metadata in this
  pass.

## Step Plan

### Step 1: Create CMASS Constants Source

Create `models/cmass/constants.py` with explicit typed data for:

- devauc and sersic CMASS profile/population constants;
- observation field aliases needed by existing profile compatibility;
- CMASS stellar-mass pivot;
- CMASS lens-redshift prior mean and scatter;
- CMASS gamma truncation bounds;
- CMASS normalization floor;
- CMASS FP-prior defaults locked to the 2026-04-29 run semantics.

### Step 2: Keep `profiles.py` as a Compatibility Wrapper

Keep the public `build_profile_spec(profile_name)` function for now, but make
it construct `ProfileSpec` from `models/cmass/constants.py`. The compatibility
wrapper may remain global until downstream code is ready for a thinner profile
metadata type.

### Step 3: Make CMASS FP Defaults Model-Owned

Update configuration/default resolution so that:

- `FPPriorConfig` remains a transport object;
- generic dataclass defaults are not treated as CMASS scientific defaults;
- `model.name: cmass` with `fp_prior.enabled: true` and no explicit numeric
  values uses the model-owned 2026-04-29 constants;
- explicit YAML numeric values override model defaults;
- Sonnenfeld models continue to receive their intended FP defaults.

### Step 4: Route CMASS Preprocessing Through Constants

Update `models/cmass/preprocessing.py` so all fixed CMASS constants come from
`models/cmass/constants.py`. The produced `CMASSModelContext` fields should be
numerically unchanged except for intentional FP-prior default changes.

### Step 5: Route Legacy Oracle Context Through Constants

Update `compiled_context.py` to consume the same CMASS constants. This preserves
legacy oracle comparability while removing duplicated hard-coded CMASS values.

### Step 6: Regression Verification

Add or update tests that prove:

- devauc/sersic `ProfileSpec` values are preserved after moving the source;
- CMASS preprocessing reads the model-owned constants;
- CMASS FP-prior defaults resolve to the 2026-04-29 values when YAML only sets
  `enabled: true`;
- explicit YAML FP-prior numeric overrides still win;
- Sonnenfeld FP-prior defaults are not accidentally changed;
- existing canonical/legacy equivalence tests still pass or, if too expensive
  for routine execution, targeted equivalent tests cover the moved constants.

## Acceptance Criteria

The work is accepted only if all of the following are true:

1. CMASS scientific constants have a single model-owned source under
   `models/cmass/`.
2. `profiles.py` no longer contains the literal devauc/sersic CMASS population
   constants.
3. `models/cmass/preprocessing.py` no longer hard-codes CMASS scientific
   constants such as `11.4`, `0.558`, `0.085`, `1.2`, `2.8`, or `1.0e-10`.
4. `compiled_context.py` no longer carries a second independent copy of those
   CMASS constants.
5. `model.name: cmass` with `fp_prior.enabled: true` and omitted FP numeric
   fields resolves to the 2026-04-29 FP constants listed above.
6. Explicit FP-prior YAML values override model-owned defaults.
7. Sonnenfeld model FP defaults remain equivalent to the Sonnenfeld reference
   constants already represented in the current codebase.
8. Targeted tests pass in the `cmass_lens` environment.
9. This target document remains unchanged after implementation begins.

