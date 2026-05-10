# Component And Kernel Refactor Target

## Locked Design

The locked architecture is:

```text
components declare reusable scientific building blocks
numba_backend/kernels implement reusable accelerated numerical functions
models/<model>/assembly.py selects and configures components
models/<model>/posterior.py assembles the concrete model posterior
runner/emcee/output remain model-agnostic framework code
```

The hard rule is that `components/` and `numba_backend/kernels/` must not contain
production-model-specific content.  CMASS, Sonnenfeld, or future model names
belong under `models/<model>/` only.

This means:

- no `components/cmass.py`;
- no `components/sonnenfeld2024_slacs.py`;
- no `numba_backend/cmass_kernels.py`;
- no `numba_backend/sonnenfeld_kernels.py`;
- no helpers such as `unpack_cmass_theta()` inside shared kernel modules;
- no paper-specific constants inside reusable component modules.

The model-specific layer begins only at `models/<model>/`.

## Final Target

The final production flow should be:

```text
canonical inference dataset
  -> models/<model>/assembly.py
       selects component specs,
       assigns model parameter names,
       fixes parameter block order,
       aggregates capabilities
  -> models/<model>/runtime.py
       builds parameter-independent model context
  -> models/<model>/posterior.py
       explicitly assembles posterior(theta)
       calls reusable Numba kernels
       owns likelihood/prior/reduction structure
  -> numba backend engine
       applies box-prior rejection,
       wraps diagnostics,
       calls model posterior
  -> emcee sampler
  -> chain.h5 outputs
```

`production.py` should be renamed to `posterior.py` during this refactor.  The
new name is more accurate: the file owns the scientific posterior structure, not
backend production mechanics.

## Component Boundary

`components/` is a scientific component repository.  It should describe reusable
building blocks from first principles.

Recommended structure:

```text
components/
  interfaces.py

  observations/
    lens_sample.py

  population/
    stellar_mass_function/
      skewnormal.py
      smooth_truncated_schechter.py
    size_relation/
      linear.py
      quadratic.py
    aperture_mass_relation/
      gaussian_linear.py
    gamma_relation/
      constant.py
      mass_size_linear.py
      sigma_star_linear.py
    source_redshift/
      gaussian.py
      truncated_nonnegative_gaussian.py

  lensing/
    powerlaw.py
    cross_section.py

  selection/
    discovery_probability.py
    velocity_proxy.py
```

`components/` should not own final observed-data likelihoods or final priors.
Those belong in `models/<model>/posterior.py`, because they define how a
particular model compares generated quantities with observed data and combines
the final posterior terms.

Each component spec should be allowed to declare the Numba kernels it expects.
This is an audit and testing contract, not a dynamic dispatch mechanism.

Example:

```python
ComponentSpec(
    name="population.stellar_mass_function.skewnormal",
    kind="stellar_mass_function",
    parameters=(...),
    required_context_fields=(...),
    required_capabilities=(...),
    required_kernels=(
        KernelRef("distributions", "skewnorm_sample"),
        KernelRef("distributions", "normal_pdf"),
    ),
)
```

`required_kernels` exists so reviewers and tests can verify that the scientific
component has a corresponding accelerated implementation.  `posterior.py` still
calls kernels explicitly.

## Kernel Boundary

`numba_backend/kernels/` is a reusable accelerated kernel library.  Kernel
modules should be organized by numerical or scientific operation, not by
production model.

Recommended structure:

```text
numba_backend/kernels/
  distributions.py
  interpolation.py
  integration.py
  lensing.py

  population/
    stellar_mass_function.py
    size_relation.py
    aperture_mass_relation.py
    gamma_relation.py
    source_redshift.py

  selection/
    discovery.py
    cross_section.py
    velocity_proxy.py
```

Likelihood factors, optional priors, and posterior reductions should normally
stay in `models/<model>/posterior.py`.  If a small numerical helper becomes
clearly reusable across models, it can be promoted into a shared kernel module,
but it must not carry a model name or model-specific constants.

## Model Boundary

Only `models/<model>/` may contain model-specific content.

Recommended structure:

```text
models/
  cmass/
    assembly.py
    runtime.py
    posterior.py
    constants.py
    context.py

  sonnenfeld2024_slacs/
    assembly.py
    runtime.py
    posterior.py
    paper_constants.py
    context.py
```

`assembly.py` owns:

- selected components;
- model-specific parameter names and public names;
- parameter block order;
- component capability aggregation;
- unit and mass-coordinate contract;
- model metadata.

`runtime.py` owns:

- canonical dataset loading for this model;
- parameter-independent preprocessing;
- context construction.

`posterior.py` owns:

- flat-theta unpacking for this model;
- posterior structure;
- selection normalization reduction;
- observed-data likelihood terms;
- optional priors;
- explicit calls to reusable Numba kernels;
- model-specific fused loops if a fused loop is needed for performance.

If a fused loop is truly model-specific, it belongs in `models/<model>/posterior.py`
or a private `models/<model>/posterior_kernels.py`, not in `numba_backend/`.

## Required Components For Current Models

CMASS should be assembled from:

```text
observations.lens_sample
population.stellar_mass_function.skewnormal
population.size_relation.linear
population.aperture_mass_relation.gaussian_linear
population.gamma_relation.sigma_star_linear
population.source_redshift.truncated_nonnegative_gaussian
lensing.powerlaw
lensing.cross_section
selection.discovery_probability
```

CMASS posterior owns:

```text
observed stellar-mass likelihood
observed size likelihood
observed velocity-dispersion likelihood
selection normalization reduction
optional Fundamental Plane prior
final posterior total
```

Sonnenfeld should be assembled from:

```text
observations.lens_sample
population.stellar_mass_function.smooth_truncated_schechter
population.size_relation.quadratic
population.aperture_mass_relation.gaussian_linear
population.gamma_relation.mass_size_linear
population.source_redshift.gaussian
lensing.powerlaw
lensing.cross_section
selection.discovery_probability
selection.velocity_proxy
```

Sonnenfeld posterior owns:

```text
observed stellar-mass likelihood
observed size likelihood
observed velocity-dispersion likelihood
velocity-proxy selection correction
finite-fibre selection normalization reduction
final posterior total
```

## Implementation Path

### Phase 1: Extend Component Interfaces

Goal:

```text
Make component specs capable of declaring their required kernels without using
that declaration for runtime dispatch.
```

Work:

1. Add a lightweight `KernelRef` type to `components/interfaces.py`.
2. Add `required_kernels` to `ComponentSpec`.
3. Add aggregation or audit helpers only if tests need them.
4. Keep `required_kernels` out of the hot path.

Acceptance:

1. Existing component aggregation tests still pass.
2. New tests can assert that a component declares expected kernel refs.

### Phase 2: Rebuild `components/` By Scientific Modules

Goal:

```text
Remove production-model-named component modules and replace them with reusable
scientific component modules.
```

Work:

1. Create the target component package layout.
2. Move CMASS and Sonnenfeld component declarations into scientific modules.
3. Convert model-specific parameter names, public names, bounds, and paper
   constants into model-side instantiation in `models/<model>/assembly.py`.
4. Delete compatibility envelope specs.
5. Keep `components/` free of CMASS/Sonnenfeld production bundles.

Acceptance:

1. No top-level component module is named after a production model.
2. No component spec contains model names as part of its scientific identity.
3. CMASS and Sonnenfeld assemblies still derive parameter order and
   capabilities from selected component specs.

### Phase 3: Rebuild Shared Kernel Library

Goal:

```text
Make `numba_backend/kernels/` a reusable component-kernel library with no
production-model-specific modules.
```

Work:

1. Split reusable functions from `cmass_kernels.py` and `sonnenfeld_kernels.py`
   into scientific/numerical kernel modules.
2. Remove model-specific theta unpackers from shared kernels.
3. Move model-specific fused loops to `models/<model>/posterior.py` or private
   `models/<model>/posterior_kernels.py`.
4. Keep backend engine/factory/diagnostics generic.

Acceptance:

1. `numba_backend/` contains no CMASS/Sonnenfeld model kernel module.
2. Shared kernel module names do not encode production model names.
3. Shared kernels have focused tests.
4. Existing production log-prob values remain equivalent within established
   tolerances.

### Phase 4: Rename Production Adapters To Posterior Modules

Goal:

```text
Make model-owned posterior assembly explicit by name and responsibility.
```

Work:

1. Rename `models/<model>/production.py` to `models/<model>/posterior.py`.
2. Update `model_registry.py` imports.
3. Keep the registry/factory backend interface unchanged.
4. Keep `posterior.py` responsible for explicit kernel calls and final
   posterior composition.

Acceptance:

1. Registry still resolves CMASS, Sonnenfeld, and toy models.
2. Runner, sampler, output writer, and posterior reader do not need model
   branches.

### Phase 5: Reassemble CMASS And Sonnenfeld Under The Locked Boundary

Goal:

```text
Make the two existing production models the reference examples for the locked
component/kernel/model boundary.
```

Work:

1. Rewrite CMASS assembly to instantiate the selected generic components with
   CMASS parameter names, public names, bounds, constants, and metadata.
2. Rewrite Sonnenfeld assembly to instantiate the selected generic components
   with Sonnenfeld parameter names, bounds, paper constants, and unit variants.
3. Keep all model-specific posterior reductions in model posterior modules.
4. Remove legacy compatibility imports once tests no longer need them.

Acceptance:

1. Reading CMASS assembly explains only the CMASS component selection and model
   parameter contract.
2. Reading Sonnenfeld assembly explains only the Sonnenfeld component selection
   and model parameter contract.
3. Shared components and kernels remain model-name-free.

### Phase 6: Verification

Goal:

```text
Prove that the refactor changed ownership boundaries without changing science
or production behavior.
```

Required checks:

1. Component tests for parameter aggregation, capability aggregation, and
   required-kernel declarations.
2. Shared kernel tests for distributions, interpolation, lensing, population
   relations, source-redshift densities, and selection kernels.
3. CMASS production log-prob regression.
4. Sonnenfeld synthetic/reference log-prob regression.
5. Short emcee smoke tests for CMASS, Sonnenfeld, and toy model.
6. Real-data CMASS equivalence check.
7. Steady-state log-prob benchmark with no unacceptable regression.
8. Static search checks proving no production-model-named component or backend
   kernel modules remain.

## Completion Definition

This work is complete when:

1. `components/` contains only reusable scientific component declarations.
2. `numba_backend/kernels/` contains only reusable accelerated kernels.
3. All CMASS/Sonnenfeld-specific parameter naming, constants, bundles, theta
   order, posterior reductions, and fused loops live under `models/<model>/`.
4. `models/<model>/posterior.py` is the only model-specific posterior assembly
   layer.
5. `runner.py`, `emcee_sampler.py`, `outputs.py`, and `posterior_corner.py`
   remain model-agnostic.
6. The verification suite above passes.
