# CMASS Lens-Only Model

`cmass_lens_only` is a Sonnenfeld-style lens-only comparison model for CMASS.
It fits the already-observed CMASS lens sample directly.

## Scientific Meaning

The model estimates the distribution of `m5` and `gamma` for observed lenses.
It does not infer a parent population that is later filtered through lensing
selection. This makes its posterior comparable to the "Lens-only" column in
Sonnenfeld 2024 Table 2, not to the fiducial selection-corrected model.

## Included Terms

- observed stellar-mass likelihood
- Gaussian stellar-mass distribution for the observed lens sample
- CMASS h-unit enclosed-mass relation
- CMASS sigma-star-dependent gamma relation
- per-lens Einstein-radius mass grid and Jacobian
- observed velocity-dispersion likelihood, assembled directly in the posterior

## Excluded Terms

- lensing cross-section
- lens-finding probability
- selection normalization
- source-redshift population parameters
- FP prior
- standalone observed-velocity-dispersion likelihood component

## Parameter Order

1. `mu_mstar_lens`
2. `sigma_mstar_lens`
3. `mu5h_0`
4. `beta5h`
5. `xi5h`
6. `sigma5h`
7. `mu_gamma_0`
8. `beta_sigma_star_gamma`
9. `sigma_gamma`

## Implementation Boundary

The model reuses CMASS canonical preprocessing for h-unit pivots, per-lens mass
grids, and velocity-dispersion grids. The posterior has a separate Numba kernel
so selection terms cannot accidentally leak from the default `cmass` posterior.
The posterior imports shared kernels directly instead of depending on helper
functions from `models.cmass.posterior`.
