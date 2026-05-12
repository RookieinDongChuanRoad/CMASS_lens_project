"""Array context for the CMASS lens-only model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..cmass.context import CMASSModelContext


@dataclass(frozen=True)
class CMASSLensOnlyContext:
    """
    Parameter-independent arrays consumed by the lens-only posterior.

    `base` stores the already validated per-lens mass, kinematic, and
    deterministic covariate grids built by the default preprocessing path.
    `mstar_observation_density` stores only the stellar-mass measurement
    likelihood on the per-lens quadrature grid.  The sampled lens stellar-mass
    population term is intentionally left for the posterior kernel because it
    changes with `mu_mstar_lens` and `sigma_mstar_lens`.
    """

    base: CMASSModelContext
    mstar_observation_density: np.ndarray


__all__ = ["CMASSLensOnlyContext"]
