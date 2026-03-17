"""
Mode-aware parameter schema definitions for the CMASS lens model.

Why this module exists:
- the gamma population model now has two public modes with different sampled
  parameter spaces
- the inference engine, PPC code, and metadata writers all need one shared
  source of truth for parameter ordering and bounds
- keeping this logic outside the kernel files makes the hot numerical path
  simpler while still leaving the scientific contract explicit and testable
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .mass_definition import MassDefinition


GAMMA_MODE_DEPENDENT = "dependent"
GAMMA_MODE_INDEPENDENT = "independent"
GAMMA_MODE_SIGMA_STAR_DEPENDENT = "sigma_star_dependent"

GAMMA_MODE_DEPENDENT_CODE = 0
GAMMA_MODE_INDEPENDENT_CODE = 1
GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE = 2

INTERNAL_MASS_PARAMETER_NAMES: tuple[str, ...] = (
    "mu5_0",
    "beta5",
    "xi5",
    "sigma5",
)

DEPENDENT_GAMMA_PARAMETER_NAMES: tuple[str, ...] = (
    "mu_gamma_0",
    "beta_gamma",
    "xi_gamma",
    "sigma_gamma",
)

INDEPENDENT_GAMMA_PARAMETER_NAMES: tuple[str, ...] = (
    "mu_gamma_0",
    "sigma_gamma",
)

SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES: tuple[str, ...] = (
    "mu_gamma_0",
    "beta_sigma_star_gamma",
    "sigma_gamma",
)

TAIL_PARAMETER_NAMES: tuple[str, ...] = (
    "mu_zs",
    "sigma_zs",
    "theta0",
    "loga",
)

BOX_PRIOR_BOUNDS_BY_INTERNAL_NAME: dict[str, tuple[float, float]] = {
    "mu5_0": (9.0, 12.0),
    "beta5": (-3.0, 3.0),
    "xi5": (-3.0, 3.0),
    "sigma5": (1.0e-2, 0.2),
    "mu_gamma_0": (1.5, 2.5),
    "beta_gamma": (-3.0, 3.0),
    "xi_gamma": (-3.0, 3.0),
    "beta_sigma_star_gamma": (-3.0, 3.0),
    "sigma_gamma": (0.0, 0.5),
    "mu_zs": (1.0, 3.0),
    "sigma_zs": (0.0, 2.0),
    "theta0": (0.0, 3.0),
    "loga": (-1.0, 3.0),
}

_REMOVED_INDEPENDENT_GAMMA_KEYS = frozenset({"beta_gamma", "xi_gamma"})
_REMOVED_SIGMA_STAR_GAMMA_KEYS = frozenset({"beta_gamma", "xi_gamma"})


@dataclass(frozen=True)
class GammaModelConfig:
    """Validated gamma population-mode configuration."""

    mode: str

    def __post_init__(self) -> None:
        if self.mode not in {
            GAMMA_MODE_DEPENDENT,
            GAMMA_MODE_INDEPENDENT,
            GAMMA_MODE_SIGMA_STAR_DEPENDENT,
        }:
            raise ValueError(
                f"Unsupported gamma model mode '{self.mode}'. Expected "
                f"'{GAMMA_MODE_DEPENDENT}', '{GAMMA_MODE_INDEPENDENT}', or "
                f"'{GAMMA_MODE_SIGMA_STAR_DEPENDENT}'."
            )

    @property
    def code(self) -> int:
        """Return the integer code passed into the hot numba kernels."""

        if self.mode == GAMMA_MODE_DEPENDENT:
            return GAMMA_MODE_DEPENDENT_CODE
        if self.mode == GAMMA_MODE_INDEPENDENT:
            return GAMMA_MODE_INDEPENDENT_CODE
        return GAMMA_MODE_SIGMA_STAR_DEPENDENT_CODE


@dataclass(frozen=True)
class ParameterSchema:
    """
    Public/internal parameter contract for one run configuration.

    The first four internal names remain the historical mass-parameter slots so
    the mass-definition translation logic continues to work without duplicating
    the scientific model.
    """

    gamma_model: GammaModelConfig
    mass_definition: MassDefinition
    internal_parameter_names: tuple[str, ...]
    public_parameter_names: tuple[str, ...]

    @property
    def gamma_mode(self) -> str:
        """Expose the mode string directly for readability at call sites."""

        return self.gamma_model.mode

    @property
    def gamma_mode_code(self) -> int:
        """Expose the compact integer representation used inside kernels."""

        return self.gamma_model.code

    @property
    def n_dim(self) -> int:
        """Return the sampled parameter dimension for this schema."""

        return len(self.internal_parameter_names)

    @property
    def prior_bounds(self) -> tuple[tuple[float, float], ...]:
        """Return box-prior bounds aligned with the internal parameter order."""

        return tuple(
            BOX_PRIOR_BOUNDS_BY_INTERNAL_NAME[name]
            for name in self.internal_parameter_names
        )

    def validate_theta_shape(self, theta: np.ndarray) -> None:
        """Raise a clear error if the provided theta vector has the wrong size."""

        if theta.shape != (self.n_dim,):
            raise ValueError(
                f"Hyper-parameter vector must contain exactly {self.n_dim} values "
                f"for gamma mode '{self.gamma_mode}'."
            )

    def normalize_public_values(
        self,
        public_values: Mapping[str, float],
    ) -> dict[str, float]:
        """
        Convert public config keys into the internal parameter-name family.

        The independent gamma mode removes `beta_gamma` and `xi_gamma`
        completely. Rejecting them explicitly is important because silently
        ignoring them would hide a model mismatch in the input config.
        """

        if self.gamma_mode == GAMMA_MODE_INDEPENDENT:
            forbidden = sorted(_REMOVED_INDEPENDENT_GAMMA_KEYS.intersection(public_values.keys()))
            if forbidden:
                raise ValueError(
                    "Independent gamma mode does not accept removed gamma slope "
                    f"parameters: {', '.join(forbidden)}."
                )
        elif self.gamma_mode == GAMMA_MODE_SIGMA_STAR_DEPENDENT:
            forbidden = sorted(_REMOVED_SIGMA_STAR_GAMMA_KEYS.intersection(public_values.keys()))
            if forbidden:
                raise ValueError(
                    "Sigma-star gamma mode does not accept the dependent-mode "
                    f"gamma slope parameters: {', '.join(forbidden)}."
                )

        expected_names = set(self.public_parameter_names)
        unexpected = sorted(set(public_values.keys()).difference(expected_names))
        if unexpected:
            raise ValueError(
                "Initial center contains parameters that are not part of the "
                f"'{self.gamma_mode}' gamma-mode schema: {', '.join(unexpected)}."
            )

        normalized: dict[str, float] = {}
        for internal_name, public_name in zip(
            self.internal_parameter_names,
            self.public_parameter_names,
            strict=True,
        ):
            normalized[internal_name] = float(public_values[public_name])
        return normalized

    def serialize_public_values(
        self,
        internal_values: Mapping[str, float],
        mass_definition: MassDefinition | None = None,
    ) -> dict[str, float]:
        """
        Expose internal values under the mode-aware public naming surface.

        The optional `mass_definition` argument keeps the older call sites
        source-compatible while still validating that callers do not mix the
        schema with a different mass-definition family.
        """

        if mass_definition is not None and mass_definition != self.mass_definition:
            raise ValueError(
                "Cannot serialize parameters with a different mass definition "
                "from the one stored in the parameter schema."
            )

        serialized: dict[str, float] = {}
        for internal_name, public_name in zip(
            self.internal_parameter_names,
            self.public_parameter_names,
            strict=True,
        ):
            serialized[public_name] = float(internal_values[internal_name])
        return serialized


def build_parameter_schema(
    gamma_model: GammaModelConfig,
    mass_definition: MassDefinition,
) -> ParameterSchema:
    """Build the single authoritative parameter schema for one run."""

    if gamma_model.mode == GAMMA_MODE_DEPENDENT:
        internal_parameter_names = (
            INTERNAL_MASS_PARAMETER_NAMES
            + DEPENDENT_GAMMA_PARAMETER_NAMES
            + TAIL_PARAMETER_NAMES
        )
        public_parameter_names = (
            mass_definition.public_parameter_names
            + DEPENDENT_GAMMA_PARAMETER_NAMES
            + TAIL_PARAMETER_NAMES
        )
    elif gamma_model.mode == GAMMA_MODE_INDEPENDENT:
        internal_parameter_names = (
            INTERNAL_MASS_PARAMETER_NAMES
            + INDEPENDENT_GAMMA_PARAMETER_NAMES
            + TAIL_PARAMETER_NAMES
        )
        public_parameter_names = (
            mass_definition.public_parameter_names
            + INDEPENDENT_GAMMA_PARAMETER_NAMES
            + TAIL_PARAMETER_NAMES
        )
    else:
        internal_parameter_names = (
            INTERNAL_MASS_PARAMETER_NAMES
            + SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES
            + TAIL_PARAMETER_NAMES
        )
        public_parameter_names = (
            mass_definition.public_parameter_names
            + SIGMA_STAR_DEPENDENT_GAMMA_PARAMETER_NAMES
            + TAIL_PARAMETER_NAMES
        )

    return ParameterSchema(
        gamma_model=gamma_model,
        mass_definition=mass_definition,
        internal_parameter_names=internal_parameter_names,
        public_parameter_names=public_parameter_names,
    )
