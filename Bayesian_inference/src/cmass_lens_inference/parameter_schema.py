"""
Generic sampled-parameter schema.

This module deliberately does not know about CMASS gamma modes, Sonnenfeld
parameter names, or any other scientific model.  A concrete model module owns
the scientific meaning of the sampled vector and builds a ``ParameterSchema``
with:

- the model name and component key used to create it
- the internal sampler order
- the public YAML/output order
- box-prior bounds in the same order
- optional static integer codes needed by compiled kernels

Keeping this layer generic is what lets future models expose a different
parameter vector without editing the NumPyro sampler, output writer, or common
validation code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ParameterSchema:
    """
    Public/internal parameter contract for one configured model.

    Parameters
    ----------
    model_name:
        Registry name of the scientific model that produced the schema.
    model_component_key:
        Short component summary, such as ``dependent`` for the CMASS gamma
        distribution.  The field is intentionally generic so non-CMASS models
        can store their own primary variant without inheriting CMASS naming.
    internal_parameter_names:
        Canonical order used by NumPyro, JAX kernels, checkpoints, and compact
        array outputs.
    public_parameter_names:
        User-facing order used by YAML config and metadata.  It may differ from
        the internal names when a model exposes unit-aware names such as
        ``mu5h_0`` while reusing an internal slot name.
    prior_bounds:
        Inclusive box-prior bounds in ``internal_parameter_names`` order.
    static_codes:
        Optional small integer switches required by JIT-compiled kernels.  The
        current CMASS model stores ``{"gamma_mode": ...}``; other models may
        leave this empty or use different keys.
    model_metadata:
        JSON-friendly explanatory payload written into outputs and useful for
        debugging.  It should not be required by hot numerical kernels.
    """

    model_name: str
    model_component_key: str
    internal_parameter_names: tuple[str, ...]
    public_parameter_names: tuple[str, ...]
    prior_bounds: tuple[tuple[float, float], ...]
    static_codes: Mapping[str, int] = field(default_factory=dict)
    model_metadata: Mapping[str, str | float | int | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.internal_parameter_names) != len(self.public_parameter_names):
            raise ValueError("Internal and public parameter name lists must have identical length.")
        if len(self.internal_parameter_names) != len(self.prior_bounds):
            raise ValueError("Prior bounds must have one entry per sampled parameter.")
        if len(set(self.internal_parameter_names)) != len(self.internal_parameter_names):
            raise ValueError("Internal parameter names must be unique.")
        if len(set(self.public_parameter_names)) != len(self.public_parameter_names):
            raise ValueError("Public parameter names must be unique.")

    @property
    def n_dim(self) -> int:
        """Return the sampled parameter dimension for this schema."""

        return len(self.internal_parameter_names)

    @property
    def gamma_mode(self) -> str:
        """
        Return the CMASS gamma-distribution key when present.

        This property is a narrow bridge for retained legacy numerical oracle
        code.  New production code should use ``model_component_key`` or
        ``model_metadata`` rather than treating every model as a gamma-mode
        variant.
        """

        value = self.model_metadata.get("gamma_distribution", self.model_component_key)
        return str(value)

    @property
    def gamma_mode_code(self) -> int:
        """
        Return the CMASS gamma-mode integer code when the active model has one.

        The compiled context still feeds the retained numba oracle and the
        migrated CMASS JAX kernel through this compact code.  A non-CMASS model
        that does not use this code should fail explicitly if that path is
        called before its backend has been implemented.
        """

        if "gamma_mode" not in self.static_codes:
            raise KeyError(
                f"Model '{self.model_name}' does not define a 'gamma_mode' static code."
            )
        return int(self.static_codes["gamma_mode"])

    def validate_theta_shape(self, theta: np.ndarray) -> None:
        """Raise a clear error if the provided theta vector has the wrong size."""

        if theta.shape != (self.n_dim,):
            raise ValueError(
                f"Hyper-parameter vector must contain exactly {self.n_dim} values "
                f"for model '{self.model_name}' component '{self.model_component_key}'."
            )

    def _validate_public_parameter_keys(
        self,
        public_keys: Sequence[str] | set[str],
        *,
        label: str,
    ) -> None:
        """
        Validate one public-name mapping against the active model contract.

        Concrete models decide which names exist.  The generic schema only
        checks that a YAML mapping is complete and has no unexpected keys.
        """

        public_key_set = set(public_keys)
        expected_names = set(self.public_parameter_names)
        unexpected = sorted(public_key_set.difference(expected_names))
        if unexpected:
            raise ValueError(
                f"{label} contains parameters that are not part of model "
                f"'{self.model_name}' component '{self.model_component_key}': "
                f"{', '.join(unexpected)}."
            )

        missing = sorted(expected_names.difference(public_key_set))
        if missing:
            raise ValueError(
                f"{label} is missing required parameters for model "
                f"'{self.model_name}' component '{self.model_component_key}': "
                f"{', '.join(missing)}."
            )

    def normalize_public_values(
        self,
        public_values: Mapping[str, float],
    ) -> dict[str, float]:
        """
        Convert public config keys into the internal parameter-name family.

        The returned mapping is keyed by ``internal_parameter_names`` so callers
        can build sampler vectors without knowing any model-specific public
        naming convention.
        """

        self._validate_public_parameter_keys(
            public_values.keys(),
            label="Initial center",
        )

        normalized: dict[str, float] = {}
        for internal_name, public_name in zip(
            self.internal_parameter_names,
            self.public_parameter_names,
            strict=True,
        ):
            normalized[internal_name] = float(public_values[public_name])
        return normalized

    def normalize_public_box_prior(
        self,
        public_bounds: Mapping[str, Sequence[float]],
    ) -> tuple[tuple[float, float], ...]:
        """Normalize one public-name box-prior mapping into internal order."""

        self._validate_public_parameter_keys(
            public_bounds.keys(),
            label="Box prior",
        )

        normalized_bounds: list[tuple[float, float]] = []
        for public_name in self.public_parameter_names:
            raw_bounds = public_bounds[public_name]
            if len(raw_bounds) != 2:
                raise ValueError(
                    f"Box prior entry '{public_name}' must contain exactly two values: [lower, upper]."
                )

            lower = float(raw_bounds[0])
            upper = float(raw_bounds[1])
            if (not math.isfinite(lower)) or (not math.isfinite(upper)):
                raise ValueError(
                    f"Box prior entry '{public_name}' must use finite numeric bounds."
                )
            if lower > upper:
                raise ValueError(
                    f"Box prior entry '{public_name}' has lower bound {lower:g} greater than upper bound {upper:g}."
                )

            normalized_bounds.append((lower, upper))

        return tuple(normalized_bounds)

    def serialize_public_box_prior(self) -> dict[str, list[float]]:
        """Serialize the configured box prior under the public naming surface."""

        serialized: dict[str, list[float]] = {}
        for public_name, (lower, upper) in zip(
            self.public_parameter_names,
            self.prior_bounds,
            strict=True,
        ):
            serialized[public_name] = [float(lower), float(upper)]
        return serialized

    def validate_theta_in_bounds(
        self,
        theta: np.ndarray,
        *,
        label: str,
    ) -> None:
        """Raise a clear error if one parameter falls outside the box prior."""

        self.validate_theta_shape(theta)
        for public_name, value, (lower, upper) in zip(
            self.public_parameter_names,
            np.asarray(theta, dtype=float),
            self.prior_bounds,
            strict=True,
        ):
            scalar_value = float(value)
            if scalar_value < lower or scalar_value > upper:
                raise ValueError(
                    f"{label} parameter '{public_name}'={scalar_value:g} lies outside "
                    f"the configured box prior [{lower:g}, {upper:g}]."
                )

    def serialize_public_values(
        self,
        internal_values: Mapping[str, float],
    ) -> dict[str, float]:
        """Expose internal values under the model-specific public names."""

        serialized: dict[str, float] = {}
        for internal_name, public_name in zip(
            self.internal_parameter_names,
            self.public_parameter_names,
            strict=True,
        ):
            serialized[public_name] = float(internal_values[internal_name])
        return serialized
