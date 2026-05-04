"""
Interfaces for registry-backed scientific lens-population models.

The inference package has two layers:

- model modules define the scientific parameterization and likelihood formula
- backend/orchestration modules run NumPyro, write outputs, and manage files

``ModelDefinition`` is the narrow contract between those layers.  It is a
dataclass of callables rather than an inheritance-heavy base class because the
current project needs a small, inspectable interface that is friendly to JAX
functions and future one-off scientific models.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .mass_definition import MassDefinition
from .parameter_schema import ParameterSchema
from .types import CompiledModel, RuntimeConfig


@dataclass(frozen=True)
class ModelDefinition:
    """
    Complete registry entry for one scientific model.

    Attributes
    ----------
    name:
        Public model key used by ``model.name`` in YAML.
    default_components:
        Model-owned component defaults.  For CMASS this includes
        ``mass_definition`` and ``gamma_distribution``; a future model may use
        a different set.
    normalize_components:
        Validates raw ``model.components`` and fills model defaults.
    resolve_mass_definition:
        Converts the model's mass component into the shared unit-aware
        ``MassDefinition`` object used by I/O and metadata.
    build_parameter_schema:
        Builds the model-specific sampled-parameter contract.
    build_compiled_model:
        Creates the compiled numerical context for the backend.
    log_prob_value:
        JAX-compatible posterior-component function used inside NumPyro.
    log_prob:
        Host-facing log-probability wrapper used by tests and benchmarks.
    """

    name: str
    default_components: Mapping[str, str]
    normalize_components: Callable[[dict[str, str] | None], dict[str, str]]
    resolve_mass_definition: Callable[[dict[str, str], str], MassDefinition]
    build_parameter_schema: Callable[..., ParameterSchema]
    build_compiled_model: Callable[[RuntimeConfig], CompiledModel]
    log_prob_value: Callable[..., tuple[Any, ...]]
    log_prob: Callable[[np.ndarray, CompiledModel], tuple[float, np.void]]
