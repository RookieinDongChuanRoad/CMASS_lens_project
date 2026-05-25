"""Runtime adapter for the Sonnenfeld sigma-star-gamma model.

The new model changes only the gamma population relation.  Its canonical input
contract, deterministic preprocessing, h-unit shifts, FP-prior context, and
finite-fibre selection data are identical to the existing Sonnenfeld runtime.
Keeping this wrapper thin makes that boundary explicit while preserving a full
peer-model package shape.
"""

from __future__ import annotations

from statistical_sl.models.interfaces import CompiledContextBundle, DataSpec, ModelRuntimeAdapter
from statistical_sl.inference.types import RuntimeConfig
from ..sonnenfeld2024_slacs import runtime as sonnenfeld_runtime


def build_context_bundle(runtime_config: RuntimeConfig) -> CompiledContextBundle:
    """Build the shared Sonnenfeld context for the sigma-star-gamma posterior."""

    return sonnenfeld_runtime.build_context_bundle(runtime_config)


def get_data_spec() -> DataSpec:
    """Return the shared Sonnenfeld context-packing declaration."""

    return sonnenfeld_runtime.get_data_spec()


def get_runtime_adapter() -> ModelRuntimeAdapter:
    """Return the runtime adapter paired with this peer model package."""

    return ModelRuntimeAdapter(
        build_context_bundle=build_context_bundle,
        data_spec=get_data_spec(),
    )


__all__ = ["build_context_bundle", "get_data_spec", "get_runtime_adapter"]
