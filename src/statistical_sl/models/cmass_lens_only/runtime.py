"""Runtime adapter for the CMASS lens-only model."""

from __future__ import annotations

from statistical_sl.models.interfaces import (
    CompiledContextBundle,
    ContextArraySpec,
    DataSpec,
    ModelRuntimeAdapter,
)
from statistical_sl.inference.types import RuntimeConfig
from .preprocessing import build_cmass_lens_only_context_from_canonical_dataset


def build_context_bundle(runtime_config: RuntimeConfig) -> CompiledContextBundle:
    """Build the lens-only source-context bundle for the generic backend."""

    if runtime_config.data.inference_dataset_path is None:
        raise ValueError("The CMASS lens-only runtime requires data.inference_dataset_path.")
    return build_cmass_lens_only_context_from_canonical_dataset(runtime_config)


def get_data_spec() -> DataSpec:
    """
    Return the lens-only context declaration.

    The current production backend stores the source context directly.  This
    declaration still names the model-owned array unique to lens-only
    evaluation so future packed backends have an explicit contract.
    """

    return DataSpec(
        backend_context_type=object,
        array_fields=(ContextArraySpec("mstar_observation_density"),),
        scalar_fields=(),
        static_fields=(),
        normalization_samples_field="base.base_normals",
        normalization_min_value_field="base.normalization_min_value",
    )


def get_runtime_adapter() -> ModelRuntimeAdapter:
    """Return the runtime adapter paired with `cmass_lens_only.get_model_spec()`."""

    return ModelRuntimeAdapter(
        build_context_bundle=build_context_bundle,
        data_spec=get_data_spec(),
    )


__all__ = ["build_context_bundle", "get_data_spec", "get_runtime_adapter"]
