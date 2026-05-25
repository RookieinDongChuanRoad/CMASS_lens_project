"""Gaussian stellar-mass component for an already-observed lens sample."""

from __future__ import annotations

from collections.abc import Mapping

from ....interfaces import ParameterSpec
from ...interfaces import ComponentSpec, KernelRef


def gaussian_lens_sample_stellar_mass_component(
    *,
    parameters: tuple[ParameterSpec, ...],
    required_context_fields: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = (),
    metadata: Mapping[str, str | float | int | bool] | None = None,
) -> ComponentSpec:
    """
    Return the observed-sample Gaussian stellar-mass declaration.

    This component describes the distribution of stellar masses inside a sample
    that has already been observed.  It is deliberately separate from
    parent-population mass-function components, because a lens-only inference
    target should not inherit selection-correction semantics from a foreground
    parent population model.
    """

    return ComponentSpec(
        name="population.stellar_mass_function.gaussian_lens_sample",
        kind="stellar_mass_function",
        parameters=parameters,
        required_context_fields=required_context_fields,
        required_capabilities=required_capabilities,
        required_kernels=(KernelRef("distributions", "normal_pdf"),),
        metadata=dict(metadata or {}),
    )


__all__ = ["gaussian_lens_sample_stellar_mass_component"]
