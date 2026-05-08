"""Stellar-mass-function component declarations."""

from .skewnormal import skewnormal_stellar_mass_function_component
from .smooth_truncated_schechter import smooth_truncated_schechter_component

__all__ = [
    "skewnormal_stellar_mass_function_component",
    "smooth_truncated_schechter_component",
]
