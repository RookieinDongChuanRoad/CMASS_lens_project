"""Density-slope relation component declarations."""

from .constant import constant_gamma_component
from .mass_size_linear import mass_size_linear_gamma_component
from .sigma_star_linear import sigma_star_linear_gamma_component

__all__ = [
    "constant_gamma_component",
    "mass_size_linear_gamma_component",
    "sigma_star_linear_gamma_component",
]
