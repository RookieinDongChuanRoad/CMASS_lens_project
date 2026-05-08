"""Lensing component declarations."""

from .cross_section import theta_gamma_cross_section_component
from .powerlaw import powerlaw_lensing_component

__all__ = ["powerlaw_lensing_component", "theta_gamma_cross_section_component"]
