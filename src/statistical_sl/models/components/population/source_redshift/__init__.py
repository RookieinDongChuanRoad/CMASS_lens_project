"""Source-redshift component declarations."""

from .gaussian import gaussian_source_redshift_component
from .truncated_nonnegative_gaussian import truncated_nonnegative_gaussian_source_redshift_component

__all__ = [
    "gaussian_source_redshift_component",
    "truncated_nonnegative_gaussian_source_redshift_component",
]
