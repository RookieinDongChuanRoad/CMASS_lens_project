"""Concrete scientific lens-population model implementations.

The package initializer deliberately stays lazy.  Some legacy modules import a
single model submodule while they are still being initialized themselves; eager
imports of every model runtime can therefore create circular imports.  The
public aliases are preserved through ``__getattr__`` so registry code can keep a
small import surface without eagerly importing every model runtime.
"""

from __future__ import annotations

from importlib import import_module


_ALIAS_TO_MODULE = {
    "cmass": "statistical_sl.models.cmass",
    "cmass_lens_only": "statistical_sl.models.cmass_lens_only",
    "sonnenfeld2024_slacs": "statistical_sl.models.sonnenfeld2024_slacs",
    "sonnenfeld2024_slacs_sigma_star_gamma": "statistical_sl.models.sonnenfeld2024_slacs_sigma_star_gamma",
    "toy_hierarchical": "statistical_sl.models.toy_hierarchical",
    "cmass_runtime": "statistical_sl.models.cmass.runtime",
    "cmass_lens_only_runtime": "statistical_sl.models.cmass_lens_only.runtime",
    "sonnenfeld2024_slacs_runtime": "statistical_sl.models.sonnenfeld2024_slacs.runtime",
    "sonnenfeld2024_slacs_sigma_star_gamma_runtime": "statistical_sl.models.sonnenfeld2024_slacs_sigma_star_gamma.runtime",
    "toy_hierarchical_runtime": "statistical_sl.models.toy_hierarchical.runtime",
}

__all__ = [
    "cmass",
    "cmass_runtime",
    "cmass_lens_only",
    "cmass_lens_only_runtime",
    "sonnenfeld2024_slacs",
    "sonnenfeld2024_slacs_runtime",
    "sonnenfeld2024_slacs_sigma_star_gamma",
    "sonnenfeld2024_slacs_sigma_star_gamma_runtime",
    "toy_hierarchical",
    "toy_hierarchical_runtime",
]


def __getattr__(name: str):
    """
    Resolve historical model aliases only when callers ask for them.

    Caching the imported module in ``globals()`` keeps repeated accesses cheap
    without reintroducing eager imports at package import time.
    """

    try:
        module_path = _ALIAS_TO_MODULE[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error

    module = import_module(module_path)
    globals()[name] = module
    return module
