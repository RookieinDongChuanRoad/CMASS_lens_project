"""Components shared by Sonnenfeld 2024 SLACS model variants.

The package owns the scientific hooks and deterministic preprocessing used by
both concrete registry entries:

- `sonnenfeld2024_slacs`, the paper-native fixed-5-kpc / physical-mass model;
- `sonnenfeld2024_slacs_hunit`, the explicit hunit canonical-backend variant.

The two public model names differ in their `ModelSpec` unit contract and in
how preprocessing places paper Table-1 mass locations into the active
canonical coordinate.  They deliberately share the same hooks so formula
changes remain auditable in one component package.
"""
