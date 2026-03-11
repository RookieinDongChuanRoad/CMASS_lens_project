"""Thin executable wrapper for the CSV-to-HDF5 sigma update utility.

The reusable implementation lives inside ``interpolation_grids.io`` so it can
be imported by tests. This file exists because the user requested an explicit
script entrypoint for one-off operational runs.
"""

from __future__ import annotations

from interpolation_grids.io.sigma_updates import main


if __name__ == "__main__":
    raise SystemExit(main())
