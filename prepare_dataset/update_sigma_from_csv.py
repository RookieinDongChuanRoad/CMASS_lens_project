"""Thin executable wrapper for the CSV-to-HDF5 sigma update utility.

The reusable implementation lives inside ``prepare_dataset.io`` so it can
be imported by tests. This file exists because the user requested an explicit
script entrypoint for one-off operational runs.
"""

from __future__ import annotations

from prepare_dataset.io.sigma_updates import main


if __name__ == "__main__":
    raise SystemExit(main())
