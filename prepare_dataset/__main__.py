"""CLI shim for ``python -m prepare_dataset`` from the repository root."""

from __future__ import annotations

from prepare_dataset.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
