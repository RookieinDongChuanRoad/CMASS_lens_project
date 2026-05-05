"""Repository-root import shim for the nested ``prepare_dataset`` package.

The source package lives in ``prepare_dataset/prepare_dataset`` because this
directory is both a small project root and a Python package root.  Tests add
the inner directory to ``sys.path`` directly, but user-facing commands are run
from the repository root as:

``conda run -n cmass_lens python -m prepare_dataset ...``

Without this shim Python resolves the outer project directory as a namespace
package and cannot find modules such as ``prepare_dataset.cli``.  Extending the
package search path keeps the public command stable without moving files.
"""

from __future__ import annotations

from pathlib import Path

_INNER_PACKAGE = Path(__file__).resolve().parent / "prepare_dataset"
if str(_INNER_PACKAGE) not in __path__:
    __path__.insert(0, str(_INNER_PACKAGE))

