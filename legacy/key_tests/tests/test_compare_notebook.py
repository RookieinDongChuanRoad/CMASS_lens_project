from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("/Users/liurongfu/Work/CMASS_lens_project/key_tests/notebooks/compare_corner.ipynb")


def _load_notebook() -> dict[str, object]:
    """Load the comparison notebook so structural expectations stay testable."""

    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_compare_notebook_displays_figures_inline_and_keeps_png_export() -> None:
    """
    The comparison notebook must render corner plots inside notebook cells.

    The user still wants the external PNG files preserved, so the plotting
    helper should both save the figure and return the live matplotlib figure
    object to the calling cell.
    """

    notebook = _load_notebook()
    plot_helper_source = "".join(notebook["cells"][3]["source"])
    sersic_plot_cell_source = "".join(notebook["cells"][5]["source"])
    devauc_plot_cell_source = "".join(notebook["cells"][6]["source"])

    assert "fig.savefig(output_path" in plot_helper_source
    assert "return fig" in plot_helper_source
    assert "return output_path" not in plot_helper_source
    assert sersic_plot_cell_source.strip().endswith("sersic_figure")
    assert devauc_plot_cell_source.strip().endswith("devauc_figure")
