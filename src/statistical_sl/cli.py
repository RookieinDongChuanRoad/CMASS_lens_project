"""Top-level ``statistical-sl`` command-line router.

The root CLI owns only workflow selection.  Each subcommand delegates to the
matching implementation under ``statistical_sl`` while preserving the familiar
``sys.argv`` shape that the workflow-local parsers expect.  Keeping the router
thin prevents it from accumulating workflow-specific config parsing or runtime
side effects.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from importlib import import_module


WorkflowMain = Callable[[], int | None]

WORKFLOW_COMMANDS: dict[str, tuple[str, str]] = {
    "prepare-dataset": ("statistical_sl.data_preparation.cli", "statistical-sl prepare-dataset"),
    "inference": ("statistical_sl.inference.cli", "statistical-sl inference"),
    "posterior-predictive": (
        "statistical_sl.posterior_predictive.cli",
        "statistical-sl posterior-predictive",
    ),
}


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the top-level parser that documents workflow routing.

    The parser intentionally treats every token after ``command`` as opaque
    workflow-local arguments.  This lets ``statistical-sl inference --help`` or
    ``statistical-sl prepare-dataset --input ...`` reach the selected workflow
    parser without the root parser interpreting those flags first.
    """

    parser = argparse.ArgumentParser(
        prog="statistical-sl",
        description="Facade CLI for statistical strong-lensing workflows.",
    )
    parser.add_argument(
        "command",
        choices=tuple(WORKFLOW_COMMANDS),
        help="Workflow to run.",
    )
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected workflow.",
    )
    return parser


def _load_workflow_main(module_name: str) -> WorkflowMain:
    """Import the selected workflow-local CLI entrypoint."""

    module = import_module(module_name)
    return module.main


def _run_workflow_main(*, executable_name: str, arguments: Sequence[str], main_function: WorkflowMain) -> int:
    """Run one workflow parser with isolated ``sys.argv`` and a normalized code."""

    original_argv = sys.argv[:]
    try:
        sys.argv = [executable_name, *arguments]
        result = main_function()
    finally:
        sys.argv = original_argv

    return 0 if result is None else int(result)


def main(argv: Sequence[str] | None = None) -> int:
    """Route one public subcommand to its package-owned implementation."""

    parser = build_argument_parser()
    parsed_args = parser.parse_args(argv)
    module_name, executable_name = WORKFLOW_COMMANDS[parsed_args.command]
    workflow_main = _load_workflow_main(module_name)

    return _run_workflow_main(
        executable_name=executable_name,
        arguments=parsed_args.arguments,
        main_function=workflow_main,
    )


if __name__ == "__main__":
    raise SystemExit(main())
