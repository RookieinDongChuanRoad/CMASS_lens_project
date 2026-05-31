"""Command-line interface for Statistical_SL pipeline recipes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from statistical_sl.pipeline.recipe import load_pipeline_recipe
from statistical_sl.pipeline.runner import plan_pipeline_run, run_pipeline


def _json_ready(payload: object) -> object:
    """Convert dataclass and Path values into JSON-friendly structures."""

    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {key: _json_ready(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_ready(value) for value in payload]
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the pipeline recipe CLI parser."""

    parser = argparse.ArgumentParser(description="Run or validate Statistical_SL pipeline recipes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a pipeline recipe")
    validate_parser.add_argument("--recipe", required=True, help="Pipeline recipe YAML path")

    run_parser = subparsers.add_parser("run", help="Run a pipeline recipe")
    run_parser.add_argument("--recipe", required=True, help="Pipeline recipe YAML path")
    run_parser.add_argument("--diagnostic-run-id", default=None, help="Optional diagnostics artifact directory name")
    run_parser.add_argument("--dry-run", action="store_true", help="Print planned actions without running inference or diagnostics")

    return parser


def main() -> int:
    """Dispatch the selected pipeline command."""

    parser = build_argument_parser()
    args = parser.parse_args()
    recipe = load_pipeline_recipe(args.recipe)

    if args.command == "validate":
        print(json.dumps({"status": "valid", "recipe": recipe.name, "mode": recipe.mode}, indent=2, sort_keys=True))
        return 0

    if args.command == "run":
        if args.dry_run:
            plan = plan_pipeline_run(recipe, diagnostic_run_id=args.diagnostic_run_id)
            print(json.dumps(_json_ready(asdict(plan)), indent=2, sort_keys=True))
            return 0
        result = run_pipeline(recipe, diagnostic_run_id=args.diagnostic_run_id)
        print(json.dumps(_json_ready(asdict(result)), indent=2, sort_keys=True))
        return 0

    parser.error(f"Unsupported pipeline command: {args.command}")
    return 2
