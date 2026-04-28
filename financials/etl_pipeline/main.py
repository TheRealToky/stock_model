"""CLI entry point for the ETL pipeline.

Examples
--------
Full rebuild over all tickers, default config::

    python -m financials.etl_pipeline.main run --mode full

Incremental update with explicit config and 4 workers::

    python -m financials.etl_pipeline.main run \\
        --mode incremental \\
        --config path/to/pipeline.yaml \\
        --workers 4

Inspect dataset state::

    python -m financials.etl_pipeline.main status
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from financials.etl_pipeline.config.etl_config import load_config
from financials.etl_pipeline.pipeline import ETLPipeline
from financials.etl_pipeline.utils.logging import configure_logging, get_logger
from financials.etl_pipeline.utils.manifest import Manifest

logger = get_logger(__name__)


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI flags into the nested override dict load_config expects."""
    overrides: dict[str, Any] = {}
    if args.tickers:
        overrides.setdefault("extract", {})["tickers"] = args.tickers
    if args.start:
        overrides.setdefault("extract", {})["start_date"] = args.start
    if args.end:
        overrides.setdefault("extract", {})["end_date"] = args.end
    if args.interval:
        overrides.setdefault("extract", {})["interval"] = args.interval
    if args.output:
        overrides.setdefault("load", {})["output_path"] = args.output
    if args.workers is not None:
        overrides.setdefault("runtime", {})["num_workers"] = args.workers
    if args.log_level:
        overrides.setdefault("runtime", {})["log_level"] = args.log_level
    return overrides


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, overrides=_build_overrides(args))
    pipeline = ETLPipeline(cfg)

    summary = (
        pipeline.run_incremental() if args.mode == "incremental" else pipeline.run_full()
    )
    json.dump(summary.as_dict(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if not summary.failed else 1


def cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    manifest = Manifest(cfg.manifest_path)
    summary = manifest.summary()
    summary["dataset_path"] = str(cfg.load.output_path)
    summary["tracked_tickers"] = manifest.all_tickers()
    json.dump(summary, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantlab-etl",
        description="Run the OHLCV → feature ETL pipeline.",
    )
    parser.add_argument("--config", default=None, help="Path to a pipeline YAML config")

    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the pipeline")
    p_run.add_argument(
        "--mode", choices=["full", "incremental"], default="full",
        help="full = process the whole window; incremental = since last manifest watermark",
    )
    p_run.add_argument("--tickers", nargs="+", default=None, help="Subset of tickers")
    p_run.add_argument("--start", default=None, help="Override start date (YYYY-MM-DD)")
    p_run.add_argument("--end", default=None, help="Override end date (YYYY-MM-DD)")
    p_run.add_argument("--interval", default=None, help="Override OHLCV interval, e.g. 1min")
    p_run.add_argument("--output", default=None, help="Override dataset output path")
    p_run.add_argument("--workers", type=int, default=None, help="Process pool size (1 = serial)")
    p_run.add_argument("--log-level", default=None)
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="Print manifest summary")
    p_status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level if hasattr(args, "log_level") and args.log_level else "INFO")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
