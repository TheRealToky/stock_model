"""CLI entry point for the data pipeline.

Provides subcommands for full ingestion, incremental updates, and
database status inspection.

Usage::

    python -m data_pipeline.runner ingest --tickers AAPL MSFT --source twelvedata --start 2020-01-01
    python -m data_pipeline.runner update --tickers AAPL MSFT
    python -m data_pipeline.runner status
"""

from __future__ import annotations

import argparse
import logging
import sys

from config.settings import settings


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_ingest(args: argparse.Namespace) -> None:
    """Full ingestion for the specified tickers."""
    from data_pipeline.ingestion import DataFetcher

    fetcher = DataFetcher()
    tickers = args.tickers or settings.pipeline.default_tickers
    data_source = args.source or settings.pipeline.default_tickers
    fetcher.run_full_ingestion(
        tickers=tickers,
        data_source=data_source,
        start_date=args.start,
        interval=args.interval,
    )


def cmd_update(args: argparse.Namespace) -> None:
    """Incremental update (new data since last row in DB)."""
    from data_pipeline.ingestion import DataFetcher

    fetcher = DataFetcher()
    tickers = args.tickers or settings.pipeline.default_tickers
    data_source = args.source or settings.pipeline.default_tickers
    fetcher.run_incremental_update(
        tickers=tickers,
        data_source=data_source,
        interval=args.interval,
    )


def cmd_status(args: argparse.Namespace) -> None:
    """Print a summary of what's in the database."""
    from data_pipeline.loader import DataLoader

    loader = DataLoader()
    tickers = loader.get_available_tickers()

    if not tickers:
        print("Database is empty -- no instruments found.")
        return

    print(f"\n{'Ticker':<10} {'Rows':>8}  {'From':>12}  {'To':>12}")
    print("-" * 50)

    for ticker in tickers:
        try:
            date_range = loader.get_date_range(ticker)
            df = loader.load_ohlcv(ticker)
            print(
                f"{ticker:<10} {len(df):>8}  "
                f"{date_range[0]!s:>12}  {date_range[1]!s:>12}"
            )
        except Exception as exc:
            print(f"{ticker:<10}  ERROR: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="quant-lab-pipeline",
        description="Data pipeline CLI for the quantitative research lab.",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- ingest ---------------------------------------------------------
    p_ingest = sub.add_parser("ingest", help="Full historical data ingestion.")
    p_ingest.add_argument(
        "--tickers", nargs="+", default=None,
        help="Ticker symbols to fetch (defaults to config list).",
    )
    p_ingest.add_argument(
        "--start", default=settings.pipeline.default_start_date,
        help="Start date in YYYY-MM-DD format.",
    )
    p_ingest.add_argument(
        "--interval", default=settings.pipeline.default_interval,
        help="Data interval: 1d, 1h, 5m, etc.",
    )
    p_ingest.add_argument(
        "--source", default=settings.pipeline.default_data_source,
        help="Base API to fetch data: 'yahoo', 'twelvedata', 'alpha_vantage', etc.",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # -- update ---------------------------------------------------------
    p_update = sub.add_parser("update", help="Incremental data update.")
    p_update.add_argument("--tickers", nargs="+", default=None)
    p_update.add_argument("--interval", default=settings.pipeline.default_interval)
    p_update.add_argument("--source", default=settings.pipeline.default_data_source)
    p_update.set_defaults(func=cmd_update)

    # -- status ---------------------------------------------------------
    p_status = sub.add_parser("status", help="Show database status.")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    _setup_logging(args.log_level)
    args.func(args)


if __name__ == "__main__":
    main()
