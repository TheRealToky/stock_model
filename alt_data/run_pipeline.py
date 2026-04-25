"""CLI runner for the alt-data flight pipeline.

Usage examples::

    # Run the full pipeline for one airport over a date range
    python -m alt_data.run_pipeline \
        --airport KJFK --start 2024-01-01 --end 2024-01-10

    # Only ingest + clean (skip features)
    python -m alt_data.run_pipeline \
        --airport KJFK --start 2024-01-01 --end 2024-01-02 --skip-features

    # Multiple airports
    python -m alt_data.run_pipeline \
        --airport KJFK --airport KLAX --start 2024-01-01 --end 2024-01-02

    # Database status
    python -m alt_data.run_pipeline --status
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from alt_data.cleaning.cleaner import FlightDataCleaner
from alt_data.config.settings import alt_settings
from alt_data.features.engine import FlightFeatureEngine
from alt_data.ingestion.fetcher import DataFetcher
from alt_data.database.loader import FlightLoader
from alt_data.database.repository import FlightRepository
from alt_data.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _parse_date(value: str) -> datetime:
    """Parse ``YYYY-MM-DD`` as UTC midnight."""
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def run_for_airport(
    airport: str,
    start: datetime,
    end: datetime,
    *,
    skip_features: bool = False,
) -> dict[str, int]:
    """Run fetch -> clean -> store -> features for a single airport."""
    logger.info(
        "Pipeline start: {} [{} -> {}]",
        airport,
        start.date(),
        end.date(),
    )

    fetcher = DataFetcher()
    cleaner = FlightDataCleaner()
    repo = FlightRepository()

    # 1. Fetch -----------------------------------------------------------
    raw_df = fetcher.fetch(airport, start, end)
    raw_count = repo.save_raw(raw_df)

    # 2. Clean -----------------------------------------------------------
    clean_df = cleaner.clean(raw_df)
    clean_count = repo.save_clean(clean_df)

    summary = {
        "raw_rows": raw_count,
        "clean_rows": clean_count,
        "feature_rows": 0,
    }

    # 3. Features --------------------------------------------------------
    if not skip_features and not clean_df.empty:
        engine = FlightFeatureEngine(repository=repo)
        features_df = engine.compute_features(clean_df)
        if not features_df.empty:
            summary["feature_rows"] = repo.save_features(features_df)

    logger.info(
        "Pipeline finished for {}: {}",
        airport,
        summary,
    )
    return summary


def show_status() -> None:
    """Print a quick summary of what's in the alt-data DB."""
    loader = FlightLoader()
    airports = loader.available_airports()
    if not airports:
        print("Alt-data DB is empty -- no airports ingested.")
        return

    print(f"\n{'Airport':<8}  {'Flights':>10}")
    print("-" * 22)
    for airport in airports:
        df = loader.load_flights(airport)
        print(f"{airport:<8}  {len(df):>10}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="alt-data-pipeline",
        description="Run the OpenSky flight-data pipeline (fetch, clean, store, features).",
    )
    parser.add_argument(
        "--airport",
        action="append",
        help="ICAO-4 airport code.  Repeat for multiple airports.",
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        help="Start date in YYYY-MM-DD (UTC).",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        help="End date in YYYY-MM-DD (UTC).",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Fetch + clean only; skip the feature-engineering stage.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print DB status and exit.",
    )
    parser.add_argument(
        "--log-level",
        default=alt_settings.log_level,
        help="Log level (DEBUG, INFO, WARNING, ERROR).",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)

    if args.status:
        show_status()
        return

    if not args.airport or not args.start or not args.end:
        parser.error("--airport, --start, and --end are required unless --status is used")

    airports = [a.upper() for a in args.airport]
    for airport in airports:
        run_for_airport(
            airport,
            args.start,
            args.end,
            skip_features=args.skip_features,
        )


if __name__ == "__main__":
    main()
