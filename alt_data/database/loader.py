"""Read-only loader for the cleaned_flights and flight_features tables.

Mirrors the ``data_pipeline.loader.DataLoader`` pattern so notebooks
can load flight data in one line, just like OHLCV data.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import text

from alt_data.database.connection import get_alt_engine
from alt_data.utils.logging import get_logger

logger = get_logger(__name__)


class FlightLoader:
    """Read-only access to stored flight data and features."""

    def __init__(self) -> None:
        self.engine = get_alt_engine()

    # ------------------------------------------------------------------
    # Cleaned flights
    # ------------------------------------------------------------------

    def load_flights(
        self,
        airport_icao: str,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
        direction: str | None = None,
    ) -> pd.DataFrame:
        """Load cleaned flights for an airport over a date range.

        Args:
            airport_icao: ICAO-4 airport code.
            start_date: Inclusive start, ISO string or datetime.
            end_date:   Inclusive end.
            direction:  Optional "arrival" or "departure" filter.

        Returns:
            DataFrame indexed by ``first_seen_dt`` with one row per flight.
        """
        clauses = ["airport_icao = :airport"]
        params: dict = {"airport": airport_icao.upper()}

        if start_date is not None:
            clauses.append("first_seen_dt >= :start")
            params["start"] = start_date
        if end_date is not None:
            clauses.append("first_seen_dt <= :end")
            params["end"] = end_date
        if direction is not None:
            clauses.append("direction = :direction")
            params["direction"] = direction

        where = " AND ".join(clauses)
        query = text(
            f"""
            SELECT icao24, callsign, direction, airport_icao,
                   first_seen, last_seen, first_seen_dt, last_seen_dt,
                   est_departure_airport, est_arrival_airport,
                   est_departure_airport_horiz_distance,
                   est_arrival_airport_horiz_distance
            FROM cleaned_flights
            WHERE {where}
            ORDER BY first_seen_dt
            """
        )
        df = pd.read_sql(query, self.engine, params=params)
        if df.empty:
            logger.info("No cleaned flights for {}", airport_icao)
            return df

        df["first_seen_dt"] = pd.to_datetime(df["first_seen_dt"], utc=True)
        df["last_seen_dt"] = pd.to_datetime(df["last_seen_dt"], utc=True)
        df = df.set_index("first_seen_dt").sort_index()
        logger.info(
            "Loaded {} flight rows for {}", len(df), airport_icao.upper()
        )
        return df

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def load_features(
        self,
        airport_icao: str,
        feature_names: list[str] | None = None,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """Load features for an airport, pivoted to wide format."""
        clauses = ["airport_icao = :airport"]
        params: dict = {"airport": airport_icao.upper()}

        if feature_names is not None:
            clauses.append("feature_name = ANY(:names)")
            params["names"] = list(feature_names)
        if start_date is not None:
            clauses.append('"timestamp" >= :start')
            params["start"] = start_date
        if end_date is not None:
            clauses.append('"timestamp" <= :end')
            params["end"] = end_date

        where = " AND ".join(clauses)
        query = text(
            f"""
            SELECT "timestamp", feature_name, feature_value
            FROM flight_features
            WHERE {where}
            ORDER BY "timestamp"
            """
        )
        long = pd.read_sql(query, self.engine, params=params)
        if long.empty:
            logger.info("No features for {}", airport_icao)
            return pd.DataFrame()

        long["timestamp"] = pd.to_datetime(long["timestamp"], utc=True)
        wide = long.pivot_table(
            index="timestamp",
            columns="feature_name",
            values="feature_value",
            aggfunc="last",
        )
        wide.columns.name = None
        wide = wide.sort_index()
        logger.info(
            "Loaded {} features ({} rows) for {}",
            wide.shape[1], wide.shape[0], airport_icao.upper(),
        )
        return wide

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def available_airports(self) -> list[str]:
        """Return sorted list of airports that have any stored flights."""
        query = text(
            """
            SELECT DISTINCT airport_icao
            FROM cleaned_flights
            ORDER BY airport_icao
            """
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query).fetchall()
        return [r[0] for r in rows]
