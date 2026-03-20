"""Load market data from TimescaleDB into pandas DataFrames."""

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from database.connection import get_engine, get_session

logger = logging.getLogger(__name__)


class DataLoader:
    """Read-only access to OHLCV and feature data stored in TimescaleDB."""

    def __init__(self):
        self.engine = get_engine()

    # ------------------------------------------------------------------
    # OHLCV loading
    # ------------------------------------------------------------------

    def load_ohlcv(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Load OHLCV data for a single ticker from the database.

        Args:
            ticker: Instrument ticker symbol.
            start_date: Inclusive start date (ISO string). If ``None``, loads
                from the earliest available timestamp.
            end_date: Inclusive end date (ISO string). If ``None``, loads up to
                the latest available timestamp.
            interval: Bar interval to filter on.

        Returns:
            DataFrame indexed by a tz-aware UTC ``timestamp`` with columns
            ``[open, high, low, close, volume, adjusted_close]``.
        """
        instrument_id = self._resolve_instrument_id(ticker)
        if instrument_id is None:
            logger.warning("Ticker %s not found in instruments table", ticker)
            return pd.DataFrame()

        query, params = self._build_ohlcv_query(
            instrument_id, start_date, end_date, interval,
        )

        df = pd.read_sql(text(query), self.engine, params=params)

        if df.empty:
            logger.info("No OHLCV data found for %s", ticker)
            return df

        df = self._format_ohlcv_result(df)
        logger.info("Loaded %d OHLCV rows for %s", len(df), ticker)
        return df

    def load_multiple(
        self,
        tickers: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Load OHLCV data for multiple tickers.

        Returns:
            Dict mapping ticker -> DataFrame. Tickers with no data are omitted.
        """
        results: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = self.load_ohlcv(ticker, start_date, end_date, interval)
            if not df.empty:
                results[ticker] = df

        logger.info(
            "Loaded data for %d/%d tickers", len(results), len(tickers),
        )
        return results

    # ------------------------------------------------------------------
    # Feature loading
    # ------------------------------------------------------------------

    def load_features(
        self,
        ticker: str,
        feature_names: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Load computed features for a ticker.

        The ``features`` table stores features in long format (one row per
        feature/timestamp).  This method pivots the result so that each
        feature becomes a column.

        Args:
            ticker: Instrument ticker symbol.
            feature_names: List of feature names to load.  If ``None``, loads
                all available features.
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Returns:
            DataFrame indexed by timestamp with one column per feature.
        """
        instrument_id = self._resolve_instrument_id(ticker)
        if instrument_id is None:
            logger.warning("Ticker %s not found in instruments table", ticker)
            return pd.DataFrame()

        query, params = self._build_features_query(
            instrument_id, feature_names, start_date, end_date,
        )

        df = pd.read_sql(text(query), self.engine, params=params)

        if df.empty:
            logger.info("No features found for %s", ticker)
            return df

        # Pivot from long to wide format
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        pivoted = df.pivot_table(
            index="timestamp",
            columns="feature_name",
            values="feature_value",
            aggfunc="last",
        )
        pivoted.columns.name = None
        pivoted = pivoted.sort_index()

        logger.info(
            "Loaded %d features (%d rows) for %s",
            len(pivoted.columns), len(pivoted), ticker,
        )
        return pivoted

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def get_available_tickers(self) -> list[str]:
        """Return a sorted list of all tickers that have OHLCV data stored."""
        session = get_session()
        try:
            rows = session.execute(
                text(
                    """
                    SELECT DISTINCT i.ticker
                    FROM instruments i
                    JOIN ohlcv_data o ON o.instrument_id = i.id
                    ORDER BY i.ticker
                    """
                )
            ).fetchall()
            tickers = [row[0] for row in rows]
            logger.info("Found %d tickers with stored data", len(tickers))
            return tickers
        finally:
            session.close()

    def get_date_range(
        self,
        ticker: str,
        interval: str = "1d",
    ) -> tuple[datetime | None, datetime | None]:
        """Return the (min_timestamp, max_timestamp) stored for a ticker.

        Returns:
            Tuple of (earliest, latest) datetime objects, or ``(None, None)``
            if the ticker has no data.
        """
        instrument_id = self._resolve_instrument_id(ticker)
        if instrument_id is None:
            return None, None

        session = get_session()
        try:
            row = session.execute(
                text(
                    """
                    SELECT MIN(timestamp), MAX(timestamp)
                    FROM ohlcv_data
                    WHERE instrument_id = :instrument_id
                      AND interval = :interval
                    """
                ),
                {"instrument_id": instrument_id, "interval": interval},
            ).fetchone()

            if row is None or row[0] is None:
                return None, None

            logger.info(
                "Date range for %s (%s): %s -> %s",
                ticker, interval, row[0], row[1],
            )
            return row[0], row[1]
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_instrument_id(self, ticker: str) -> int | None:
        """Look up the instrument id for a ticker symbol."""
        session = get_session()
        try:
            row = session.execute(
                text("SELECT id FROM instruments WHERE ticker = :ticker"),
                {"ticker": ticker},
            ).fetchone()
            return row[0] if row else None
        finally:
            session.close()

    @staticmethod
    def _build_ohlcv_query(
        instrument_id: int,
        start_date: str | None,
        end_date: str | None,
        interval: str,
    ) -> tuple[str, dict]:
        """Build a parameterized SQL query for OHLCV data."""
        clauses = [
            "instrument_id = :instrument_id",
            "interval = :interval",
        ]
        params: dict = {
            "instrument_id": instrument_id,
            "interval": interval,
        }

        if start_date is not None:
            clauses.append("timestamp >= :start_date")
            params["start_date"] = start_date

        if end_date is not None:
            clauses.append("timestamp <= :end_date")
            params["end_date"] = end_date

        where = " AND ".join(clauses)
        query = f"""
            SELECT timestamp, open, high, low, close, volume, adjusted_close
            FROM ohlcv_data
            WHERE {where}
            ORDER BY timestamp
        """
        return query, params

    @staticmethod
    def _build_features_query(
        instrument_id: int,
        feature_names: list[str] | None,
        start_date: str | None,
        end_date: str | None,
    ) -> tuple[str, dict]:
        """Build a parameterized SQL query for feature data."""
        clauses = ["instrument_id = :instrument_id"]
        params: dict = {"instrument_id": instrument_id}

        if feature_names is not None:
            clauses.append("feature_name = ANY(:feature_names)")
            params["feature_names"] = feature_names

        if start_date is not None:
            clauses.append("timestamp >= :start_date")
            params["start_date"] = start_date

        if end_date is not None:
            clauses.append("timestamp <= :end_date")
            params["end_date"] = end_date

        where = " AND ".join(clauses)
        query = f"""
            SELECT timestamp, feature_name, feature_value
            FROM features
            WHERE {where}
            ORDER BY timestamp, feature_name
        """
        return query, params

    @staticmethod
    def _format_ohlcv_result(df: pd.DataFrame) -> pd.DataFrame:
        """Convert raw query results into the standard DataFrame format."""
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        df = df.sort_index()

        # Ensure correct dtypes
        float_cols = ["open", "high", "low", "close", "adjusted_close"]
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)
        if "volume" in df.columns:
            df["volume"] = df["volume"].astype(int)

        return df
