"""TimescaleDB extractor with chunked, server-side cursor reads.

Key properties:
  * Never loads the full dataset into memory -- streams via SQLAlchemy's
    ``stream_results`` execution option (server-side cursor) and yields
    pandas DataFrames.
  * Splits a long history into fixed-length time windows so that progress
    is observable and worker memory stays bounded.
  * Works one ticker at a time so feature engineering can compute rolling
    windows without leaking data across instruments.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterator

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from financials.database.connection import get_engine
from financials.etl_pipeline.config.etl_config import ExtractConfig
from financials.etl_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


_OHLCV_QUERY = text(
    """
    SELECT timestamp, open, high, low, close, volume, adjusted_close
    FROM ohlcv_data
    WHERE instrument_id = :instrument_id
      AND interval      = :interval
      AND timestamp     >= :start_ts
      AND timestamp     <  :end_ts
    ORDER BY timestamp
    """
)


class TimescaleExtractor:
    """Stream OHLCV bars from a TimescaleDB hypertable.

    The extractor keeps an SQLAlchemy engine but does not hold any open
    connections between calls -- each ``stream_chunks`` invocation opens
    a single connection and closes it when the generator is exhausted.
    """

    def __init__(
        self,
        cfg: ExtractConfig,
        engine: Engine | None = None,
    ) -> None:
        self.cfg = cfg
        self.engine = engine or get_engine()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_tickers(self) -> list[str]:
        """Return every ticker that has at least one row at the requested interval."""
        if self.cfg.tickers:
            return list(self.cfg.tickers)

        query = text(
            """
            SELECT DISTINCT i.ticker
            FROM instruments i
            JOIN ohlcv_data o ON o.instrument_id = i.id
            WHERE o.interval = :interval
            ORDER BY i.ticker
            """
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query, {"interval": self.cfg.interval}).fetchall()
        tickers = [r[0] for r in rows]
        logger.info("Discovered {} tickers in DB at interval={}", len(tickers), self.cfg.interval)
        return tickers

    def get_latest_timestamp(self, ticker: str) -> datetime | None:
        """Highest available timestamp for *ticker* -- used by incremental runs."""
        instrument_id = self._resolve_instrument_id(ticker)
        if instrument_id is None:
            return None

        query = text(
            """
            SELECT MAX(timestamp)
            FROM ohlcv_data
            WHERE instrument_id = :instrument_id
              AND interval      = :interval
            """
        )
        with self.engine.connect() as conn:
            row = conn.execute(
                query, {"instrument_id": instrument_id, "interval": self.cfg.interval}
            ).fetchone()
        if row is None or row[0] is None:
            return None
        return _ensure_utc(row[0])

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream_chunks(
        self,
        ticker: str,
        start: datetime,
        end: datetime | None,
    ) -> Iterator[pd.DataFrame]:
        """Yield OHLCV chunks for *ticker* between *start* and *end* (UTC).

        The window ``[start, end)`` is split into ``chunk_days``-long
        sub-windows. Each sub-window is fetched with a server-side cursor
        and concatenated into a single DataFrame before yielding -- this
        keeps the unit of work small enough for feature computation but
        avoids per-row Python overhead.
        """
        instrument_id = self._resolve_instrument_id(ticker)
        if instrument_id is None:
            logger.warning("Ticker {} not in instruments table -- skipping", ticker)
            return

        start = _ensure_utc(start)
        if end is None:
            end = self.get_latest_timestamp(ticker)
            if end is None:
                logger.warning("No data available for {} -- skipping", ticker)
                return
            # +1 microsecond so the strict-less-than predicate includes the
            # latest row.
            end = end + timedelta(microseconds=1)
        else:
            end = _ensure_utc(end)

        if end <= start:
            logger.info("{}: nothing to extract ({} >= {})", ticker, start, end)
            return

        chunk_delta = timedelta(days=self.cfg.chunk_days)
        cursor_start = start
        chunks_yielded = 0

        while cursor_start < end:
            cursor_end = min(cursor_start + chunk_delta, end)
            df = self._fetch_window(instrument_id, cursor_start, cursor_end)
            cursor_start = cursor_end
            if df.empty:
                continue
            chunks_yielded += 1
            logger.debug(
                "{}: chunk {} rows={} window=[{}, {})",
                ticker, chunks_yielded, len(df), df.index.min(), df.index.max(),
            )
            yield df

        logger.info("{}: streamed {} chunk(s) total", ticker, chunks_yielded)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_window(
        self,
        instrument_id,
        start_ts: datetime,
        end_ts: datetime,
    ) -> pd.DataFrame:
        """Pull a single time window using a server-side cursor."""
        params = {
            "instrument_id": instrument_id,
            "interval": self.cfg.interval,
            "start_ts": start_ts,
            "end_ts": end_ts,
        }

        # ``stream_results=True`` enables a server-side cursor on PostgreSQL,
        # so the rows are streamed back instead of buffered server-side. We
        # manually iterate ``yield_per`` blocks and concatenate them once;
        # this is the most memory-efficient way to materialize a window
        # without surprising the consumer with a generator-of-generators.
        frames: list[pd.DataFrame] = []
        execution_options = {"stream_results": True}

        with self.engine.connect().execution_options(**execution_options) as conn:
            result = conn.execute(_OHLCV_QUERY, params)
            for batch in result.yield_per(self.cfg.server_fetch_size).partitions():
                if not batch:
                    continue
                cols = list(result.keys())
                frames.append(pd.DataFrame.from_records(batch, columns=cols))

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        return self._format(df)

    def _resolve_instrument_id(self, ticker: str):
        """Look up the UUID/int instrument_id for a ticker symbol."""
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT id FROM instruments WHERE ticker = :ticker"),
                {"ticker": ticker},
            ).fetchone()
        return row[0] if row else None

    @staticmethod
    def _format(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize dtypes & index to match the FeatureEngine contract."""
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        for col in ("open", "high", "low", "close", "adjusted_close"):
            if col in df.columns:
                df[col] = df[col].astype("float64")
        if "volume" in df.columns:
            df["volume"] = df["volume"].astype("float64")

        return df


def _ensure_utc(ts: datetime) -> datetime:
    """Return a tz-aware UTC copy of *ts*, treating naive inputs as already UTC."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)
