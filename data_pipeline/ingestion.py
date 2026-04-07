"""Data ingestion from yfinance into TimescaleDB."""

import logging
import time
from datetime import datetime, date, timedelta, timezone

import pandas as pd
import yfinance as yf
from sqlalchemy import text
from twelvedata import TDClient
from collections import deque

from config.settings import settings
from database.connection import get_engine, get_session

from dotenv import load_dotenv
import os
load_dotenv()

av_key = os.getenv("AV_API_KEY", "default_value")
td_key = os.getenv("TD_API_KEY", "default_value")

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches market data from yfinance and stores it in TimescaleDB."""

    def __init__(
        self,
        max_retries: int | None = None,
        retry_delay: int | None = None,
    ):
        self.max_retries = max_retries or settings.pipeline.max_retries
        self.retry_delay = retry_delay or settings.pipeline.retry_delay_seconds
        self.engine = get_engine()
        self._rate_limit_windows: dict[str, deque] = {}

    # ------------------------------------------------------------------
    # yfinance helpers
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        ticker: str,
        data_source: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance for a single ticker.

        Args:
            ticker: Stock/ETF ticker symbol.
            data_source: Selects which provider to fetch data from.
            start_date: ISO date string (YYYY-MM-DD). Defaults to pipeline config.
            end_date: ISO date string. Defaults to today.
            interval: Interval string (1d, 1h, etc.).

        Returns:
            DataFrame with columns [open, high, low, close, volume, adjusted_close]
            indexed by a tz-aware UTC DatetimeIndex named ``timestamp``.
        """
        start_date = start_date or settings.pipeline.default_start_date
        end_date = end_date

        logger.info(
            "Fetching %s OHLCV  [%s -> %s]  interval=%s",
            ticker, start_date, end_date, interval,
        )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if data_source == "yahoo":
                    tk = yf.Ticker(ticker)
                    df = tk.history(
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        auto_adjust=False,
                    )
                elif data_source == "twelvedata":
                    td = TDClient(apikey=td_key)
                    df = self._fetch_twelve_data(
                        ticker=ticker,
                        td=td,
                        start_date=start_date,
                        end_date=end_date,
                        interval=interval,
                    )
                else:
                    df = pd.DataFrame()
                    raise ValueError("No available data source corresponding to {}".format(data_source))

                if df.empty:
                    logger.warning("No data returned for %s", ticker)
                    return pd.DataFrame()

                df = self._normalize_ohlcv(df)
                logger.info(
                    "Fetched %d rows for %s", len(df), ticker,
                )
                return df

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Attempt %d/%d for %s failed: %s",
                    attempt, self.max_retries, ticker, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        logger.error("All %d attempts failed for %s", self.max_retries, ticker)
        raise RuntimeError(
            f"Failed to fetch {ticker} after {self.max_retries} attempts"
        ) from last_error

    def fetch_batch(
        self,
        tickers: list[str],
        data_source: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple tickers.

        Returns:
            Dict mapping ticker -> DataFrame. Tickers that failed are omitted.
        """
        results: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                df = self.fetch_ohlcv(ticker, data_source, start_date, end_date, interval)
                if not df.empty:
                    results[ticker] = df
            except RuntimeError:
                logger.error("Skipping %s after repeated failures", ticker)
        logger.info(
            "Batch fetch complete: %d/%d tickers succeeded",
            len(results), len(tickers),
        )
        return results

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def upsert_instrument(self, ticker: str) -> int:
        """Ensure an instrument row exists for *ticker* and return its id.

        If the ticker is new, a minimal row is inserted using metadata from
        yfinance.  If the ticker already exists, its ``id`` is returned
        directly.
        """
        session = get_session()
        try:
            row = session.execute(
                text("SELECT id FROM instruments WHERE ticker = :ticker"),
                {"ticker": ticker},
            ).fetchone()

            if row is not None:
                return row[0]

            # Fetch metadata from yfinance for the new instrument
            info = self._fetch_instrument_info(ticker)

            session.execute(
                text(
                    """
                    INSERT INTO instruments (ticker, name, exchange, asset_type,
                                             sector, currency, is_active)
                    VALUES (:ticker, :name, :exchange, :asset_type,
                            :sector, :currency, :is_active)
                    ON CONFLICT (ticker) DO NOTHING
                    """
                ),
                info,
            )
            session.commit()

            row = session.execute(
                text("SELECT id FROM instruments WHERE ticker = :ticker"),
                {"ticker": ticker},
            ).fetchone()

            instrument_id = row[0]
            logger.info(
                "Upserted instrument %s -> id=%d", ticker, instrument_id,
            )
            return instrument_id

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def store_ohlcv(
        self,
        instrument_id: int,
        df: pd.DataFrame,
        interval: str = "1d",
    ) -> int:
        """Upsert OHLCV rows into the database.

        Uses an ``ON CONFLICT ... DO UPDATE`` clause so re-runs are safe.

        Args:
            instrument_id: FK into the ``instruments`` table.
            df: DataFrame with the standard OHLCV columns plus a UTC
                ``timestamp`` index (as produced by :meth:`fetch_ohlcv`).
            interval: The bar interval (``1d``, ``1h``, etc.).

        Returns:
            Number of rows upserted.
        """
        if df.empty:
            return 0

        records = self._prepare_ohlcv_records(instrument_id, df, interval)

        upsert_sql = text(
            """
            INSERT INTO ohlcv_data
                (instrument_id, timestamp, open, high, low, close,
                 volume, adjusted_close, interval)
            VALUES
                (:instrument_id, :timestamp, :open, :high, :low, :close,
                 :volume, :adjusted_close, :interval)
            ON CONFLICT (instrument_id, timestamp, interval)
            DO UPDATE SET
                open            = EXCLUDED.open,
                high            = EXCLUDED.high,
                low             = EXCLUDED.low,
                close           = EXCLUDED.close,
                volume          = EXCLUDED.volume,
                adjusted_close  = EXCLUDED.adjusted_close
            """
        )

        batch_size = settings.pipeline.batch_size
        total_upserted = 0

        session = get_session()
        try:
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                session.execute(upsert_sql, batch)
                session.commit()
                total_upserted += len(batch)

            logger.info(
                "Upserted %d OHLCV rows for instrument_id=%d",
                total_upserted, instrument_id,
            )
            return total_upserted

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------
    # High-level orchestration
    # ------------------------------------------------------------------

    def run_full_ingestion(
        self,
        tickers: list[str] | None = None,
        data_source: str = 'yahoo',
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> dict[str, int]:
        """End-to-end ingestion: fetch from yfinance and store in DB.

        Args:
            tickers: List of ticker symbols. Defaults to pipeline config.
            data_source: Selects which provider to fetch data from.
            start_date: ISO date string. Defaults to pipeline config.
            end_date: ISO date string. Defaults to today.
            interval: Bar interval.

        Returns:
            Dict mapping ticker -> number of rows upserted.
        """
        tickers = tickers or settings.pipeline.default_tickers
        start_date = start_date or settings.pipeline.default_start_date

        logger.info(
            "Starting full ingestion for %d tickers [%s -> %s] interval=%s",
            len(tickers), start_date, end_date or "today", interval,
        )

        results: dict[str, int] = {}
        for ticker in tickers:
            try:
                instrument_id = self.upsert_instrument(ticker)
                df = self.fetch_ohlcv(ticker, data_source, start_date, end_date, interval)
                if df.empty:
                    results[ticker] = 0
                    continue
                count = self.store_ohlcv(instrument_id, df, interval)
                results[ticker] = count
            except Exception as exc:
                logger.error("Failed ingestion for %s: %s", ticker, exc)
                results[ticker] = 0

        total = sum(results.values())
        logger.info("Full ingestion complete: %d total rows upserted", total)
        return results

    def run_incremental_update(
        self,
        tickers: list[str] | None = None,
        data_source: str = 'yahoo',
        interval: str = "1d",
    ) -> dict[str, int]:
        """Fetch only new data since the last stored timestamp for each ticker.

        Args:
            tickers: List of ticker symbols. Defaults to pipeline config.
            data_source: Base API to fetch the data. Defaults to yahoo.
            interval: Bar interval.

        Returns:
            Dict mapping ticker -> number of new rows upserted.
        """
        tickers = tickers or settings.pipeline.default_tickers
        logger.info(
            "Starting incremental update for %d tickers, interval=%s",
            len(tickers), interval,
        )

        results: dict[str, int] = {}
        for ticker in tickers:
            try:
                instrument_id = self.upsert_instrument(ticker)
                last_ts = self._get_last_timestamp(instrument_id, interval)

                if last_ts is not None:
                    if interval == "1d":
                        # Start one day after last stored bar to avoid duplication.
                        start_date = (last_ts + timedelta(days=1)).strftime("%Y-%m-%d")
                    elif interval == "1min":
                        # Start one minute after last stored bar to avoid duplication.
                        start_date = (last_ts + timedelta(minutes=1)).strftime("%Y-%m-%d")
                    else:
                        # Start one day after last stored bar to avoid duplication.
                        start_date = (last_ts + timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    start_date = settings.pipeline.default_start_date

                df = self.fetch_ohlcv(ticker, data_source=data_source, start_date=start_date, interval=interval)
                if df.empty:
                    logger.info("No new data for %s", ticker)
                    results[ticker] = 0
                    continue

                count = self.store_ohlcv(instrument_id, df, interval)
                results[ticker] = count

            except Exception as exc:
                logger.error("Incremental update failed for %s: %s", ticker, exc)
                results[ticker] = 0

        total = sum(results.values())
        logger.info("Incremental update complete: %d total new rows", total)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_twelve_data(
        self,
        ticker: str,
        td: TDClient,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1min",
    ):
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date) if end_date else None
        current = end if end else None
        all_dfs = []

        if end is None:
            self._rate_limit("twelvedata", max_per_minute=7)
            # Send the first request on the latest available date
            ts = td.time_series(
                symbol=ticker,
                interval=interval,
                outputsize=5000,
                timezone="utc",
            )
            temp_df = ts.as_pandas()
            temp_df.index.name = 'timestamp'

            all_dfs.append(temp_df)

            current = temp_df.index.min() - timedelta(minutes=1)

        while current > start:
            self._rate_limit("twelvedata", max_per_minute=7)
            ts = td.time_series(
                symbol=ticker,
                interval=interval,
                end_date=current.isoformat(),
                outputsize=5000,
                timezone="utc",
            )
            temp_df = ts.as_pandas()

            if temp_df.empty:
                logger.warning(
                    "API returned empty dataframe for ticker %s [%s -> %s]  interval=%s",
                    ticker, start_date, end_date, interval,
                )
                break

            temp_df.index.name = 'timestamp'

            all_dfs.append(temp_df)
            current = temp_df.index.min() - timedelta(minutes=1)

            # The loop has reached the earliest available datetime
            if len(temp_df) < 5000:
                logger.info(
                    "API returned <5000 row dataframe for ticker %s [%s -> %s]  interval=%s",
                    ticker, start_date, end_date, interval,
                )
                break

        df = pd.concat(all_dfs) if all_dfs else pd.DataFrame()

        # ensure numeric OHLCV
        cols = ["open", "high", "low", "close", "volume"]
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

        # normalize index
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[df.index.notna()]

        # dedup & sort on the index
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        df = df[(df['close'] >= 0) & (df['volume'] >= 0)]

        return df

    def _rate_limit(
        self,
        key: str,
        max_per_minute: int,
    ) -> None:
        if key not in self._rate_limit_windows:
            self._rate_limit_windows[key] = deque(maxlen=max_per_minute)

        request_times = self._rate_limit_windows[key]
        now = time.monotonic()

        # If we've hit the limit
        if len(request_times) == max_per_minute:
            elapsed = now - request_times[0]

            if elapsed < 60:
                wait = 60 - elapsed
                logger.debug("Rate limit [%s]: sleeping %.2fs", key, wait)
                time.sleep(wait)

                # recompute time after sleep
                now = time.monotonic()

        # Record this request
        request_times.append(now)

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a raw yfinance DataFrame into a standard schema.

        * Renames columns to lowercase.
        * Renames ``Adj Close`` -> ``adjusted_close``.
        * Converts the index to tz-aware UTC.
        * Drops non-OHLCV columns (``Dividends``, ``Stock Splits``).
        """
        df = df.copy()

        # yfinance may return columns with mixed casing
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Rename adj_close variant
        for variant in ("adj_close", "adjclose"):
            if variant in df.columns:
                df = df.rename(columns={variant: "adjusted_close"})

        # If there's no adjusted_close column, fall back to close
        if "adjusted_close" not in df.columns:
            df["adjusted_close"] = df["close"]

        # Keep only the columns we need
        keep = ["open", "high", "low", "close", "volume", "adjusted_close"]
        df = df[[c for c in keep if c in df.columns]]

        # Normalize timezone to UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.index.name = "timestamp"
        return df

    def _fetch_instrument_info(self, ticker: str) -> dict:
        """Pull metadata from yfinance for a ticker."""
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
        except Exception as exc:
            logger.warning("Could not fetch info for %s: %s", ticker, exc)
            info = {}

        return {
            "ticker": ticker,
            "name": info.get("shortName") or info.get("longName") or ticker,
            "exchange": info.get("exchange", "UNKNOWN"),
            "asset_type": info.get("quoteType", "EQUITY"),
            "sector": info.get("sector"),
            "currency": info.get("currency", "USD"),
            "is_active": True,
        }

    @staticmethod
    def _prepare_ohlcv_records(
        instrument_id: int,
        df: pd.DataFrame,
        interval: str,
    ) -> list[dict]:
        """Convert a DataFrame into a list of dicts ready for SQL insert."""
        records = []
        for ts, row in df.iterrows():
            records.append(
                {
                    "instrument_id": instrument_id,
                    "timestamp": ts.to_pydatetime(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]),
                    "adjusted_close": float(row.get("adjusted_close", row["close"])),
                    "interval": interval,
                }
            )
        return records

    def _get_last_timestamp(
        self,
        instrument_id: int,
        interval: str,
    ) -> datetime | None:
        """Get the most recent timestamp stored for an instrument/interval."""
        session = get_session()
        try:
            row = session.execute(
                text(
                    """
                    SELECT MAX(timestamp)
                    FROM ohlcv_data
                    WHERE instrument_id = :instrument_id
                      AND interval = :interval
                    """
                ),
                {"instrument_id": instrument_id, "interval": interval},
            ).fetchone()

            if row is None or row[0] is None:
                return None

            ts = row[0]
            # Ensure the returned datetime is tz-aware (UTC)
            if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts

        finally:
            session.close()
