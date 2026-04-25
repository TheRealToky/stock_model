"""DataHub: unified access to flight data + financial data.

Researchers use this class in notebooks to pull joined DataFrames
that are immediately ready for factor modeling / backtesting.

Example::

    hub = DataHub()
    df = hub.get_joined_data(
        airport="KJFK",
        ticker="DAL",
        start="2024-01-01",
        end="2024-02-01",
    )
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import text

from alt_data.query.mappings import AssetMapper
from alt_data.query.time_align import (
    align_on_trading_day,
    resample_daily,
)
from alt_data.database.connection import (
    get_alt_engine,
    get_financial_engine,
)
from alt_data.database.loader import FlightLoader
from alt_data.utils.logging import get_logger

logger = get_logger(__name__)


class DataHub:
    """Unified query entry point for multi-source research datasets."""

    def __init__(
        self,
        flight_loader: FlightLoader | None = None,
        mapper: AssetMapper | None = None,
    ) -> None:
        self.flight_loader = flight_loader or FlightLoader()
        self.mapper = mapper or AssetMapper()
        self._financial_engine = get_financial_engine()
        self._alt_engine = get_alt_engine()

    # ------------------------------------------------------------------
    # Flight data
    # ------------------------------------------------------------------

    def get_flight_data(
        self,
        airport: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        direction: str | None = None,
        features: list[str] | None = None,
        daily: bool = True,
    ) -> pd.DataFrame:
        """Return flight data for an airport.

        Args:
            airport:  ICAO-4 airport code.
            start/end: Date range (inclusive).
            direction: Optional ``"arrival"`` or ``"departure"`` filter.
            features: If provided, load these stored features instead
                      of the raw flight rows.
            daily:    When True and *features* is None, aggregate the
                      raw flight rows to one row per UTC day
                      (arrivals/departures/total).

        Returns:
            DataFrame indexed by a tz-aware UTC datetime.
        """
        if features is not None:
            return self.flight_loader.load_features(
                airport_icao=airport,
                feature_names=features,
                start_date=start,
                end_date=end,
            )

        raw = self.flight_loader.load_flights(
            airport_icao=airport,
            start_date=start,
            end_date=end,
            direction=direction,
        )
        if raw.empty or not daily:
            return raw

        return self._daily_aggregate(raw)

    # ------------------------------------------------------------------
    # OHLCV data
    # ------------------------------------------------------------------

    def get_ohlcv_data(
        self,
        ticker: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV for a ticker from the financial DB.

        The financial DB is queried read-only using its own engine, so
        no dependency on the quant-lab Python package is required at
        runtime.
        """
        clauses = ["i.ticker = :ticker", "o.interval = :interval"]
        params: dict = {"ticker": ticker.upper(), "interval": interval}
        if start is not None:
            clauses.append("o.timestamp >= :start")
            params["start"] = start
        if end is not None:
            clauses.append("o.timestamp <= :end")
            params["end"] = end

        where = " AND ".join(clauses)
        query = text(
            f"""
            SELECT o.timestamp,
                   o.open, o.high, o.low, o.close,
                   o.volume, o.adjusted_close
            FROM ohlcv_data o
            JOIN instruments i ON i.id = o.instrument_id
            WHERE {where}
            ORDER BY o.timestamp
            """
        )
        df = pd.read_sql(query, self._financial_engine, params=params)
        if df.empty:
            logger.info("No OHLCV for {}", ticker)
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        for col in ("open", "high", "low", "close", "adjusted_close"):
            if col in df.columns:
                df[col] = df[col].astype(float)
        logger.info("Loaded {} OHLCV rows for {}", len(df), ticker)
        return df

    # ------------------------------------------------------------------
    # Joined cross-dataset view
    # ------------------------------------------------------------------

    def get_joined_data(
        self,
        airport: str,
        ticker: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        features: list[str] | None = None,
        interval: str = "1d",
        shift_days: int = 0,
        how: str = "inner",
    ) -> pd.DataFrame:
        """Return flights (or features) joined with OHLCV on trading days.

        Workflow:
            1. Pull flight data (daily aggregated or feature-loaded).
            2. Pull OHLCV from the financial DB.
            3. Align by trading day, optionally shifting flights.

        Useful join patterns:
            * ``shift_days=1`` -> flights today vs returns tomorrow.
            * ``how="left"``   -> keep all trading days even if flight
                                  data is missing (will be NaN).
        """
        if not ticker:
            # Try to auto-resolve via the mapper.
            tickers = self.mapper.tickers_for_airport(airport)
            if not tickers:
                raise ValueError(
                    f"No ticker supplied and no mapping for airport {airport!r}"
                )
            ticker = tickers[0]
            logger.info(
                "Auto-mapped airport {} -> primary ticker {}", airport, ticker
            )

        flight = self.get_flight_data(
            airport=airport, start=start, end=end, features=features
        )
        ohlcv = self.get_ohlcv_data(
            ticker=ticker, start=start, end=end, interval=interval
        )

        if flight.empty or ohlcv.empty:
            logger.warning(
                "Empty side in join: flight={}, ohlcv={}",
                flight.empty,
                ohlcv.empty,
            )
            return pd.DataFrame()

        joined = align_on_trading_day(
            flight, ohlcv, how=how, shift_days=shift_days
        )
        joined.attrs["airport"] = airport
        joined.attrs["ticker"] = ticker
        logger.info(
            "Joined dataset for {}/{}: {} rows, {} cols",
            airport, ticker, len(joined), joined.shape[1],
        )
        return joined

    # ------------------------------------------------------------------
    # Multi-asset convenience
    # ------------------------------------------------------------------

    def get_airport_basket(
        self,
        airport: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        feature: str = "net_flow",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Join a single flight feature against *all* tickers mapped to the airport.

        Returns a wide DataFrame with columns::

            <feature>  <ticker1>_close  <ticker2>_close  ...
        """
        tickers = self.mapper.tickers_for_airport(airport)
        if not tickers:
            raise ValueError(f"No tickers mapped to airport {airport}")

        flight = self.get_flight_data(
            airport=airport, start=start, end=end, features=[feature]
        )
        if flight.empty:
            return pd.DataFrame()

        frames = [flight]
        for tk in tickers:
            ohlcv = self.get_ohlcv_data(tk, start=start, end=end, interval=interval)
            if ohlcv.empty:
                continue
            frames.append(
                ohlcv[["close"]].rename(columns={"close": f"{tk}_close"})
            )

        merged = _daily_concat(frames)
        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _daily_aggregate(flight: pd.DataFrame) -> pd.DataFrame:
        """Collapse raw flight rows to daily arrivals/departures/total."""
        d = flight.copy()
        if "direction" not in d.columns:
            raise ValueError("Flight data missing 'direction' column")
        d.index = d.index.tz_convert("UTC").floor("D")
        pivot = (
            d.assign(one=1)
            .groupby([d.index, "direction"])["one"]
            .sum()
            .unstack(fill_value=0)
        )
        pivot["total"] = pivot.sum(axis=1)
        pivot.index.name = "timestamp"
        return pivot


def _daily_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    aligned = [resample_daily(f, how="last") if "close" in f.columns else resample_daily(f, how="sum") for f in frames]
    if not aligned:
        return pd.DataFrame()
    out = aligned[0]
    for extra in aligned[1:]:
        out = out.join(extra, how="outer")
    return out.sort_index()
