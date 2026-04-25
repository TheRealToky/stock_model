"""Time-alignment and resampling helpers for cross-dataset joins.

Flight data is typically daily-aggregated in UTC while OHLCV data is
tied to a market's *trading day* (which may be tz-local and will skip
weekends/holidays).  The helpers here make it trivial to align the
two on a common index.
"""

from __future__ import annotations

import pandas as pd


def resample_daily(
    df: pd.DataFrame,
    how: str = "sum",
    *,
    index_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate a DataFrame to one row per UTC calendar day.

    Args:
        df: Input DataFrame.
        how: Aggregation method (``"sum"``, ``"mean"``, ``"last"``, ...).
        index_col: Optional column to set as the time index before
            resampling.  If ``None`` the current index is used.

    Returns:
        DataFrame indexed by UTC midnight.
    """
    if df.empty:
        return df.copy()

    work = df.copy()
    if index_col is not None:
        work = work.set_index(index_col)

    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index, utc=True)
    if work.index.tz is None:
        work.index = work.index.tz_localize("UTC")
    else:
        work.index = work.index.tz_convert("UTC")

    resampled = work.resample("1D").agg(how)
    return resampled


def align_on_trading_day(
    flight: pd.DataFrame,
    ohlcv: pd.DataFrame,
    *,
    how: str = "inner",
    shift_days: int = 0,
) -> pd.DataFrame:
    """Merge flight + OHLCV DataFrames on the intersection of trading days.

    * Both frames are expected to be indexed by a tz-aware datetime.
    * The trading-day index is inherited from the OHLCV frame, so
      non-trading days are dropped from the flight frame automatically.
    * ``shift_days`` shifts the flight frame *before* the join -- useful
      for lead/lag alignment (e.g. flights today vs returns tomorrow).

    Args:
        flight: DataFrame with daily flight-derived values.
        ohlcv: DataFrame indexed by trading day with OHLCV columns.
        how: Merge join kind (``"inner"``, ``"left"``, ``"right"``,
             ``"outer"``).
        shift_days: Positive value shifts flights later; negative
                    shifts earlier.

    Returns:
        A merged DataFrame with ``flight_*`` and ``ohlcv_*`` columns
        (originals preserved where unambiguous).
    """
    if flight.empty or ohlcv.empty:
        return pd.DataFrame()

    f = _to_utc_daily(flight.copy())
    o = _to_utc_daily(ohlcv.copy())

    if shift_days:
        f = f.shift(shift_days, freq="1D")

    merged = o.join(f, how=how, lsuffix="_ohlcv", rsuffix="_flight")
    return merged


def _to_utc_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce *df*'s index to UTC midnight timestamps."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index = df.index.floor("D")
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()
