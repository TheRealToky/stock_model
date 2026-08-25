"""Turn the 1-minute feature store into a multi-horizon one.

The store holds 1,337 date partitions x ~126 symbols of 1-minute bars, and the
research so far has only ever asked one question of it: can we predict the next
minute? The answer was no, and the reason was mostly cost -- at 1-minute
frequency, round-trip friction swamps any edge the signal might carry.

Rather than re-ingest anything, this module aggregates the bars already on disk
to whatever interval a study wants, so the same data can answer the question at
5 minutes, an hour, or a day.

Two details that are easy to get wrong and expensive to get wrong quietly:

* **Bins must be anchored to the session open, not the clock.** US regular
  hours run 14:30-21:00 UTC, so a naive ``df.resample("1h")`` does not in fact
  glue two sessions together -- but it does bucket from 14:00, which makes each
  session's first "hour" only 30 minutes long and offsets every bin after it
  from the session structure. Bar *n* then means something different from bar
  *n* on the next day, which quietly corrupts any intraday-seasonality feature.
  Bins here start at each session's own first bar, so every session is diced
  identically. (For a market whose session crosses UTC midnight, clock
  bucketing is worse still: it splits one session across two daily bars.)
* **Indicators cannot be resampled, only recomputed.** The mean of twelve
  5-minute RSI values is not the hourly RSI. :func:`add_features` recomputes
  them from the aggregated OHLCV via the existing
  :class:`~financials.features.FeatureEngine`, so the definitions stay
  identical to the ones the ETL already writes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from financials.backtesting.constants import (
    PERIODS_PER_YEAR,
    normalise_interval,
    periods_per_year,
)
from research.lib.labeling import session_ids

__all__ = [
    "PERIODS_PER_YEAR",
    "SUPPORTED_INTERVALS",
    "normalise_interval",
    "periods_per_year",
    "bars_per_period",
    "resample_ohlcv",
    "add_features",
    "load_resampled",
]

#: Intervals this module can aggregate 1-minute bars into.
SUPPORTED_INTERVALS: tuple[str, ...] = ("1min", "5min", "15min", "30min", "1h", "1d")

#: OHLCV aggregation rules. Anything else numeric is carried as "last".
_OHLCV_AGG: dict[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "adjusted_close": "last",
}


def bars_per_period(interval: str, *, source_interval: str = "1min") -> int:
    """How many *source_interval* bars make up one *interval* bar.

    Args:
        interval: Target interval, e.g. ``"15min"``.
        source_interval: Interval of the bars on disk.

    Returns:
        Bars per aggregated bar, e.g. 15 for 15min from 1min.

    Raises:
        ValueError: If the target is finer than the source.
    """
    src = periods_per_year(source_interval)
    tgt = periods_per_year(interval)
    if tgt > src:
        raise ValueError(
            f"Cannot upsample: {interval!r} is finer than the source "
            f"{source_interval!r}."
        )
    return int(round(src / tgt))


def resample_ohlcv(
    df: pd.DataFrame,
    interval: str,
    *,
    session_aware: bool = True,
    drop_partial_bins: bool = False,
    min_bars: int = 1,
) -> pd.DataFrame:
    """Aggregate bars to a coarser *interval*.

    Bins are anchored to the first bar of each session, so a 15-minute bar on a
    09:30 open covers 09:30-09:44 and never straddles the previous close.

    Args:
        df: Bar data indexed by a tz-aware ``DatetimeIndex``, with at least
            ``open``/``high``/``low``/``close``. ``volume`` is summed when
            present; any other numeric column is carried forward as its last
            value in the bin.
        interval: Target interval from :data:`SUPPORTED_INTERVALS`.
        session_aware: Anchor bins per session. Disable only for data that has
            no session structure (e.g. crypto).
        drop_partial_bins: Drop bins holding fewer than a full complement of
            source bars. Useful when a partial final bin would otherwise look
            like a real observation.
        min_bars: Drop bins built from fewer than this many source bars.

    Returns:
        Aggregated DataFrame indexed by each bin's opening timestamp, sorted,
        with a ``n_bars`` column recording how many source bars it aggregates.

    Raises:
        ValueError: If *df* lacks a ``DatetimeIndex`` or required columns.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"resample_ohlcv requires a DatetimeIndex, got {type(df.index).__name__}."
        )
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")

    key = normalise_interval(interval)
    if key not in SUPPORTED_INTERVALS:
        raise ValueError(
            f"interval {interval!r} resolves to {key!r}, which is not in "
            f"{SUPPORTED_INTERVALS}."
        )
    if df.empty:
        return df.copy()

    if key == "1min":
        out = df.copy()
        out["n_bars"] = 1
        return out

    freq = pd.Timedelta(minutes=1) * bars_per_period(key)

    if session_aware:
        sess = session_ids(df.index)
        ts = pd.Series(df.index, index=df.index)
        anchor = ts.groupby(sess).transform("first")
        # Floor each bar's offset from its own session open onto the grid.
        offset = ((df.index - anchor) // freq) * freq
        bin_key = anchor + offset
    else:
        bin_key = df.index.floor(freq)

    agg: dict[str, str] = {}
    for col in df.columns:
        if col in _OHLCV_AGG:
            agg[col] = _OHLCV_AGG[col]
        elif pd.api.types.is_numeric_dtype(df[col]):
            agg[col] = "last"

    grouper = pd.Series(np.asarray(bin_key), index=df.index, name="bin")
    out = df.groupby(grouper).agg(agg)
    out["n_bars"] = df.groupby(grouper).size()
    out.index.name = df.index.name or "timestamp"

    if drop_partial_bins:
        out = out[out["n_bars"] >= bars_per_period(key)]
    if min_bars > 1:
        out = out[out["n_bars"] >= min_bars]

    return out.sort_index()


def add_features(
    df: pd.DataFrame,
    feature_names: list[str] | None = None,
    *,
    drop_warmup: bool = False,
) -> pd.DataFrame:
    """Recompute technical indicators on aggregated bars.

    Indicators are recomputed rather than resampled because most of them are
    non-linear: averaging twelve 5-minute RSI readings does not give the hourly
    RSI, and an aggregated MACD is meaningless. Using the same
    :class:`~financials.features.FeatureEngine` the ETL uses keeps definitions
    identical across horizons.

    Args:
        df: Aggregated OHLCV bars.
        feature_names: Registry feature names. ``None`` computes all of them.
        drop_warmup: Drop leading rows where any feature is still NaN.

    Returns:
        *df* with feature columns appended.

    Raises:
        ImportError: If the feature engine's dependencies are unavailable.
    """
    # Imported lazily: FeatureEngine lives in a package that reaches into the
    # DB layer, so a pure resampling call should not pay for that import.
    from financials.features import FeatureEngine

    # compute_features returns a copy of the input with feature columns
    # appended, so there is nothing to join back on.
    out = FeatureEngine().compute_features(df, feature_names)
    if drop_warmup:
        out = out.dropna()
    return out


def load_resampled(
    loader,
    ticker: str,
    interval: str,
    *,
    start: str | None = None,
    end: str | None = None,
    with_features: bool = False,
    feature_names: list[str] | None = None,
    columns: list[str] | None = None,
    drop_partial_bins: bool = False,
) -> pd.DataFrame:
    """Load one ticker from the store and aggregate it to *interval*.

    Args:
        loader: A configured
            :class:`~financials.etl_pipeline.load.reader.MLDataLoader`.
        ticker: Symbol to load.
        interval: Target interval.
        start: Inclusive start date (``YYYY-MM-DD``).
        end: Inclusive end date.
        with_features: Recompute indicators on the aggregated bars.
        feature_names: Which indicators to recompute. ``None`` = all.
        columns: Extra store columns to carry through the aggregation.
        drop_partial_bins: Drop bins with fewer than a full complement of bars.

    Returns:
        Aggregated bars indexed by bin opening timestamp. Empty if the ticker
        has no data in the range.
    """
    # reslib owns the store's quirks (the truncated final day, dtype coercion).
    from research.lib import reslib

    raw = reslib.load_ohlcv(loader, ticker, start, end, columns=columns)
    if raw.empty:
        return raw

    out = resample_ohlcv(raw, interval, drop_partial_bins=drop_partial_bins)
    if with_features:
        out = add_features(out, feature_names)
    return out
