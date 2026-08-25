"""Annualisation constants shared by every performance-metric call site.

Getting these wrong is not a rounding error.  Annualising 1-minute bars at
252 periods/year -- the daily convention -- overstates the Sharpe ratio by
``sqrt(98_280 / 252) ~= 19.7x`` and turns a 500k-bar backtest into a CAGR
computed over ~1,984 imaginary years.  Every aggregate metric helper must
therefore be told the bar interval explicitly; there is no safe default.

Intraday counts assume US regular trading hours: 390 minutes per session,
252 sessions per year.
"""

from __future__ import annotations

import pandas as pd

TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_SESSION = 390

#: Bar interval -> number of bars per year.
PERIODS_PER_YEAR: dict[str, int] = {
    "1min": MINUTES_PER_SESSION * TRADING_DAYS_PER_YEAR,          # 98_280
    "5min": (MINUTES_PER_SESSION // 5) * TRADING_DAYS_PER_YEAR,   # 19_656
    "15min": (MINUTES_PER_SESSION // 15) * TRADING_DAYS_PER_YEAR,  # 6_552
    "30min": (MINUTES_PER_SESSION // 30) * TRADING_DAYS_PER_YEAR,  # 3_276
    "1h": 6 * TRADING_DAYS_PER_YEAR + TRADING_DAYS_PER_YEAR // 2,  # 1_638
    "1d": TRADING_DAYS_PER_YEAR,
    "1wk": 52,
    "1mo": 12,
}

#: Accepted aliases for the canonical interval keys above.
_INTERVAL_ALIASES: dict[str, str] = {
    "1m": "1min", "1t": "1min", "min": "1min", "minute": "1min",
    "5m": "5min", "5t": "5min",
    "15m": "15min", "15t": "15min",
    "30m": "30min", "30t": "30min",
    "60min": "1h", "60m": "1h", "h": "1h", "hour": "1h", "hourly": "1h",
    "d": "1d", "day": "1d", "daily": "1d", "1day": "1d",
    "w": "1wk", "1w": "1wk", "week": "1wk", "weekly": "1wk",
    "m": "1mo", "1month": "1mo", "month": "1mo", "monthly": "1mo",
}


def normalise_interval(interval: str) -> str:
    """Map an interval string onto a canonical :data:`PERIODS_PER_YEAR` key.

    Args:
        interval: Bar interval, e.g. ``"1min"``, ``"1m"``, ``"daily"``.

    Returns:
        The canonical key.

    Raises:
        ValueError: If *interval* is not recognised.
    """
    key = str(interval).strip().lower()
    key = _INTERVAL_ALIASES.get(key, key)
    if key not in PERIODS_PER_YEAR:
        raise ValueError(
            f"Unknown interval {interval!r}. "
            f"Known intervals: {sorted(PERIODS_PER_YEAR)}"
        )
    return key


def periods_per_year(interval: str) -> int:
    """Return the number of bars per year for *interval*.

    Args:
        interval: Bar interval, e.g. ``"1min"`` or ``"1d"``.

    Returns:
        Bars per year, for use as the ``periods`` annualisation factor.

    Raises:
        ValueError: If *interval* is not recognised.
    """
    return PERIODS_PER_YEAR[normalise_interval(interval)]


def infer_interval(index: pd.Index) -> str:
    """Infer the bar interval from a ``DatetimeIndex``.

    Uses the *median* spacing between consecutive timestamps so that
    overnight and weekend gaps do not skew the estimate.  The result is
    snapped to the nearest known interval.

    Args:
        index: A ``DatetimeIndex`` with at least two entries.

    Returns:
        A canonical interval key.

    Raises:
        TypeError: If *index* is not a ``DatetimeIndex``.
        ValueError: If *index* has fewer than two entries.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"infer_interval requires a DatetimeIndex, got {type(index).__name__}.")
    if len(index) < 2:
        raise ValueError("infer_interval requires at least 2 timestamps.")

    median_seconds = float(pd.Series(index).diff().dt.total_seconds().median())
    if not median_seconds > 0:
        raise ValueError("Could not infer a positive bar spacing from the index.")

    # Nominal spacing in seconds for each known interval.
    nominal = {
        "1min": 60.0,
        "5min": 300.0,
        "15min": 900.0,
        "30min": 1_800.0,
        "1h": 3_600.0,
        "1d": 86_400.0,
        "1wk": 604_800.0,
        "1mo": 2_629_800.0,
    }
    # Snap on a log scale so relative error drives the choice.
    return min(nominal, key=lambda k: abs(median_seconds / nominal[k] - 1.0))


def resolve_periods(
    interval: str | None = None,
    periods: int | None = None,
    index: pd.Index | None = None,
) -> int:
    """Resolve the annualisation factor from whichever hint is available.

    Exactly one source is needed.  Precedence is *periods* (explicit wins),
    then *interval*, then inference from *index*.

    Args:
        interval: Bar interval string.
        periods: Explicit bars-per-year override.
        index: Optional ``DatetimeIndex`` to infer the interval from.

    Returns:
        Bars per year.

    Raises:
        ValueError: If none of the three is usable.
    """
    if periods is not None:
        if periods <= 0:
            raise ValueError(f"periods must be positive, got {periods}.")
        return int(periods)
    if interval is not None:
        return periods_per_year(interval)
    if index is not None and isinstance(index, pd.DatetimeIndex) and len(index) >= 2:
        return periods_per_year(infer_interval(index))
    raise ValueError(
        "Cannot annualise without a period count. Pass one of: "
        "periods=<bars per year>, interval='1min'|'1d'|..., or a DatetimeIndex. "
        "There is deliberately no default -- assuming 252 on intraday bars "
        "overstates Sharpe by ~19.7x."
    )
