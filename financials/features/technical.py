"""Pure technical-indicator functions for the quant-lab feature pipeline.

Every public function in this module follows the same contract:
  * Accepts a pandas DataFrame that **must** contain standard OHLCV columns
    (``open``, ``high``, ``low``, ``close``, ``volume``).
  * Returns either a ``pd.Series`` (single indicator) or a ``pd.DataFrame``
    (composite indicator such as MACD or Bollinger Bands).
  * Handles edge cases -- leading NaN values produced by rolling windows are
    left in place so the caller can decide how to deal with them.

No database or I/O side-effects happen here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_ohlcv(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if required OHLCV columns are missing."""
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required OHLCV columns: {missing}")


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def compute_returns(df: pd.DataFrame, periods: int = 1) -> pd.Series:
    """Simple (arithmetic) returns over *periods* bars.

    ``r_t = close_t / close_{t-periods} - 1``
    """
    return df["close"].pct_change(periods=periods)


def compute_log_returns(df: pd.DataFrame, periods: int = 1) -> pd.Series:
    """Logarithmic returns over *periods* bars.

    ``lr_t = ln(close_t / close_{t-periods})``
    """
    return np.log(df["close"] / df["close"].shift(periods))


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def compute_sma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Simple moving average of the close price."""
    return df["close"].rolling(window=window).mean()


def compute_ema(df: pd.DataFrame, span: int = 20) -> pd.Series:
    """Exponential moving average of the close price."""
    return df["close"].ewm(span=span, adjust=False).mean()


def compute_wma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Weighted moving average of the close price.

    Weights increase linearly: the most recent bar receives weight *window*,
    the bar before it *window - 1*, and so on down to 1 for the oldest bar
    in the window.
    """
    weights = np.arange(1, window + 1, dtype=float)
    return df["close"].rolling(window=window).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True,
    )


def compute_hma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Hull moving average of the close price.

    HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))

    The Hull MA reduces lag significantly compared to a standard WMA while
    keeping the curve smooth.
    """
    half_window = max(int(window / 2), 1)
    sqrt_window = max(int(np.sqrt(window)), 1)

    wma_half = compute_wma(df, window=half_window)
    wma_full = compute_wma(df, window=window)

    diff = 2 * wma_half - wma_full

    # Apply a final WMA(sqrt(n)) on the diff series
    weights = np.arange(1, sqrt_window + 1, dtype=float)
    hma = diff.rolling(window=sqrt_window).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True,
    )
    return hma


# ---------------------------------------------------------------------------
# Momentum / oscillators
# ---------------------------------------------------------------------------

def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Relative Strength Index (RSI).

    Uses the *exponential* (Wilder) smoothing method so results match
    most charting platforms.
    """
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence (MACD).

    Returns a DataFrame with columns ``macd``, ``signal``, ``histogram``.
    """
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


# ---------------------------------------------------------------------------
# Volatility / bands
# ---------------------------------------------------------------------------

def compute_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands around the close price.

    Returns a DataFrame with columns ``upper``, ``middle``, ``lower``.
    """
    middle = df["close"].rolling(window=window).mean()
    std = df["close"].rolling(window=window).std()

    return pd.DataFrame({
        "upper": middle + num_std * std,
        "middle": middle,
        "lower": middle - num_std * std,
    })


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (ATR).

    Uses Wilder's smoothing (EWM with ``alpha = 1/window``).
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling standard deviation of simple returns (annualisation is **not**
    applied -- the caller can scale if needed).
    """
    returns = df["close"].pct_change()
    return returns.rolling(window=window).std()


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def compute_rolling_stats(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling descriptive statistics of simple returns.

    Returns a DataFrame with columns ``rolling_mean``, ``rolling_std``,
    ``rolling_skew``, ``rolling_kurtosis``.
    """
    returns = df["close"].pct_change()

    return pd.DataFrame({
        "rolling_mean": returns.rolling(window=window).mean(),
        "rolling_std": returns.rolling(window=window).std(),
        "rolling_skew": returns.rolling(window=window).skew(),
        "rolling_kurtosis": returns.rolling(window=window).kurt(),
    })


# ---------------------------------------------------------------------------
# Volume features
# ---------------------------------------------------------------------------

def compute_volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volume-derived features.

    Returns a DataFrame with columns:
    * ``volume_sma``  -- simple moving average of volume
    * ``volume_ratio`` -- current volume / volume SMA
    * ``obv``         -- On-Balance Volume (cumulative)
    """
    volume_sma = df["volume"].rolling(window=window).mean()

    # Avoid division by zero when SMA is 0 (e.g. at the very start)
    volume_ratio = df["volume"] / volume_sma.replace(0, np.nan)

    # On-Balance Volume: cumulative sum of signed volume
    direction = np.sign(df["close"].diff())
    obv = (direction * df["volume"]).fillna(0).cumsum()

    return pd.DataFrame({
        "volume_sma": volume_sma,
        "volume_ratio": volume_ratio,
        "obv": obv,
    })


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------

def compute_price_direction(df: pd.DataFrame, periods: int = 1) -> pd.Series:
    """Binary price direction: 1 if close rose over *periods* bars, else 0.

    ``direction_t = 1  if close_t > close_{t-periods}  else 0``

    The first *periods* rows will be NaN (no prior close to compare against).
    """
    return (df["close"] > df["close"].shift(periods)).astype("Int8")


# ---------------------------------------------------------------------------
# Price features
# ---------------------------------------------------------------------------

def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Intra-bar and inter-bar price features.

    Returns a DataFrame with columns:
    * ``high_low_range``   -- ``high - low``
    * ``close_open_range`` -- ``close - open``
    * ``gap``              -- ``open_t - close_{t-1}``
    """
    return pd.DataFrame({
        "high_low_range": df["high"] - df["low"],
        "close_open_range": df["close"] - df["open"],
        "gap": df["open"] - df["close"].shift(1),
    })
