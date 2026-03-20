"""EMA Crossover trading strategy.

This module implements a classic dual-EMA (Exponential Moving Average)
crossover strategy.  The strategy tracks two EMAs -- one *fast* (shorter
lookback) and one *slow* (longer lookback) -- and generates signals when
they cross:

* **Buy** when the fast EMA crosses **above** the slow EMA (bullish
  crossover), indicating upward momentum.
* **Sell** when the fast EMA crosses **below** the slow EMA (bearish
  crossover), indicating downward momentum.
* **Hold** at all other times.

The EMA is calculated using :meth:`pandas.Series.ewm` with ``span``
equal to the chosen period, which applies exponential weighting so that
recent prices have more influence than older ones.
"""

import logging

import pandas as pd

from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class EMACrossover(BaseStrategy):
    """Dual Exponential Moving Average crossover strategy.

    Args:
        fast_period: Span (in bars) for the fast EMA.  A smaller value
            makes the EMA more responsive to recent price changes.
            Defaults to ``12``.
        slow_period: Span (in bars) for the slow EMA.  A larger value
            smooths out noise but reacts more slowly.  Defaults to ``26``.

    Raises:
        ValueError: If *fast_period* >= *slow_period* or either period
            is less than 1.

    Example::

        strategy = EMA_Crossover(fast_period=10, slow_period=30)
        signals = strategy.generate_signals(price_df)
    """

    name: str = "ema_crossover"
    description: str = (
        "Generates buy/sell signals when a fast EMA crosses above/below "
        "a slow EMA, capturing trend reversals via momentum shifts."
    )

    def __init__(self, fast_period: int = 12, slow_period: int = 26) -> None:
        if fast_period < 1 or slow_period < 1:
            raise ValueError("EMA periods must be >= 1.")
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be strictly less than "
                f"slow_period ({slow_period})."
            )
        self.fast_period = fast_period
        self.slow_period = slow_period
        logger.info(
            "Initialised %s with fast_period=%d, slow_period=%d",
            self.name,
            self.fast_period,
            self.slow_period,
        )

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate crossover signals from the ``close`` price column.

        The method computes two EMAs from ``df["close"]`` and detects
        crossovers by comparing the sign of the difference between the
        fast and slow EMA on consecutive bars.

        Args:
            df: DataFrame with at least a ``close`` column.

        Returns:
            A Series of ``1`` (buy), ``-1`` (sell), or ``0`` (hold),
            indexed identically to *df*.
        """
        self.validate_dataframe(df)
        close = df["close"]

        # Compute EMAs using pandas exponential weighted functions.
        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        # Determine the position of the fast EMA relative to the slow EMA.
        # +1 when fast is above slow, -1 when below.
        position = (fast_ema > slow_ema).astype(int) - (fast_ema < slow_ema).astype(int)

        # A crossover occurs when the position changes between consecutive
        # bars.  diff() captures this:  +2 = bullish cross, -2 = bearish
        # cross, 0 = no change.
        crossover = position.diff()

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[crossover > 0] = 1   # fast crossed above slow -> buy
        signals[crossover < 0] = -1  # fast crossed below slow -> sell

        logger.debug(
            "%s generated %d buy and %d sell signals out of %d bars.",
            self.name,
            (signals == 1).sum(),
            (signals == -1).sum(),
            len(signals),
        )
        return signals

    def get_parameters(self) -> dict:
        """Return the EMA crossover parameters.

        Returns:
            ``{"fast_period": <int>, "slow_period": <int>}``
        """
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
        }

    def get_dependencies(self) -> list[str]:
        """Declare that this strategy requires the ``close`` price.

        Returns:
            ``["close"]``
        """
        return ["close"]
