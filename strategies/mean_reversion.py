"""Mean Reversion trading strategy.

This module implements a mean-reversion strategy built on Bollinger
Band-like logic.  The core premise is that prices tend to oscillate
around a rolling mean, and extreme deviations from that mean are likely
to reverse:

* **Buy** when the price falls *entry_std* standard deviations **below**
  the rolling mean -- the asset is considered oversold.
* **Sell** when the price returns to within *exit_std* standard
  deviations of the rolling mean -- the reversion has played out.
* **Hold** at all other times.

The rolling statistics (mean and standard deviation) are computed over a
configurable *lookback* window using :meth:`pandas.Series.rolling`.
"""

import logging

import pandas as pd

from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class MeanReversion(BaseStrategy):
    """Bollinger Band-style mean reversion strategy.

    The strategy maintains an internal ``in_position`` flag that tracks
    whether the previous signal opened a long position.  This prevents
    duplicate buy signals while already holding and ensures a sell signal
    is only emitted once the price reverts toward the mean.

    Args:
        lookback: Number of bars for the rolling window used to compute
            the mean and standard deviation.  Defaults to ``20``.
        entry_std: Number of standard deviations *below* the rolling
            mean at which a buy signal is generated.  Defaults to ``2.0``.
        exit_std: Number of standard deviations from the rolling mean
            within which a sell (exit) signal is generated.  Defaults to
            ``0.5``.

    Raises:
        ValueError: If *lookback* < 2, *entry_std* <= 0, *exit_std* < 0,
            or *exit_std* >= *entry_std*.

    Example::

        strategy = MeanReversion(lookback=30, entry_std=2.5, exit_std=0.5)
        signals = strategy.generate_signals(price_df)
    """

    name: str = "mean_reversion"
    description: str = (
        "Buys when the price drops a configurable number of standard "
        "deviations below its rolling mean and sells when it reverts "
        "back toward the mean (Bollinger Band-like logic)."
    )

    def __init__(
        self,
        lookback: int = 20,
        entry_std: float = 2.0,
        exit_std: float = 0.5,
    ) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2 to compute a standard deviation.")
        if entry_std <= 0:
            raise ValueError(f"entry_std must be > 0, got {entry_std}.")
        if exit_std < 0:
            raise ValueError(f"exit_std must be >= 0, got {exit_std}.")
        if exit_std >= entry_std:
            raise ValueError(
                f"exit_std ({exit_std}) must be strictly less than "
                f"entry_std ({entry_std})."
            )
        self.lookback = lookback
        self.entry_std = entry_std
        self.exit_std = exit_std
        logger.info(
            "Initialised %s with lookback=%d, entry_std=%.2f, exit_std=%.2f",
            self.name,
            self.lookback,
            self.entry_std,
            self.exit_std,
        )

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate mean-reversion signals from the ``close`` column.

        The method calculates a rolling mean and rolling standard
        deviation over the *lookback* window.  It then walks the price
        series and:

        1. Emits a **buy** (``1``) when the close drops below
           ``rolling_mean - entry_std * rolling_std`` and the strategy
           is not already in a position.
        2. Emits a **sell** (``-1``) when the close rises back above
           ``rolling_mean - exit_std * rolling_std`` and the strategy
           is currently in a position.

        Args:
            df: DataFrame with at least a ``close`` column.

        Returns:
            A Series of ``1`` (buy), ``-1`` (sell), or ``0`` (hold),
            indexed identically to *df*.
        """
        self.validate_dataframe(df)
        close = df["close"]

        rolling_mean = close.rolling(window=self.lookback).mean()
        rolling_std = close.rolling(window=self.lookback).std()

        # Define the entry (lower) and exit bands.
        lower_band = rolling_mean - self.entry_std * rolling_std
        exit_band = rolling_mean - self.exit_std * rolling_std

        # Walk the series to produce stateful signals.  We need to track
        # whether we are "in a position" to avoid repeated buy signals
        # and to know when to sell.
        signals = pd.Series(0, index=df.index, dtype=int)
        in_position = False

        for i in range(self.lookback, len(close)):
            if pd.isna(rolling_mean.iloc[i]):
                continue

            price = close.iloc[i]

            if not in_position and price < lower_band.iloc[i]:
                # Price is extremely low relative to recent history -> buy.
                signals.iloc[i] = 1
                in_position = True
            elif in_position and price > exit_band.iloc[i]:
                # Price has reverted toward the mean -> sell / exit.
                signals.iloc[i] = -1
                in_position = False

        logger.debug(
            "%s generated %d buy and %d sell signals out of %d bars.",
            self.name,
            (signals == 1).sum(),
            (signals == -1).sum(),
            len(signals),
        )
        return signals

    def get_parameters(self) -> dict:
        """Return the mean-reversion parameters.

        Returns:
            ``{"lookback": <int>, "entry_std": <float>, "exit_std": <float>}``
        """
        return {
            "lookback": self.lookback,
            "entry_std": self.entry_std,
            "exit_std": self.exit_std,
        }

    def get_dependencies(self) -> list[str]:
        """Declare that this strategy requires the ``close`` price.

        Returns:
            ``["close"]``
        """
        return ["close"]
