"""Momentum trading strategy.

This module implements a momentum (relative-strength) strategy that
ranks assets by their rolling return over a configurable lookback window
and selects the strongest performers:

* **Buy** when the rolling return over the *lookback* period is in the
  top *top_pct* percentile of all observed returns for that asset --
  strong positive momentum.
* **Sell** when the rolling return turns **negative** -- momentum has
  reversed.
* **Hold** at all other times.

Rolling returns are computed as the simple percentage change over
*lookback* bars using :meth:`pandas.Series.pct_change`.
"""

import logging

import pandas as pd

from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class Momentum(BaseStrategy):
    """Rolling-return momentum strategy.

    The strategy is designed for single-asset signal generation (i.e.
    one DataFrame per ticker).  Cross-sectional ranking across multiple
    assets should be handled by a higher-level portfolio module.

    Args:
        lookback: Number of bars over which to compute the rolling
            return.  Defaults to ``20``.
        top_pct: Fraction (0, 1) representing the percentile threshold
            above which a buy signal is generated.  For example, ``0.2``
            means only returns in the top 20 % of the asset's own
            historical distribution trigger a buy.  Defaults to ``0.2``.

    Raises:
        ValueError: If *lookback* < 1 or *top_pct* is not in (0, 1).

    Example::

        strategy = Momentum(lookback=30, top_pct=0.15)
        signals = strategy.generate_signals(price_df)
    """

    name: str = "momentum"
    description: str = (
        "Buys when the rolling return over the lookback period is in the "
        "top percentile of historical returns, and sells when momentum "
        "turns negative."
    )

    def __init__(self, lookback: int = 20, top_pct: float = 0.2) -> None:
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}.")
        if not (0 < top_pct < 1):
            raise ValueError(
                f"top_pct must be between 0 and 1 (exclusive), got {top_pct}."
            )
        self.lookback = lookback
        self.top_pct = top_pct
        logger.info(
            "Initialised %s with lookback=%d, top_pct=%.2f",
            self.name,
            self.lookback,
            self.top_pct,
        )

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate momentum signals from the ``close`` column.

        Steps:

        1. Compute the rolling return as the percentage change of
           ``close`` over the *lookback* window.
        2. Determine the percentile threshold using an **expanding**
           quantile so the threshold adapts as more data becomes
           available.
        3. Emit **buy** (``1``) when the rolling return exceeds the
           threshold (top *top_pct* of all returns seen so far).
        4. Emit **sell** (``-1``) when the rolling return is negative.

        Args:
            df: DataFrame with at least a ``close`` column.

        Returns:
            A Series of ``1`` (buy), ``-1`` (sell), or ``0`` (hold),
            indexed identically to *df*.
        """
        self.validate_dataframe(df)
        close = df["close"]

        # Rolling return over the lookback window.
        rolling_return = close.pct_change(periods=self.lookback)

        # Expanding quantile: the threshold grows more stable as the
        # history lengthens.  We use (1 - top_pct) because quantile()
        # returns the value *below* which that fraction of data falls.
        threshold = rolling_return.expanding().quantile(1 - self.top_pct)

        signals = pd.Series(0, index=df.index, dtype=int)

        # Buy when momentum is in the top percentile.
        signals[rolling_return >= threshold] = 1

        # Sell when momentum is negative (overrides a buy if both
        # conditions were somehow met, which cannot happen in practice
        # because a negative return cannot exceed a positive threshold).
        signals[rolling_return < 0] = -1

        # Rows where rolling_return is NaN (insufficient history) stay 0.
        signals[rolling_return.isna()] = 0

        logger.debug(
            "%s generated %d buy and %d sell signals out of %d bars.",
            self.name,
            (signals == 1).sum(),
            (signals == -1).sum(),
            len(signals),
        )
        return signals

    def get_parameters(self) -> dict:
        """Return the momentum strategy parameters.

        Returns:
            ``{"lookback": <int>, "top_pct": <float>}``
        """
        return {
            "lookback": self.lookback,
            "top_pct": self.top_pct,
        }

    def get_dependencies(self) -> list[str]:
        """Declare that this strategy requires the ``close`` price.

        Returns:
            ``["close"]``
        """
        return ["close"]
