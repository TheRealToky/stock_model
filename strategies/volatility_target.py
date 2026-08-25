"""Volatility-targeting overlay strategy (1-minute research deliverable).

Grounded in the EDA finding that at 1-minute frequency *direction* is
essentially unpredictable (return autocorrelation ~ 0, variance ratio ~ 1) but
*volatility* is highly predictable (slowly-decaying |return| autocorrelation,
strong intraday seasonality). The overlay therefore does NOT try to time
direction; it sizes a long-only exposure inversely to predicted volatility,
de-risking in turbulent regimes and holding fully otherwise:

    pred_vol_d = EMA_span(realized_vol_from_1min_returns)   # known at prior close
    target_d   = expanding q-quantile of pred_vol            # past-only, adaptive
    exposure_d = clip(target_d / pred_vol_d, 0, 1)           # unlevered, daily

Canonical evaluation uses the continuous-exposure simulator
(``research.lib.reslib.simulate_exposure``) -- a documented sizing extension of
the long/flat ``BacktestEngine``. ``generate_signals`` below exposes a long/flat
PROXY (long unless the overlay would de-risk below 50%) so the class still
satisfies the :class:`BaseStrategy` contract and can be registered/inspected
through the standard tooling; it is not the canonical sizing logic.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class VolatilityTargetOverlay(BaseStrategy):
    name: str = "vol_target_overlay"
    description: str = (
        "Unlevered volatility-targeting overlay on 1-min bars: exposure = "
        "clip(target/pred_vol, 0, 1), where pred_vol is an EMA of daily realized "
        "vol from 1-min returns and target is its expanding quantile. De-risks in "
        "high-vol regimes; reduces drawdown vs buy & hold with negligible turnover."
    )

    def __init__(self, span: int = 22, target_q: float = 0.50, min_days: int = 60) -> None:
        self.span = span
        self.target_q = target_q
        self.min_days = min_days

    # ------------------------------------------------------------------
    def _exposure(self, df: pd.DataFrame) -> pd.Series:
        day = pd.Series(df.index.normalize(), index=df.index)
        r = df["close"].pct_change()
        r[(day != day.shift(1)).values] = np.nan
        rv = r.groupby(day).apply(lambda s: np.sqrt(np.nansum(s.to_numpy() ** 2)))
        rv.index = pd.to_datetime(rv.index)
        rv = rv.sort_index()
        pred = rv.ewm(span=self.span).mean().shift(1)
        target = pred.expanding(min_periods=self.min_days).quantile(self.target_q)
        e_daily = pd.Series(1.0, index=rv.index)
        valid = pred.notna() & target.notna()
        e_daily[valid] = (target[valid] / pred[valid]).clip(0.0, 1.0)
        return pd.Series(df.index.normalize().map(e_daily).to_numpy(float), index=df.index).fillna(1.0)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Long/flat proxy: flat (-1 to exit) when the overlay would de-risk
        below 50% exposure, long (1) otherwise. See class docstring -- the
        canonical strategy is continuous exposure, not this binary proxy."""
        self.validate_dataframe(df)
        want_long = (self._exposure(df) >= 0.5).astype(int)
        change = want_long.diff().fillna(want_long.iloc[0])
        signals = pd.Series(0, index=df.index, dtype=int)
        signals[change > 0] = 1
        signals[change < 0] = -1
        return signals

    def get_parameters(self) -> dict:
        return {
            "span": self.span,
            "target_q": self.target_q,
            "min_days": self.min_days,
            "type": "continuous_exposure_overlay",
        }

    def get_dependencies(self) -> list[str]:
        return ["open", "close"]


class BuyAndHold(BaseStrategy):
    name: str = "buy_and_hold"
    description: str = "Buy on the first bar and hold to the end (benchmark)."

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index, dtype=int)
        if len(df):
            signals.iloc[0] = 1
        return signals

    def get_parameters(self) -> dict:
        return {}

    def get_dependencies(self) -> list[str]:
        return ["close"]
