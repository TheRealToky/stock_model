"""Locked strategy logic for the volatility-targeting overlay.

Kept in one place so the robustness checks and the final sealed-test run share
identical code. Parameters are frozen after the TRAIN+VAL exploration (see
research/scripts/p3_voltarget_explore.py): span=22, q=0.50 (target = adaptive
expanding median of predicted vol), min_days=60.

No look-ahead:
  * realized vol uses only within-day 1-min returns of the day itself;
  * predicted vol = EMA(realized vol).shift(1)  -> known at the prior close;
  * the target = expanding quantile of predicted vol -> uses only past days;
  * reslib then shifts the per-bar exposure by one more bar at execution.
During warmup (before min_days of history) exposure defaults to 1.0 (i.e. plain
buy & hold), so the overlay never benefits from being flat early.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Frozen hyperparameters (selected on TRAIN+VAL only).
SPAN = 22
TARGET_Q = 0.50
MIN_DAYS = 60


def realized_vol_daily(df: pd.DataFrame) -> pd.Series:
    """Daily realized volatility from within-day 1-min returns (no overnight)."""
    day = pd.Series(df.index.normalize(), index=df.index)
    r = df["close"].pct_change()
    r[(day != day.shift(1)).values] = np.nan
    rv = r.groupby(day).apply(lambda s: np.sqrt(np.nansum(s.to_numpy() ** 2)))
    rv.index = pd.to_datetime(rv.index)
    return rv.sort_index()


def voltarget_exposure(
    df: pd.DataFrame,
    span: int = SPAN,
    q: float = TARGET_Q,
    min_days: int = MIN_DAYS,
) -> pd.Series:
    """Per-bar target exposure in [0,1] from the daily vol-target rule.

    e_d = clip(target_d / pred_vol_d, 0, 1), constant within day d. Warmup days
    (insufficient history to form the expanding quantile) default to 1.0.
    """
    rv = realized_vol_daily(df)
    pred = rv.ewm(span=span).mean().shift(1)                 # info through prior day
    target = pred.expanding(min_periods=min_days).quantile(q)  # past-only, adaptive

    e_daily = pd.Series(1.0, index=rv.index)                 # warmup = buy & hold
    valid = pred.notna() & target.notna()
    e_daily[valid] = (target[valid] / pred[valid]).clip(0.0, 1.0)

    day = df.index.normalize()
    e_bar = pd.Series(day.map(e_daily).to_numpy(dtype=float), index=df.index)
    return e_bar.fillna(1.0)
