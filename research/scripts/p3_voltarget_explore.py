"""Phase 3 (explore) - volatility-targeting overlay, in-sample on TRAIN+VAL.

Idea grounded in the EDA: 1-min *direction* is unpredictable but 1-min
*volatility* is highly predictable and clustered. Use realized vol from 1-min
returns to size a continuous, unlevered exposure e in [0,1], rebalanced daily:

    pred_vol_{d} = EMA_span(realized_vol) through day d-1   (no look-ahead)
    e_{d}        = clip(target_vol / pred_vol_{d}, 0, 1)

target_vol set as a quantile of the in-sample pred_vol distribution (a higher
quantile => de-risk only the most turbulent days => closer to buy & hold).

This is exploratory (decide whether the effect exists). Tuning/robustness with
walk-forward and a sealed test set follow in later scripts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from financials.etl_pipeline.load.reader import MLDataLoader
from research.lib import reslib
from research import config

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
loader = MLDataLoader("data/feature_store")


def realized_vol_daily(df: pd.DataFrame) -> pd.Series:
    """Daily realized vol from within-day 1-min returns (overnight excluded)."""
    day = pd.Series(df.index.normalize(), index=df.index)
    r = df["close"].pct_change()
    r[(day != day.shift(1)).values] = np.nan          # drop overnight jumps
    rv = r.groupby(day).apply(lambda s: np.sqrt(np.nansum(s.to_numpy() ** 2)))
    rv.index = pd.to_datetime(rv.index)
    return rv


def vol_target_exposure(df: pd.DataFrame, span: int, target: float) -> pd.Series:
    """Per-bar exposure from a daily vol-target rule (constant within a day)."""
    rv = realized_vol_daily(df)
    pred = rv.ewm(span=span).mean().shift(1)           # info through prior day
    e_daily = (target / pred).clip(0.0, 1.0)
    e_daily = e_daily.fillna(0.0)
    day = df.index.normalize()
    e_bar = pd.Series(day.map(e_daily).astype(float), index=df.index)
    return e_bar.fillna(0.0), pred


TK = config.PRIMARY
df = reslib.load_ohlcv(loader, TK, config.TRAINVAL_START, config.TRAINVAL_END)
print(f"{TK} TRAIN+VAL bars={len(df):,}  {df.index[0]} .. {df.index[-1]}")

# ---- validation: e==1 must reproduce buy & hold ----
ones = pd.Series(1.0, index=df.index)
ev_bh = reslib.evaluate_exposure(df, ones, commission=0.0001, slippage=0.0001)
print("\n[validate] exposure==1 vs internal B&H:")
print(f"  strat total_return={ev_bh.strat['total_return']:.4f}  "
      f"bh total_return={ev_bh.bh['total_return']:.4f}  "
      f"sharpe strat={ev_bh.strat['sharpe']:.4f} bh={ev_bh.bh['sharpe']:.4f}")

rv = realized_vol_daily(df)
pred_all = rv.ewm(span=10).mean().shift(1).dropna()
print(f"\npred daily vol quantiles: "
      f"p30={pred_all.quantile(.3):.5f} p50={pred_all.quantile(.5):.5f} "
      f"p70={pred_all.quantile(.7):.5f} p85={pred_all.quantile(.85):.5f}")

rows = []
for span in (5, 10, 22):
    rv_s = rv.ewm(span=span).mean().shift(1).dropna()
    for qlevel in (0.5, 0.7, 0.85, 0.95):
        target = float(rv_s.quantile(qlevel))
        exposure, _ = vol_target_exposure(df, span, target)
        for comm, slip, clabel, rt in [(0.0001, 0.0001, "low_2bp", 4.0),
                                       (0.001, 0.0005, "lab_15bp", 30.0)]:
            ev = reslib.evaluate_exposure(df, exposure, commission=comm, slippage=slip,
                                          label=f"vt_s{span}_q{qlevel}")
            r = ev.row()
            rows.append({
                "span": span, "target_q": qlevel, "cost": clabel,
                "strat_sharpe": r["strat_sharpe"], "bh_sharpe": r["bh_sharpe"],
                "edge": r["sharpe_edge"], "strat_maxDD": r["strat_max_drawdown"],
                "bh_maxDD": r["bh_max_drawdown"], "dd_ok": r["dd_ok"],
                "strat_totret": r["strat_total_return"], "bh_totret": r["bh_total_return"],
                "exposure": r["strat_exposure"], "turn/day": r["strat_turnover_per_day"],
                "beats_bh": r["beats_bh"],
            })

res = pd.DataFrame(rows)
res.to_csv("research/outputs/p3_voltarget_explore.csv", index=False)
fmt = lambda v: f"{v:.3f}"
print("\n" + "=" * 120)
print(f"VOL-TARGET OVERLAY (in-sample TRAIN+VAL, {TK}); B&H sharpe="
      f"{res['bh_sharpe'].iloc[0]:.3f} maxDD={res['bh_maxDD'].iloc[0]:.3f} "
      f"totret={res['bh_totret'].iloc[0]:.3f}")
print("=" * 120)
print(res.to_string(index=False, float_format=fmt))
print(f"\nConfigs beating B&H in-sample (Sharpe up AND maxDD<=B&H): "
      f"{int(res['beats_bh'].sum())}/{len(res)}")
print("\nBest by Sharpe edge (low_2bp):")
sub = res[res.cost == "low_2bp"].sort_values("edge", ascending=False).head(5)
print(sub.to_string(index=False, float_format=fmt))
