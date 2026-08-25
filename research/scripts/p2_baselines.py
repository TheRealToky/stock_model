"""Phase 2 - Establish the bar to beat: existing & naive strategies at 1-min,
net of costs, vs buy & hold, across a cost-sensitivity grid (TRAIN+VAL only).

Goal: quantify how 1-min turnover interacts with costs. Signals are computed
once per strategy and reused across every cost level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from financials.etl_pipeline.load.reader import MLDataLoader
from strategies import EMACrossover, MeanReversion
from research.lib import reslib
from research import config

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
loader = MLDataLoader("data/feature_store")

TK = config.PRIMARY
df = reslib.load_ohlcv(loader, TK, config.TRAINVAL_START, config.TRAINVAL_END)
print(f"{TK} TRAIN+VAL {df.index[0]} .. {df.index[-1]}  bars={len(df):,}")

# ---- desired-position helpers (decided at bar i, executed next bar) ----
def within_day_ret(df: pd.DataFrame) -> pd.Series:
    day = pd.Series(df.index.normalize(), index=df.index)
    r = df["close"].pct_change()
    r[(day != day.shift(1)).values] = 0.0
    return r

r1 = within_day_ret(df)
pos_reversal = (r1 < 0).astype(float)     # bet on bounce after a down minute
pos_momentum = (r1 > 0).astype(float)     # chase an up minute
# SMA(5) vs SMA(30) long/flat trend overlay
sma_fast = df["close"].rolling(5).mean()
sma_slow = df["close"].rolling(30).mean()
pos_smatrend = (sma_fast > sma_slow).astype(float)

# ---- BaseStrategy action signals (computed once) ----
ema_action = EMACrossover(12, 26).generate_signals(df)
mr_action = MeanReversion(20, 2.0, 0.5).generate_signals(df)

strategies = {
    "ema_crossover(12,26)": ("action", ema_action),
    "mean_reversion(20,2,.5)": ("action", mr_action),
    "naive_reversal_1bar": ("pos", pos_reversal),
    "naive_momentum_1bar": ("pos", pos_momentum),
    "sma_trend(5,30)": ("pos", pos_smatrend),
}

rows = []
for name, (kind, sig) in strategies.items():
    for comm, slip, clabel, rt in config.COST_GRID:
        if kind == "action":
            ev = reslib.evaluate(df, sig, commission=comm, slippage=slip,
                                 label=name, meta={"cost": clabel, "rt_bps": rt})
        else:
            ev = reslib.evaluate_position(df, sig, commission=comm, slippage=slip,
                                          label=name, meta={"cost": clabel, "rt_bps": rt})
        r = ev.row()
        rows.append({
            "strategy": name, "cost": clabel, "rt_bps": rt,
            "strat_sharpe": r["strat_sharpe"], "bh_sharpe": r["bh_sharpe"],
            "edge": r["sharpe_edge"],
            "strat_maxDD": r["strat_max_drawdown"], "bh_maxDD": r["bh_max_drawdown"],
            "strat_totret": r["strat_total_return"], "bh_totret": r["bh_total_return"],
            "n_trades": r["strat_n_trades"], "trades/day": r["strat_trades_per_day"],
            "exposure": r["strat_exposure"], "beats_bh": r["beats_bh"],
        })

res = pd.DataFrame(rows)
res.to_csv("research/outputs/p2_baselines.csv", index=False)

fmt = lambda v: f"{v:.3f}"
print("\n" + "=" * 120)
print(f"BASELINES vs BUY&HOLD across cost grid  ({TK}, TRAIN+VAL, Sharpe annualized @ {reslib.PERIODS_1MIN})")
print("B&H Sharpe (this window):", round(res['bh_sharpe'].iloc[0], 3),
      " B&H maxDD:", round(res['bh_maxDD'].iloc[0], 3),
      " B&H total return:", round(res['bh_totret'].iloc[0], 3))
print("=" * 120)
show = res[["strategy", "cost", "rt_bps", "strat_sharpe", "edge", "strat_maxDD",
            "strat_totret", "n_trades", "trades/day", "exposure", "beats_bh"]]
print(show.to_string(index=False, float_format=fmt))

print("\n--- Frictionless (gross) vs lab-default summary ---")
piv = res.pivot_table(index="strategy", columns="cost", values="strat_sharpe")
cols = [c for c in ["frictionless", "low_2bp", "mid_5bp", "lab_default_15bp"] if c in piv.columns]
print(piv[cols].to_string(float_format=fmt))
print(f"\nAny config beating B&H (higher Sharpe AND maxDD<=B&H)? "
      f"{res['beats_bh'].any()}  ({int(res['beats_bh'].sum())}/{len(res)} configs)")
