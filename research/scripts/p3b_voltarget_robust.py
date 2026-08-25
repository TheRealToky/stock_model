"""Phase 3 (robustness) - locked vol-target overlay on TRAIN+VAL only.

Two robustness lenses, both out-of-sample-in-spirit (no per-fold tuning; the
rule self-calibrates via an expanding quantile):
  A. Per-year stability on the primary ticker (does it help every year, esp.
     the 2022 bear?).
  B. Cross-ticker consistency across the liquid universe (same frozen params).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from financials.etl_pipeline.load.reader import MLDataLoader
from research.lib import reslib, overlays
from research import config

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
loader = MLDataLoader("data/feature_store")
fmt = lambda v: f"{v:.3f}"

COST = (0.0001, 0.0001)       # realistic 2 bps round-trip
LAB = (0.001, 0.0005)         # pessimistic 30 bps round-trip

print("=" * 100)
print(f"LOCKED OVERLAY: span={overlays.SPAN} q={overlays.TARGET_Q} "
      f"min_days={overlays.MIN_DAYS}")
print("=" * 100)

# ---------- A. Per-year stability on AAPL ----------
print("\n### A. Per-year stability (AAPL, TRAIN+VAL) ###")
df = reslib.load_ohlcv(loader, config.PRIMARY, config.TRAINVAL_START, config.TRAINVAL_END)
e = overlays.voltarget_exposure(df)
for (comm, slip, name) in [(*COST, "2bp"), (*LAB, "30bp")]:
    sim = reslib.simulate_exposure(df, e, commission=comm, slippage=slip)
    bh = reslib.simulate_exposure(df, pd.Series(1.0, index=df.index), commission=comm, slippage=slip)
    sret, bret = sim["ret"], bh["ret"]
    expo = sim["pos"]
    years = sret.index.year
    rows = []
    for y in sorted(np.unique(years)):
        m = years == y
        s = reslib.summarize_returns(sret[m].to_numpy())
        b = reslib.summarize_returns(bret[m].to_numpy())
        rows.append({
            "year": int(y),
            "strat_sharpe": s["sharpe"], "bh_sharpe": b["sharpe"],
            "edge": s["sharpe"] - b["sharpe"],
            "strat_maxDD": s["max_drawdown"], "bh_maxDD": b["max_drawdown"],
            "strat_totret": s["total_return"], "bh_totret": b["total_return"],
            "avg_expo": float(expo[m].mean()),
        })
    yt = pd.DataFrame(rows)
    print(f"\n cost={name}:")
    print(yt.to_string(index=False, float_format=fmt))
    yt.to_csv(f"research/outputs/p3b_peryear_{name}.csv", index=False)

# ---------- B. Cross-ticker consistency ----------
print("\n\n### B. Cross-ticker (TRAIN+VAL, 2bp round-trip) ###")
rows = []
for tk in config.UNIVERSE:
    dft = reslib.load_ohlcv(loader, tk, config.TRAINVAL_START, config.TRAINVAL_END)
    et = overlays.voltarget_exposure(dft)
    ev = reslib.evaluate_exposure(dft, et, commission=COST[0], slippage=COST[1], label=tk)
    r = ev.row()
    rows.append({
        "ticker": tk,
        "strat_sharpe": r["strat_sharpe"], "bh_sharpe": r["bh_sharpe"],
        "edge": r["sharpe_edge"],
        "strat_maxDD": r["strat_max_drawdown"], "bh_maxDD": r["bh_max_drawdown"],
        "dd_ok": r["dd_ok"],
        "strat_totret": r["strat_total_return"], "bh_totret": r["bh_total_return"],
        "avg_expo": r["strat_exposure"], "beats_bh": r["beats_bh"],
    })
ct = pd.DataFrame(rows)
ct.to_csv("research/outputs/p3b_crossticker.csv", index=False)
print(ct.to_string(index=False, float_format=fmt))
print("\nmedians:")
print(ct[["edge", "strat_maxDD", "bh_maxDD", "strat_totret", "bh_totret", "avg_expo"]]
      .median().to_string(float_format=fmt))
print(f"\nbeats_bh (Sharpe up AND maxDD<=B&H): {int(ct['beats_bh'].sum())}/{len(ct)}")
print(f"maxDD <= B&H: {int(ct['dd_ok'].sum())}/{len(ct)}")
print(f"Sharpe edge > 0: {int((ct['edge'] > 0).sum())}/{len(ct)}")
