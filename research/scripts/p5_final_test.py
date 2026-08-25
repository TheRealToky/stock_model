"""Phase 5 - ONE final pass on the sealed test window (2025-05-01..2026-04-29).

The vol-target overlay is warmed on the full train+val history (its expanding
quantile carries over), then metrics are computed ONLY on the test slice -- the
realistic "we were already running it" evaluation, with no spurious entry cost
at the test boundary. Hyperparameters were frozen in research/lib/overlays.py
before this script was ever run.

Outputs: in-sample vs out-of-sample comparison tables (strategy vs buy & hold),
a cost-sensitivity sweep on the test window, and equity/drawdown plots.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from financials.etl_pipeline.load.reader import MLDataLoader
from research.lib import reslib, overlays
from research import config

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 80)
loader = MLDataLoader("data/feature_store")
fmt = lambda v: f"{v:.3f}"

PERIODS = reslib.PERIODS_1MIN
RF = reslib.RF_DEFAULT


def drawdown(equity: np.ndarray) -> np.ndarray:
    run_max = np.maximum.accumulate(equity)
    return (equity - run_max) / run_max


def slice_metrics(ret: np.ndarray, asset_ret: np.ndarray, expo: np.ndarray,
                  index: pd.DatetimeIndex) -> dict:
    s = reslib.summarize_returns(ret, PERIODS, RF)
    n = len(ret)
    years = n / PERIODS if PERIODS else 0.0
    eq = np.cumprod(1.0 + ret)
    cagr = (eq[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    annvol = float(np.std(ret, ddof=1) * np.sqrt(PERIODS)) if n > 1 else 0.0
    a, b = reslib.alpha_beta(ret, asset_ret, PERIODS)
    e_prev = np.r_[expo[0], expo[:-1]]
    n_days = max(pd.Series(index).dt.normalize().nunique(), 1)
    turn_day = float(np.abs(expo - e_prev).sum()) / n_days
    return {
        "sharpe": s["sharpe"], "sortino": s["sortino"], "maxDD": s["max_drawdown"],
        "cagr": cagr, "totret": s["total_return"], "annvol": annvol,
        "alpha": a, "beta": b, "exposure": float(expo.mean()), "turn/day": turn_day,
    }


_CACHE: dict[str, tuple] = {}


def _load(tk: str):
    """Load full history + exposure once per ticker (cost-independent)."""
    if tk not in _CACHE:
        df = reslib.load_ohlcv(loader, tk, config.TRAINVAL_START, config.TEST_END)
        e = overlays.voltarget_exposure(df)
        _CACHE[tk] = (df, e)
    return _CACHE[tk]


def run_ticker(tk: str, comm: float, slip: float):
    """Simulate overlay + B&H over FULL history; return (IS dict, OOS dict, series)."""
    df, e = _load(tk)
    sim = reslib.simulate_exposure(df, e, commission=comm, slippage=slip)
    bh = reslib.simulate_exposure(df, pd.Series(1.0, index=df.index), commission=comm, slippage=slip)
    asset_ret = df["close"].pct_change().fillna(0.0).to_numpy()

    def in_range(start, end):
        s = pd.Timestamp(start, tz="UTC")
        e_ = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)   # inclusive end day
        return (df.index >= s) & (df.index < e_)

    is_mask = in_range(config.TRAINVAL_START, config.TRAINVAL_END)
    oos_mask = in_range(config.TEST_START, config.TEST_END)

    def block(mask):
        sret = sim["ret"].to_numpy()[mask]
        bret = bh["ret"].to_numpy()[mask]
        ar = asset_ret[mask]
        se = sim["pos"].to_numpy()[mask]
        be = bh["pos"].to_numpy()[mask]
        idx = df.index[mask]
        return slice_metrics(sret, ar, se, idx), slice_metrics(bret, ar, be, idx)

    is_s, is_b = block(is_mask)
    oos_s, oos_b = block(oos_mask)
    series = {
        "index": df.index[oos_mask],
        "strat_eq": np.cumprod(1.0 + sim["ret"].to_numpy()[oos_mask]),
        "bh_eq": np.cumprod(1.0 + bh["ret"].to_numpy()[oos_mask]),
    }
    return is_s, is_b, oos_s, oos_b, series


# ============================================================================
# 1. Per-ticker IS vs OOS at realistic 2bp round-trip
# ============================================================================
COMM, SLIP = 0.0001, 0.0001
rows = []
aapl_series = None
for tk in config.UNIVERSE:
    is_s, is_b, oos_s, oos_b, series = run_ticker(tk, COMM, SLIP)
    if tk == config.PRIMARY:
        aapl_series = series
    rows.append({
        "ticker": tk,
        "IS_strat_sh": is_s["sharpe"], "IS_bh_sh": is_b["sharpe"], "IS_edge": is_s["sharpe"] - is_b["sharpe"],
        "OOS_strat_sh": oos_s["sharpe"], "OOS_bh_sh": oos_b["sharpe"], "OOS_edge": oos_s["sharpe"] - oos_b["sharpe"],
        "OOS_strat_DD": oos_s["maxDD"], "OOS_bh_DD": oos_b["maxDD"],
        "OOS_strat_ret": oos_s["totret"], "OOS_bh_ret": oos_b["totret"],
        "OOS_expo": oos_s["exposure"],
        "OOS_dd_ok": oos_s["maxDD"] <= oos_b["maxDD"] + 1e-9,
        "OOS_beats": (oos_s["sharpe"] > oos_b["sharpe"]) and (oos_s["maxDD"] <= oos_b["maxDD"] + 1e-9),
    })
tbl = pd.DataFrame(rows)
tbl.to_csv("research/outputs/p5_is_vs_oos.csv", index=False)
print("=" * 130)
print(f"IN-SAMPLE vs OUT-OF-SAMPLE  (vol-target overlay vs B&H, 2bp round-trip, Sharpe@{PERIODS})")
print("=" * 130)
print(tbl.to_string(index=False, float_format=fmt))
print("\nOOS summary:")
print(f"  maxDD <= B&H:            {int(tbl['OOS_dd_ok'].sum())}/{len(tbl)}")
print(f"  Sharpe edge > 0:         {int((tbl['OOS_edge'] > 0).sum())}/{len(tbl)}")
print(f"  beats B&H (Sharpe+DD):   {int(tbl['OOS_beats'].sum())}/{len(tbl)}")
print(f"  median OOS Sharpe edge:  {tbl['OOS_edge'].median():.3f}")
print(f"  median OOS DD reduction: {(tbl['OOS_bh_DD'] - tbl['OOS_strat_DD']).median():.3f}")

# ============================================================================
# 2. Full detailed metric table for the PRIMARY ticker (AAPL), IS & OOS
# ============================================================================
is_s, is_b, oos_s, oos_b, _ = run_ticker(config.PRIMARY, COMM, SLIP)
detail = pd.DataFrame({
    "IS_strat": is_s, "IS_bh": is_b, "OOS_strat": oos_s, "OOS_bh": oos_b,
}).T
detail.to_csv("research/outputs/p5_aapl_detail.csv")
print("\n" + "=" * 130)
print(f"PRIMARY {config.PRIMARY}: full metric detail (2bp round-trip)")
print("=" * 130)
print(detail.to_string(float_format=fmt))

# ============================================================================
# 3. Cost-sensitivity sweep on the TEST window (AAPL + universe median edge)
# ============================================================================
print("\n" + "=" * 130)
print("COST SENSITIVITY (OOS test window): overlay turnover is tiny => near cost-invariant")
print("=" * 130)
sweep_rows = []
for comm, slip, clabel, rt in config.COST_GRID:
    edges, dds = [], []
    aapl_edge = None
    for tk in config.UNIVERSE:
        _, _, oos_s, oos_b, _ = run_ticker(tk, comm, slip)
        edge = oos_s["sharpe"] - oos_b["sharpe"]
        edges.append(edge)
        dds.append(oos_b["maxDD"] - oos_s["maxDD"])
        if tk == config.PRIMARY:
            aapl_edge = edge
    sweep_rows.append({
        "cost": clabel, "rt_bps": rt,
        "AAPL_OOS_edge": aapl_edge,
        "univ_median_edge": float(np.median(edges)),
        "univ_median_DDreduction": float(np.median(dds)),
        "n_pos_edge": int(np.sum(np.array(edges) > 0)),
    })
sweep = pd.DataFrame(sweep_rows)
sweep.to_csv("research/outputs/p5_cost_sweep_oos.csv", index=False)
print(sweep.to_string(index=False, float_format=fmt))

# ============================================================================
# 4. Plots: AAPL OOS equity + drawdown (strat vs B&H)
# ============================================================================
idx = aapl_series["index"]
seq, beq = aapl_series["strat_eq"], aapl_series["bh_eq"]
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
axes[0].plot(idx, beq, label="Buy & Hold", color="gray", lw=1.0)
axes[0].plot(idx, seq, label="Vol-target overlay", color="steelblue", lw=1.0)
axes[0].set_title(f"{config.PRIMARY} OUT-OF-SAMPLE equity (net of 2bp round-trip), test 2025-05..2026-04")
axes[0].set_ylabel("growth of $1"); axes[0].legend()
axes[1].fill_between(idx, drawdown(beq) * 100, 0, color="gray", alpha=0.5, label="Buy & Hold")
axes[1].fill_between(idx, drawdown(seq) * 100, 0, color="steelblue", alpha=0.5, label="Overlay")
axes[1].set_title(f"{config.PRIMARY} OUT-OF-SAMPLE drawdown (%)")
axes[1].set_ylabel("drawdown %"); axes[1].legend()
plt.tight_layout()
plt.savefig("research/outputs/p5_aapl_oos.png", dpi=110)
print("\nsaved research/outputs/p5_aapl_oos.png")

# cross-ticker OOS bars
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].bar(tbl["ticker"], tbl["OOS_edge"], color=["seagreen" if v > 0 else "indianred" for v in tbl["OOS_edge"]])
ax[0].axhline(0, color="k", lw=0.8); ax[0].set_title("OOS Sharpe edge (overlay - B&H)")
ax[1].bar(tbl["ticker"], (tbl["OOS_bh_DD"] - tbl["OOS_strat_DD"]) * 100, color="steelblue")
ax[1].axhline(0, color="k", lw=0.8); ax[1].set_title("OOS maxDD reduction (B&H - overlay), %")
plt.tight_layout()
plt.savefig("research/outputs/p5_crossticker_oos.png", dpi=110)
print("saved research/outputs/p5_crossticker_oos.png")

print("\nDONE p5")
