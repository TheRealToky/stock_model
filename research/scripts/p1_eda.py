"""Phase 1 - EDA on 1-minute bars (TRAIN+VAL window only; test stays sealed).

Questions:
  1. Return autocorrelation: is there short-horizon reversal (negative lag-1)
     or momentum (positive) in within-day 1-min returns?
  2. Intraday seasonality: U-shaped volatility/volume across the session?
  3. Volatility clustering: slow ACF decay of |returns|?
  4. Variance ratio: VR(q) < 1 => mean reversion, > 1 => trending.

Overnight returns (first bar of each day vs prior close) are EXCLUDED from the
1-min return series so we measure genuine intraday microstructure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from financials.etl_pipeline.load.reader import MLDataLoader
from research.lib import reslib
from research import config

pd.set_option("display.width", 200)
loader = MLDataLoader("data/feature_store")


def within_day_returns(df: pd.DataFrame) -> pd.Series:
    """1-min simple returns, NaN at each day's first bar (no overnight)."""
    day = pd.Series(df.index.normalize(), index=df.index)
    r = df["close"].pct_change()
    first_of_day = day != day.shift(1)
    r[first_of_day] = np.nan
    return r


def autocorr(x: np.ndarray, lags: int) -> list[float]:
    x = x[~np.isnan(x)]
    x = x - x.mean()
    denom = np.sum(x * x)
    out = []
    for k in range(1, lags + 1):
        out.append(float(np.sum(x[k:] * x[:-k]) / denom) if denom > 0 else 0.0)
    return out


def variance_ratio(r: np.ndarray, q: int) -> float:
    """Variance ratio VR(q) = Var(q-period sum) / (q * Var(1-period)).

    VR(q) ~= 1 under a random walk; <1 indicates mean reversion, >1 trending.
    Uses overlapping q-sums; both variances are sample variances so the q=2
    case reduces to ~1 + rho_1 as expected.
    """
    r = r[~np.isnan(r)]
    n = len(r)
    if n < q * 2:
        return float("nan")
    var1 = np.var(r, ddof=1)
    cs = np.cumsum(r)
    rq = cs[q:] - cs[:-q]          # overlapping sums of q consecutive returns
    varq = np.var(rq, ddof=1)
    return float(varq / (q * var1)) if var1 > 0 else float("nan")


def overnight_intraday(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day intraday (open->close), overnight (prev close->open), and
    full close-to-close returns. Daily frequency."""
    day = pd.Series(df.index.normalize(), index=df.index)
    g = df.groupby(day)
    day_open = g["open"].first()
    day_close = g["close"].last()
    intraday = (day_close / day_open - 1.0)
    overnight = (day_open / day_close.shift(1) - 1.0)
    full = (day_close / day_close.shift(1) - 1.0)
    return pd.DataFrame({"intraday": intraday, "overnight": overnight, "full": full}).dropna()


def ann_sharpe_daily(x: pd.Series, rf: float = 0.04, periods: int = 252) -> float:
    x = x.dropna().to_numpy()
    if len(x) < 2:
        return 0.0
    ex = x - rf / periods
    sd = np.std(ex, ddof=1)
    return float(np.mean(ex) / sd * np.sqrt(periods)) if sd > 0 else 0.0


print("=" * 78)
print("1. WITHIN-DAY 1-MIN RETURN AUTOCORRELATION (TRAIN+VAL)")
print("=" * 78)
rows = []
oni_rows = []
ac_primary = None
for tk in config.UNIVERSE:
    df = reslib.load_ohlcv(loader, tk, config.TRAINVAL_START, config.TRAINVAL_END)
    r = within_day_returns(df).to_numpy()
    ac = autocorr(r, 10)
    if tk == config.PRIMARY:
        ac_primary = ac
    rows.append({
        "ticker": tk, "bars": len(df),
        "lag1": ac[0], "lag2": ac[1], "lag3": ac[2],
        "lag5": ac[4], "lag10": ac[9],
        "mean_abs_ret_bps": float(np.nanmean(np.abs(r)) * 1e4),
    })
    # overnight vs intraday decomposition (daily frequency)
    oni = overnight_intraday(df)
    oni_rows.append({
        "ticker": tk,
        "intraday_ann_ret%": oni["intraday"].mean() * 252 * 100,
        "overnight_ann_ret%": oni["overnight"].mean() * 252 * 100,
        "full_ann_ret%": oni["full"].mean() * 252 * 100,
        "intraday_sharpe": ann_sharpe_daily(oni["intraday"]),
        "overnight_sharpe": ann_sharpe_daily(oni["overnight"]),
        "full_sharpe": ann_sharpe_daily(oni["full"]),
    })
ac_tbl = pd.DataFrame(rows)
print(ac_tbl.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
print(f"\nMean lag-1 autocorr across universe: {ac_tbl['lag1'].mean():.5f}")
ac_tbl.to_csv("research/outputs/p1_autocorr.csv", index=False)

print("\n" + "=" * 78)
print("1b. OVERNIGHT vs INTRADAY DECOMPOSITION (daily, TRAIN+VAL, gross)")
print("=" * 78)
oni_tbl = pd.DataFrame(oni_rows)
print(oni_tbl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nMeans across universe:")
print(oni_tbl.drop(columns="ticker").mean().to_string(float_format=lambda v: f"{v:.3f}"))
oni_tbl.to_csv("research/outputs/p1_overnight_intraday.csv", index=False)

print("\n" + "=" * 78)
print("2. VARIANCE RATIOS (AAPL, within-day returns)")
print("=" * 78)
df = reslib.load_ohlcv(loader, config.PRIMARY, config.TRAINVAL_START, config.TRAINVAL_END)
r = within_day_returns(df).to_numpy()
for q in (2, 5, 10, 30, 60):
    print(f"  VR({q:3d}) = {variance_ratio(r, q):.4f}   (<1 mean-revert, >1 trend)")

print("\n" + "=" * 78)
print("3. INTRADAY SEASONALITY (AAPL): by minutes-since-open")
print("=" * 78)
# minutes since session open per day, robust to DST
day = df.index.normalize()
minute_idx = df.groupby(day).cumcount()
df2 = pd.DataFrame({
    "m": minute_idx.values,
    "ret": df["close"].pct_change().values,
    "absret": df["close"].pct_change().abs().values,
    "vol": df["volume"].values,
})
prof = df2.groupby("m").agg(
    mean_ret_bps=("ret", lambda s: np.nanmean(s) * 1e4),
    std_ret_bps=("absret", lambda s: np.nanmean(s) * 1e4),
    mean_vol=("vol", "mean"),
).reset_index()
# show open/close extremes
print("First 5 minutes of session:")
print(prof.head(5).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print("Last 5 minutes of session:")
print(prof.tail(5).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print("Midday (minutes 190-194):")
print(prof.iloc[190:195].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
prof.to_csv("research/outputs/p1_intraday_profile.csv", index=False)

print("\n" + "=" * 78)
print("4. VOLATILITY CLUSTERING (AAPL): ACF of |returns| vs returns")
print("=" * 78)
ac_ret = autocorr(r, 60)
ac_abs = autocorr(np.abs(r), 60)
print(f"  ret  ACF lag1={ac_ret[0]:.4f} lag5={ac_ret[4]:.4f} lag30={ac_ret[29]:.4f}")
print(f"  |ret| ACF lag1={ac_abs[0]:.4f} lag5={ac_abs[4]:.4f} lag30={ac_abs[29]:.4f} lag60={ac_abs[59]:.4f}")

# ---- plots ----
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes[0, 0].bar(range(1, 11), ac_primary, color="steelblue")
axes[0, 0].axhline(0, color="k", lw=0.8)
axes[0, 0].set_title(f"{config.PRIMARY} 1-min within-day return ACF (lags 1-10)")
axes[0, 0].set_xlabel("lag (minutes)"); axes[0, 0].set_ylabel("autocorr")

axes[0, 1].plot(prof["m"], prof["std_ret_bps"], color="darkred")
axes[0, 1].set_title(f"{config.PRIMARY} mean |1-min return| by minutes-since-open")
axes[0, 1].set_xlabel("minutes since open"); axes[0, 1].set_ylabel("bps")

axes[1, 0].plot(prof["m"], prof["mean_vol"], color="darkgreen")
axes[1, 0].set_title(f"{config.PRIMARY} mean volume by minutes-since-open")
axes[1, 0].set_xlabel("minutes since open"); axes[1, 0].set_ylabel("shares")

axes[1, 1].plot(range(1, 61), ac_abs, label="|ret| ACF", color="purple")
axes[1, 1].plot(range(1, 61), ac_ret, label="ret ACF", color="gray")
axes[1, 1].axhline(0, color="k", lw=0.8); axes[1, 1].legend()
axes[1, 1].set_title(f"{config.PRIMARY} ACF: returns vs |returns| (lags 1-60)")
axes[1, 1].set_xlabel("lag (minutes)")

plt.tight_layout()
plt.savefig("research/outputs/p1_eda.png", dpi=110)
print("\nsaved research/outputs/p1_eda.png")
