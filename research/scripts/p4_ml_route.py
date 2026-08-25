"""Phase 3 (ML route) - direction classifier + probability->signal adapter.

Tests whether a gradient-boosted classifier on standard 1-min features can
predict next-bar direction well enough to beat buy & hold net of costs. The
EDA (random-walk returns, AUC~0.5 expected) predicts it cannot; this quantifies
it honestly and demonstrates the ML->signal adapter the brief calls out.

Split (temporal, no leakage):
  train 2021-01..2023-12  |  val 2024-01..2025-04 (threshold tuned here)  |
  test 2025-05..2026-04 (sealed; touched once).

Adapter: long (exposure 1) when P(up) > threshold else flat; threshold chosen to
maximize *validation* net-of-cost Sharpe, then applied unchanged to test.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from financials.etl_pipeline.load.reader import MLDataLoader
from research.lib import reslib
from research import config

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
loader = MLDataLoader("data/feature_store")
SEED = config.SEED

FEATURE_COLS = [
    "returns", "log_returns", "rsi", "macd_macd", "macd_histogram",
    "volatility", "atr", "rolling_stats_rolling_std", "rolling_stats_rolling_skew",
    "rolling_stats_rolling_kurtosis", "volume_features_volume_ratio",
    "price_features_high_low_range", "price_features_close_open_range",
]

df = reslib.load_ohlcv(loader, config.PRIMARY, config.TRAINVAL_START, config.TEST_END,
                       columns=FEATURE_COLS)
print(f"{config.PRIMARY} loaded {len(df):,} bars with features")

# bollinger position + a few return lags (backward-looking only)
for lag in (1, 2, 3, 5):
    df[f"ret_lag{lag}"] = df["close"].pct_change().shift(lag)
feat = FEATURE_COLS + [f"ret_lag{l}" for l in (1, 2, 3, 5)]

# target: next-bar up (within-day; drop the day's last bar where next bar is overnight)
day = pd.Series(df.index.normalize(), index=df.index)
fwd = df["close"].shift(-1) / df["close"] - 1.0
fwd[(day != day.shift(-1)).values] = np.nan        # no cross-day label
df["y"] = (fwd > 0).astype("float")

data = df.dropna(subset=feat + ["y"]).copy()

def split(d, a, b):
    s = pd.Timestamp(a, tz="UTC"); e = pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1)
    return d[(d.index >= s) & (d.index < e)]

tr = split(data, "2021-01-04", "2023-12-31")
va = split(data, "2024-01-01", "2025-04-30")
te = split(data, config.TEST_START, config.TEST_END)
print(f"train={len(tr):,}  val={len(va):,}  test={len(te):,}")

try:
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=4,
                          random_state=SEED, eval_metric="logloss")
    mname = "XGBClassifier"
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                            max_iter=300, random_state=SEED)
    mname = "HistGradientBoosting"

model.fit(tr[feat].to_numpy(), tr["y"].to_numpy())
print(f"\nmodel: {mname}")
for nm, d in [("train", tr), ("val", va), ("test", te)]:
    p = model.predict_proba(d[feat].to_numpy())[:, 1]
    print(f"  AUC {nm:5s} = {roc_auc_score(d['y'], p):.4f}   "
          f"P(up) mean={p.mean():.4f} std={p.std():.4f}")

# ---- adapter: tune threshold on validation net-of-cost Sharpe ----
va_p = model.predict_proba(va[feat].to_numpy())[:, 1]
te_p = model.predict_proba(te[feat].to_numpy())[:, 1]
COMM, SLIP = 0.0001, 0.0001

best = None
for thr in np.round(np.arange(0.45, 0.60, 0.01), 2):
    pos = pd.Series((va_p > thr).astype(float), index=va.index)
    ev = reslib.evaluate_position(va, pos, commission=COMM, slippage=SLIP)
    sh = ev.strat["sharpe"]
    if best is None or sh > best[1]:
        best = (thr, sh, ev.bh["sharpe"], ev.strat["trades_per_day"])
thr = best[0]
print(f"\nbest val threshold={thr}  val_strat_sharpe={best[1]:.3f}  "
      f"val_bh_sharpe={best[2]:.3f}  val_trades/day={best[3]:.1f}")

# ---- apply to TEST (sealed) ----
print("\n" + "=" * 90)
print("ML DIRECTION CLASSIFIER on SEALED TEST (net of costs), threshold fixed from val")
print("=" * 90)
for comm, slip, clabel, rt in [(0.0, 0.0, "frictionless", 0.0),
                               (0.0001, 0.0001, "low_2bp", 4.0),
                               (0.001, 0.0005, "lab_15bp", 30.0)]:
    pos = pd.Series((te_p > thr).astype(float), index=te.index)
    ev = reslib.evaluate_position(te, pos, commission=comm, slippage=slip)
    r = ev.row()
    print(f"  cost={clabel:12s} strat_sharpe={r['strat_sharpe']:7.3f}  "
          f"bh_sharpe={r['bh_sharpe']:6.3f}  edge={r['sharpe_edge']:7.3f}  "
          f"strat_DD={r['strat_max_drawdown']:.3f} bh_DD={r['bh_max_drawdown']:.3f}  "
          f"trades/day={r['strat_trades_per_day']:.1f}  beats={r['beats_bh']}")
print("\nDONE p4_ml")
