"""Phase 1 - multi-horizon harness: does a coarser bar rescue the edge?

The 1-minute study concluded there is no reliable edge, and pinned the blame
mostly on friction: at ~11 trades/day a 30 bp round trip costs more than the
signal is worth. That leaves an obvious question it could not answer, because
everything was hard-wired to 1-minute bars -- *is the signal absent, or just
uneconomic at that frequency?*

This script answers it by running the identical pipeline at every horizon:

    resample -> recompute features -> triple-barrier labels -> uniqueness
    weights -> purged CV -> probability->position -> net-of-cost metrics
    -> DSR / PBO

Everything that could flatter a result is held constant across horizons, so the
comparison is about frequency and nothing else. In particular:

* The **sealed test window is never touched.** All of this runs inside
  ``config.TRAINVAL_*``; ``config.TEST_*`` stays untouched for a final pass.
* Cross-validation is **purged and embargoed**, so a label still open when a
  fold begins cannot sit in that fold's training set.
* Sharpe is annualised with each interval's **own** periods-per-year. Using
  252 everywhere -- the bug this codebase shipped until recently -- would
  inflate the 1-minute row by ~19.7x and make the sweep meaningless.
* The reported Deflated Sharpe **discounts for the whole search**, counting
  every (horizon x threshold) pair tried, not just the winner.

Usage::

    python -m research.scripts.p7_multihorizon
    python -m research.scripts.p7_multihorizon --ticker MSFT --start 2023-01-01
    python -m research.scripts.p7_multihorizon --intervals 15min,1h,1d
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from financials.etl_pipeline.load.reader import MLDataLoader
from research import config
from research.lib import labeling, resample, reslib, validation, weights

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

OUT_DIR = Path("research/outputs")
DATASET = "data/feature_store"

# Backward-looking features that exist in the store at every interval once
# recomputed. Deliberately excludes price_direction (contemporaneous) and any
# raw price level (non-stationary).
FEATURES = [
    "returns", "log_returns", "rsi", "macd_macd", "macd_signal", "macd_histogram",
    "volatility", "atr", "rolling_stats_rolling_std", "rolling_stats_rolling_skew",
    "rolling_stats_rolling_kurtosis", "volume_features_volume_ratio",
    "price_features_high_low_range", "price_features_close_open_range",
]

# Probability thresholds for the position adapter. Every one counts as a trial
# against the Deflated Sharpe.
THRESHOLDS = (0.50, 0.52, 0.55, 0.58)


def build_model(seed: int):
    """Gradient-boosted classifier, falling back to sklearn if xgboost is absent."""
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=4,
            random_state=seed, eval_metric="logloss",
        ), "XGBClassifier"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=200, random_state=seed
        ), "HistGradientBoosting"


def prepare_horizon(
    loader: MLDataLoader,
    ticker: str,
    spec: dict,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """Load, resample, label and weight one horizon.

    Args:
        loader: Feature-store reader.
        ticker: Symbol.
        spec: One entry of ``config.HORIZONS``.
        start: Inclusive start date.
        end: Inclusive end date.

    Returns:
        ``(bars, dataset)`` where *bars* is the resampled OHLCV+features frame
        and *dataset* is the model-ready subset with label and weight columns.
        ``(None, None)`` when there is not enough data.
    """
    interval, horizon = spec["interval"], spec["bars"]

    bars = resample.load_resampled(
        loader, ticker, interval, start=start, end=end, with_features=True
    )
    if bars.empty or len(bars) < 500:
        return None, None

    # Barriers scale with local volatility so a label means the same thing in
    # calm and turbulent regimes -- and, crucially, across intervals.
    vol = labeling.ewm_volatility(bars["close"], span=config.TB_VOL_SPAN)
    tb = labeling.triple_barrier_labels(
        bars,
        horizon_bars=horizon,
        pt=config.TB_PROFIT_TAKE,
        sl=config.TB_STOP_LOSS,
        target=vol,
        high_col="high",
        low_col="low",
        stop_at_session_end=(interval != "1d"),
    )

    data = bars.copy()
    data["label"] = (tb.label > 0).astype(float).where(tb.label.notna())
    # Carry the barrier touch as a TIMESTAMP, not a position. Positions here
    # index the full resampled frame, and the model-ready subset below drops
    # warm-up rows -- so a position would silently point at the wrong bar (or
    # off the end) once filtered.
    data["touch_time"] = tb.touch_time
    data["holding"] = tb.holding_bars

    # Overlapping labels are not independent observations; weight them down.
    data["weight"] = weights.combined_weights(tb.touch_idx, bars["close"])
    data["uniqueness"] = weights.average_uniqueness(tb.touch_idx)

    ready = data.dropna(subset=[*FEATURES, "label", "weight", "touch_time"]).copy()
    return bars, ready


def oos_predictions(
    dataset: pd.DataFrame, seed: int
) -> tuple[np.ndarray, float, int]:
    """Purged-CV out-of-sample probabilities across the whole window.

    Args:
        dataset: Model-ready frame with features, label, weight, touch_idx.
        seed: RNG seed.

    Returns:
        ``(proba, auc, n_folds)``. ``proba`` is NaN where a row never appeared
        in a test fold.
    """
    from sklearn.metrics import roc_auc_score

    X = dataset[FEATURES].to_numpy(np.float32)
    y = dataset["label"].to_numpy(np.float64)
    w = dataset["weight"].to_numpy(np.float64)

    # Map each label's barrier-touch timestamp onto a position *within this
    # filtered dataset*, so the purge measures the right spans. searchsorted
    # gives the first row at or after the touch, which is the conservative
    # choice: it never understates how long a label stays open.
    n = len(dataset)
    own = np.arange(n)
    touch = dataset["touch_time"].to_numpy("datetime64[ns]")
    t1 = np.searchsorted(dataset.index.to_numpy("datetime64[ns]"), touch, side="left")
    t1 = np.where(np.isnat(touch), own, t1)
    t1 = np.clip(t1, own, n - 1)

    cv = validation.PurgedKFold(
        n_splits=config.CV_N_SPLITS, touch_idx=t1, embargo_pct=config.CV_EMBARGO_PCT
    )
    proba = np.full(len(dataset), np.nan)
    folds = 0
    for train, test in cv.split(X):
        if len(np.unique(y[train])) < 2:
            continue
        model, _ = build_model(seed)
        model.fit(X[train], y[train], sample_weight=w[train])
        proba[test] = model.predict_proba(X[test])[:, 1]
        folds += 1

    ok = ~np.isnan(proba)
    scorable = bool(ok.sum()) and len(np.unique(y[ok])) > 1
    auc = float(roc_auc_score(y[ok], proba[ok])) if scorable else float("nan")
    return proba, auc, folds


def evaluate_thresholds(
    bars: pd.DataFrame,
    dataset: pd.DataFrame,
    proba: np.ndarray,
    spec: dict,
    costs: tuple[float, float],
) -> tuple[list[dict], pd.DataFrame]:
    """Turn probabilities into positions and score them net of costs.

    Returns:
        ``(rows, returns)`` -- one metric row per threshold, plus the matrix of
        per-bar net returns (bars x thresholds) that PBO consumes.
    """
    periods = spec["periods_per_year"]
    commission, slippage = costs
    rows = []
    returns: dict[str, pd.Series] = {}

    ok = ~np.isnan(proba)
    sub = dataset.loc[ok]
    px = bars.loc[sub.index]

    for thr in THRESHOLDS:
        desired = pd.Series((proba[ok] > thr).astype(float), index=sub.index)
        ev = reslib.evaluate_position(
            px, desired,
            commission=commission, slippage=slippage,
            periods=periods,
            label=f"{spec['interval']}@{thr:.2f}",
            meta={"interval": spec["interval"], "threshold": thr},
        )
        row = ev.row()
        row["exposure"] = float(desired.mean())
        rows.append(row)
        returns[f"thr_{thr:.2f}"] = ev.strat_sim["ret"]

    return rows, pd.DataFrame(returns)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default=config.PRIMARY)
    ap.add_argument("--start", default=config.TRAINVAL_START)
    ap.add_argument("--end", default=config.TRAINVAL_END)
    ap.add_argument("--intervals", default="", help="comma-separated subset")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    if args.end > config.TRAINVAL_END:
        print(
            f"REFUSING: --end {args.end} reaches past TRAINVAL_END "
            f"({config.TRAINVAL_END}) into the sealed test window.",
            file=sys.stderr,
        )
        return 2

    specs = config.HORIZONS
    if args.intervals:
        wanted = {s.strip() for s in args.intervals.split(",")}
        specs = [s for s in specs if s["interval"] in wanted]

    loader = MLDataLoader(DATASET)
    _, model_name = build_model(args.seed)

    print("=" * 100)
    print(f"Multi-horizon sweep  ticker={args.ticker}  window={args.start}..{args.end}")
    print(f"model={model_name}  seed={args.seed}  "
          f"barriers=+/-{config.TB_PROFIT_TAKE}sigma  "
          f"cv={config.CV_N_SPLITS}-fold purged, embargo {config.CV_EMBARGO_PCT:.0%}")
    print("=" * 100)

    all_rows: list[dict] = []
    diagnostics: list[dict] = []

    for spec in specs:
        t0 = time.time()
        bars, dataset = prepare_horizon(loader, args.ticker, spec, args.start, args.end)
        if dataset is None or len(dataset) < 500:
            print(f"\n[{spec['interval']}] insufficient data -- skipped")
            continue

        proba, auc, folds = oos_predictions(dataset, args.seed)
        eff_n = weights.effective_sample_size(dataset["uniqueness"])

        print(f"\n[{spec['interval']}]  bars={len(bars):,}  labelled={len(dataset):,}  "
              f"effective_N={eff_n:,.0f} ({eff_n / max(len(dataset), 1):.1%} of raw)")
        print(f"  label balance={dataset['label'].mean():.3f}  "
              f"median hold={dataset['holding'].median():.0f} bars  "
              f"purged-CV AUC={auc:.4f}  folds={folds}  ({time.time() - t0:.1f}s)")

        rows, ret_matrix = evaluate_thresholds(
            bars, dataset, proba, spec, config.DEFAULT_COST
        )

        # PBO across this horizon's thresholds: if picking the best threshold
        # in-sample lands below median out-of-sample, the selection is noise.
        try:
            pbo = validation.probability_of_backtest_overfitting(
                ret_matrix.dropna(), n_splits=8
            )
        except ValueError:
            pbo = {"pbo": float("nan"), "mean_logit": float("nan")}

        for r in rows:
            r["auc"] = auc
            r["effective_n"] = eff_n
            r["n_labelled"] = len(dataset)
            r["pbo"] = pbo["pbo"]
        all_rows.extend(rows)

        best = max(rows, key=lambda r: r["strat_sharpe"])
        print(f"  best threshold={best['threshold']:.2f}  "
              f"strat_sharpe={best['strat_sharpe']:+.3f}  "
              f"bh_sharpe={best['bh_sharpe']:+.3f}  "
              f"edge={best['sharpe_edge']:+.3f}  "
              f"trades/day={best.get('strat_trades_per_day', float('nan')):.2f}  "
              f"beats_bh={best['beats_bh']}")
        print(f"  PBO across thresholds={pbo['pbo']:.3f}  "
              f"(0.5 = threshold choice carries no information)")

        diagnostics.append({
            "interval": spec["interval"], "auc": auc, "effective_n": eff_n,
            "n_labelled": len(dataset), "best_sharpe": best["strat_sharpe"],
            "best_edge": best["sharpe_edge"], "pbo": pbo["pbo"],
        })

    if not all_rows:
        print("\nNo horizon produced enough data.")
        return 1

    table = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "p7_multihorizon.csv"
    table.to_csv(csv_path, index=False)

    # --- honest accounting for the search --------------------------------
    n_trials = len(all_rows)
    sharpes = table["strat_sharpe"].to_numpy(np.float64)
    best_row = table.iloc[int(np.argmax(sharpes))]
    per_bar = best_row["strat_sharpe"] / np.sqrt(
        next(s["periods_per_year"] for s in specs if s["interval"] == best_row["interval"])
    )
    dsr = validation.deflated_sharpe_ratio(
        per_bar,
        int(best_row["n_labelled"]),
        n_trials=n_trials,
        sharpe_variance=float(np.var(sharpes / np.sqrt(98_280), ddof=1)) if n_trials > 1 else 0.0,
    )

    print("\n" + "=" * 100)
    print("SUMMARY (sealed test window untouched)")
    print("=" * 100)
    cols = ["label", "strat_sharpe", "bh_sharpe", "sharpe_edge", "strat_max_drawdown",
            "strat_trades_per_day", "exposure", "auc", "pbo", "beats_bh"]
    print(table[[c for c in cols if c in table.columns]].to_string(index=False))

    print(f"\nBest configuration : {best_row['label']}  sharpe={best_row['strat_sharpe']:+.3f}")
    print(f"Trials searched    : {n_trials} (horizon x threshold)")
    print(f"Deflated Sharpe    : {dsr:.4f}   "
          f"({'survives' if dsr > 0.95 else 'DOES NOT survive'} the multiple-testing discount)")
    print(f"Any config beating buy&hold: {int(table['beats_bh'].sum())} / {n_trials}")
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
