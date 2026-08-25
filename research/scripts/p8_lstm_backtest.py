"""Phase 2 - what is the shipped LSTM actually worth, net of costs?

``models/saved/lstm_direction_v2.pt`` posts test accuracy 0.533 and ROC-AUC
0.536, and those are the only numbers anyone has ever had for it. It has never
been backtested. Classification metrics cannot answer the only question that
matters -- a 53%-accurate next-bar classifier is a catastrophic strategy if it
trades every minute and a fine one if it trades once a week, and accuracy reads
the same either way.

This runs the checkpoint through
:func:`~financials.backtesting.ml_adapter.cost_sweep` and reports Sharpe
against buy-and-hold at all six cost points from ``research.config.COST_GRID``,
from frictionless to the lab's retail-tier 30 bp round trip.

Method notes:

* Inference reproduces the training pipeline exactly -- same 30 feature
  columns in the same order, the same fitted scaler, the same 390-bar window
  and stride 15. Reproducing the *ordering* matters: a StandardScaler applied
  to permuted columns silently produces garbage rather than an error.
* Predictions land at stride 15, so the model re-decides every 15 minutes and
  holds in between. That is how it was trained and evaluated, and turnover
  costs are charged on the resulting position changes only.
* The probability for a window ending at bar ``i-1`` is stamped at bar
  ``i-1``'s timestamp; the adapter's ``shift=1`` then trades it at bar ``i``.
  No signal is ever acted on before it could have existed.
* The window here (2025-01-01 onward) is the LSTM's *own* held-out test set
  from ``notebooks/06_lstm_training_2.ipynb``. It overlaps the research sealed
  window, but that seal was for the vol-target/XGBoost study; this checkpoint
  was trained through 2024-12-31 and has never seen these bars.

Usage::

    python -m research.scripts.p8_lstm_backtest
    python -m research.scripts.p8_lstm_backtest --symbols AAPL,MSFT,NVDA
    python -m research.scripts.p8_lstm_backtest --checkpoint models/saved/lstm_direction_v1.pt
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from financials.backtesting.ml_adapter import DEFAULT_COST_GRID, backtest_predictions
from financials.etl_pipeline.load.reader import MLDataLoader

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)

OUT_DIR = Path("research/outputs")
DATASET = "data/feature_store"

# Exactly the ordering the scaler was fitted on. Do not reorder.
FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "returns", "log_returns",
    "sma", "ema", "wma", "hma",
    "rsi",
    "macd_macd", "macd_signal", "macd_histogram",
    "bollinger_bands_upper", "bollinger_bands_middle", "bollinger_bands_lower",
    "atr", "volatility",
    "rolling_stats_rolling_mean", "rolling_stats_rolling_std",
    "rolling_stats_rolling_skew", "rolling_stats_rolling_kurtosis",
    "volume_features_volume_sma", "volume_features_volume_ratio",
    "volume_features_obv",
    "price_features_high_low_range", "price_features_close_open_range",
    "price_features_gap",
]

SEQUENCE_LENGTH = 390
STRIDE = 15
TEST_START = "2025-01-01"
TEST_END = "2026-03-31"


def predict_symbol(
    model,
    scaler,
    loader: MLDataLoader,
    symbol: str,
    start: str,
    end: str,
    batch_size: int = 512,
) -> tuple[pd.DataFrame, pd.Series] | tuple[None, None]:
    """Run the checkpoint over one symbol and return bars plus probabilities.

    Args:
        model: A loaded :class:`~models.lstm_model.LSTMModel`.
        scaler: The fitted scaler saved alongside it.
        loader: Feature-store reader.
        symbol: Ticker.
        start: Inclusive start date.
        end: Inclusive end date.
        batch_size: Windows per forward pass.

    Returns:
        ``(bars, proba)`` aligned on the same index, or ``(None, None)`` when
        the symbol has too little data to form a single window.
    """
    import torch

    cols = list(dict.fromkeys(["timestamp", "symbol", *FEATURE_COLUMNS]))
    df = (
        loader.load_pandas(symbols=[symbol], start=start, end=end, columns=cols)
        .dropna(subset=FEATURE_COLUMNS)
        .reset_index(drop=True)
    )
    if len(df) <= SEQUENCE_LENGTH + 1:
        return None, None

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")

    feats = scaler.transform(df[FEATURE_COLUMNS].to_numpy(np.float32)).astype(np.float32)

    # Window ends at bar i-1; that is the bar the prediction describes.
    ends = np.arange(SEQUENCE_LENGTH, len(df), STRIDE) - 1
    if not len(ends):
        return None, None

    model._net.eval()
    probs = np.empty(len(ends), dtype=np.float64)
    with torch.no_grad():
        for b0 in range(0, len(ends), batch_size):
            chunk = ends[b0 : b0 + batch_size]
            batch = np.stack([feats[e + 1 - SEQUENCE_LENGTH : e + 1] for e in chunk])
            logits = model._net(torch.from_numpy(batch))
            probs[b0 : b0 + len(chunk)] = torch.sigmoid(logits).numpy()

    proba = pd.Series(np.nan, index=df.index, dtype=np.float64)
    proba.iloc[ends] = probs
    # Hold the last decision until the next one, so the model re-decides every
    # STRIDE bars rather than flickering to flat in between.
    proba = proba.ffill()

    bars = df[["open", "high", "low", "close", "volume"]].astype(np.float64)
    return bars, proba


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="models/saved/lstm_direction_v2.pt")
    ap.add_argument("--scaler", default="models/saved/lstm_direction_v2_scaler.joblib")
    ap.add_argument("--symbols", default="AAPL,MSFT,NVDA,AMZN,META")
    ap.add_argument("--start", default=TEST_START)
    ap.add_argument("--end", default=TEST_END)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    from models.lstm_model import LSTMModel

    model = LSTMModel.load(args.checkpoint)
    scaler = joblib.load(args.scaler)
    loader = MLDataLoader(DATASET)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    print("=" * 108)
    print(f"LSTM net-of-cost backtest  checkpoint={args.checkpoint}")
    print(f"window={args.start}..{args.end}  symbols={','.join(symbols)}  "
          f"seq_len={SEQUENCE_LENGTH}  stride={STRIDE}  threshold={args.threshold}")
    print("=" * 108)

    all_rows: list[dict] = []
    for symbol in symbols:
        t0 = time.time()
        bars, proba = predict_symbol(model, scaler, loader, symbol, args.start, args.end)
        if bars is None:
            print(f"\n[{symbol}] insufficient data -- skipped")
            continue

        decided = int(proba.notna().sum())
        print(f"\n[{symbol}] bars={len(bars):,}  decisions={decided // STRIDE:,}  "
              f"mean P(up)={np.nanmean(proba):.4f}  ({time.time() - t0:.1f}s)")

        for commission, slippage, cost_label, rt_bps in DEFAULT_COST_GRID:
            result = backtest_predictions(
                bars, proba.fillna(0.5),
                interval="1min",
                threshold=args.threshold,
                commission=commission, slippage=slippage,
                label=f"{symbol}/{cost_label}",
                meta={"symbol": symbol, "cost_label": cost_label,
                      "round_trip_bps": rt_bps},
            )
            row = result.row()
            all_rows.append(row)

        sub = pd.DataFrame([r for r in all_rows if r["symbol"] == symbol])
        for _, r in sub.iterrows():
            print(f"    {r['cost_label']:<17} ({r['round_trip_bps']:>4.1f} bp)  "
                  f"strat_sharpe={r['strat_sharpe_ratio']:>9.3f}   "
                  f"bh={r['bh_sharpe_ratio']:>6.3f}   "
                  f"edge={r['sharpe_edge']:>9.3f}   "
                  f"maxDD={r['strat_max_drawdown']:>6.2%}   "
                  f"trades/day={r['strat_trades_per_day']:>5.2f}   "
                  f"beats_bh={r['beats_benchmark']}")

    if not all_rows:
        print("\nNo symbol produced enough data.")
        return 1

    table = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "p8_lstm_backtest.csv"
    table.to_csv(csv_path, index=False)

    print("\n" + "=" * 108)
    print("SUMMARY - mean across symbols, by cost point")
    print("=" * 108)
    summary = (
        table.groupby(["cost_label", "round_trip_bps"], as_index=False)
        .agg(
            strat_sharpe=("strat_sharpe_ratio", "mean"),
            bh_sharpe=("bh_sharpe_ratio", "mean"),
            sharpe_edge=("sharpe_edge", "mean"),
            max_drawdown=("strat_max_drawdown", "mean"),
            trades_per_day=("strat_trades_per_day", "mean"),
            exposure=("strat_exposure", "mean"),
            beats=("beats_benchmark", "sum"),
        )
        .sort_values("round_trip_bps")
    )
    print(summary.to_string(index=False))

    n = len(table)
    print(f"\nConfigurations beating buy & hold: {int(table['beats_benchmark'].sum())} / {n}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
