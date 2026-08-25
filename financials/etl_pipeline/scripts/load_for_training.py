"""Programmatic example: load features from the dataset for ML training.

Two flavours covered:
  * XGBoost / sklearn — flat pandas DataFrame across many symbols/dates.
  * LSTM (PyTorch)    — per-symbol sliding windows of fixed length.

The LSTM example also shows the correct way to define a target: every column
in the feature store is backward-looking, so the label must be derived with a
forward shift rather than pulled from an existing column.
"""

from __future__ import annotations

import pandas as pd

from financials.etl_pipeline import MLDataLoader, StreamingSequenceDataset

DATASET_PATH = "data/feature_store"


# ---------------------------------------------------------------------------
# 1. Tabular load for XGBoost
# ---------------------------------------------------------------------------


def load_xgboost_frame() -> pd.DataFrame:
    """Cross-sectional load: many symbols × a date range, flat pandas."""
    reader = MLDataLoader(DATASET_PATH)

    df = reader.load_pandas(
        symbols=["AAPL", "MSFT", "NVDA"],
        start="2024-01-01",
        end="2024-06-30",
        columns=[
            "timestamp", "symbol",
            "log_returns", "rsi", "macd_macd", "macd_histogram",
            "bollinger_bands_upper", "bollinger_bands_lower",
            "atr", "volume_features_volume_ratio",
        ],
    )
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 2. PyTorch sequence dataset for LSTM
# ---------------------------------------------------------------------------


def build_lstm_dataset():
    """Streaming sliding windows per symbol, target = next-bar direction.

    Note the target. Every column in the feature store is **backward-looking**
    by construction, so none of them is a valid label:

      * ``log_returns`` is the return *into* the current bar.
      * ``price_direction`` is ``close_t > close_{t-1}`` -- contemporaneous,
        despite reading like a ready-made label.

    Windows end at bar ``i - 1`` and the label is read from that same bar, so
    handing either column in as ``target_column`` trains the model on its own
    answer. Derive the target with a forward shift instead; the reader raises
    if a target column also appears in ``feature_columns``.
    """
    from sklearn.preprocessing import StandardScaler

    reader = MLDataLoader(DATASET_PATH)

    feature_columns = [
        "log_returns", "rsi", "macd_macd", "macd_signal", "macd_histogram",
        "bollinger_bands_upper", "bollinger_bands_middle", "bollinger_bands_lower",
        "atr", "volume_features_volume_ratio",
    ]

    # Fit the scaler on TRAIN rows only, one symbol at a time to bound memory.
    scaler = StandardScaler()
    for sym in ("AAPL", "MSFT"):
        df_sym = reader.load_pandas(
            symbols=[sym], start="2024-01-01", end="2024-06-30",
            columns=["timestamp", "symbol", *feature_columns],
        )
        scaler.partial_fit(df_sym[feature_columns].to_numpy(dtype="float32"))
        del df_sym

    dataset = StreamingSequenceDataset(
        loader=reader,
        scaler=scaler,
        feature_columns=feature_columns,
        # Forward-shifted: the label describes the bar AFTER the window ends.
        target_fn=lambda df: (df["close"].shift(-1) > df["close"]).astype("float32"),
        # `close` is read by the target only -- it is not fed to the model.
        target_source_columns=["close"],
        symbols=["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-06-30",
        sequence_length=60,
        stride=1,
        shuffle_buffer=10_000,  # 0 for validation/test -- preserves time order
        seed=42,
    )

    x, y = next(iter(dataset))
    print(f"First sample: X.shape={tuple(x.shape)}, y={float(y):.0f}")
    return dataset


def main() -> None:
    print("=== XGBoost frame ===")
    df = load_xgboost_frame()
    print(df.head())

    print("\n=== LSTM dataset ===")
    try:
        build_lstm_dataset()
    except ImportError:
        print("PyTorch not installed -- skipping LSTM example")


if __name__ == "__main__":
    main()
