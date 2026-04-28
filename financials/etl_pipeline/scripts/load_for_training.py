"""Programmatic example: load features from the dataset for ML training.

Two flavours covered:
  * XGBoost / sklearn — flat pandas DataFrame across many symbols/dates.
  * LSTM (PyTorch)    — per-symbol sliding windows of fixed length.
"""

from __future__ import annotations

import pandas as pd

from financials.etl_pipeline import MLDataLoader

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
    """Sliding 60-bar windows per symbol, target = next-bar log return."""
    reader = MLDataLoader(DATASET_PATH)

    feature_columns = [
        "log_returns", "rsi", "macd_macd", "macd_signal", "macd_histogram",
        "bollinger_bands_upper", "bollinger_bands_middle", "bollinger_bands_lower",
        "atr", "volume_features_volume_ratio",
    ]

    dataset = reader.to_torch_dataset(
        sequence_length=60,
        feature_columns=feature_columns,
        target_column="log_returns",
        symbols=["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-06-30",
        stride=1,
    )

    # The returned object is a torch.utils.data.Dataset
    print(f"Dataset size: {len(dataset)}")
    x, y = dataset[0]
    print(f"First sample: X.shape={tuple(x.shape)}, y={float(y):.6f}")
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
