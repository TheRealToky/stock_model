"""Phase 0 - probe the on-disk parquet feature store schema & coverage."""
from __future__ import annotations

import pandas as pd

from financials.etl_pipeline.load.reader import MLDataLoader

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 80)

loader = MLDataLoader("data/feature_store")

dates = loader.available_dates()
syms = loader.available_symbols()
print(f"dates: {len(dates)}  [{dates[0]} .. {dates[-1]}]")
print(f"symbols: {len(syms)}")
print("sample symbols:", syms[:30])

# One symbol, small slice, to see schema + raw price sanity.
df = loader.load_pandas(symbols=["AAPL"], start="2021-01-04", end="2021-01-06")
print("\n--- AAPL 2021-01-04..06 ---")
print("shape:", df.shape)
print("columns:", list(df.columns))
print("dtypes:\n", df.dtypes)
print("\nhead:\n", df.head(3).to_string())
print("\ntail:\n", df.tail(3).to_string())

# Intraday continuity: gaps between consecutive timestamps within the slice.
ts = pd.to_datetime(df["timestamp"], utc=True).sort_values().reset_index(drop=True)
deltas = ts.diff().dropna().dt.total_seconds() / 60.0
print("\nminute-delta value_counts (top):")
print(deltas.round().astype(int).value_counts().head(10).to_string())

# Price sanity for OHLC presence.
for c in ("open", "high", "low", "close", "volume"):
    print(f"{c:8s} present={c in df.columns}")
print("\nclose describe:\n", df["close"].describe().to_string())
