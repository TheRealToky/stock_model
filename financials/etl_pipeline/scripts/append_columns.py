# scripts/append_columns.py  (run once)
import pathlib
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from financials.features.technical import compute_wma, compute_hma, compute_price_direction

STORE = pathlib.Path("data/feature_store")

for parquet_file in STORE.rglob("*.parquet"):
    table = pq.read_table(parquet_file)
    df = table.to_pandas()

    # Skip if columns already present
    if "wma" in df.columns:
        continue

    df["wma"] = compute_wma(df, window=20).values
    df["hma"] = compute_hma(df, window=20).values
    df["price_direction"] = compute_price_direction(df, periods=1).values

    pq.write_table(
        pa.Table.from_pandas(df, schema=table.schema.append(
            pa.field("wma", pa.float32())
        ).append(
            pa.field("hma", pa.float32())
        ).append(
            pa.field("price_direction", pa.int8())
        )),
        parquet_file,
        compression="zstd",
        compression_level=3,
    )