"""Parquet writer with Hive-style partitioning.

Layout
------
::

    <output_path>/
        date=2024-03-01/
            symbol=AAPL/
                part-<uuid>.parquet
                part-<uuid>.parquet
            symbol=MSFT/
                part-<uuid>.parquet
        date=2024-03-02/
            ...

Why ``date=`` first, then ``symbol=``?
  * **XGBoost / cross-sectional reads** filter by date range -- they read
    a contiguous block of top-level partitions and can ignore symbol.
  * **LSTM / sequence-per-stock reads** can either read all partitions
    and filter on symbol (cheap because partition pruning still happens),
    or be replayed per-symbol via the same predicate.
  * Date-first means **incremental writes only touch new partitions** and
    retention/archival is a directory move.
  * Disabling ``partition_by_symbol`` packs all symbols for one day into
    a single partition with multiple row groups, which is best for
    cross-sectional reads when the dataset is small enough.

Compression & row-group sizing
------------------------------
Defaults: zstd level 3 + 200k-row row groups. zstd typically delivers
~30 % smaller files than snappy on financial floats while staying CPU-cheap
to decode. Row groups around 128 MB give DuckDB / PyArrow good
parallelism without bloating per-row metadata.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from financials.etl_pipeline.config.etl_config import LoadConfig
from financials.etl_pipeline.utils.logging import get_logger
from financials.etl_pipeline.utils.validation import cast_float32

logger = get_logger(__name__)


class ParquetLoader:
    """Append-only writer for the Hive-partitioned feature dataset."""

    def __init__(self, cfg: LoadConfig) -> None:
        self.cfg = cfg
        self.output_path = Path(cfg.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def write(self, df: pd.DataFrame, *, ticker: str) -> int:
        """Write *df* to the dataset and return the number of rows written."""
        if df.empty:
            return 0

        if "date" not in df.columns:
            raise ValueError("Feature frame missing required 'date' column")
        if "symbol" not in df.columns:
            raise ValueError("Feature frame missing required 'symbol' column")

        # Cast floats once for the entire frame instead of per-partition.
        if self.cfg.cast_float32:
            df = cast_float32(df.copy())

        rows_written = 0
        for partition_path, partition_df in self._iter_partitions(df, ticker):
            partition_path.mkdir(parents=True, exist_ok=True)
            file_path = partition_path / f"part-{uuid.uuid4().hex[:12]}.parquet"
            self._write_one(partition_df, file_path)
            rows_written += len(partition_df)

        logger.info("Wrote {} rows for {} into {}", rows_written, ticker, self.output_path)
        return rows_written

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _iter_partitions(self, df: pd.DataFrame, ticker: str):
        """Yield ``(partition_path, sub_dataframe)`` for each Hive partition.

        Drops the partition columns from the sub-frame so they aren't
        duplicated in the parquet payload (Hive convention).
        """
        for date_value, group in df.groupby("date", sort=True):
            partition_path = self.output_path / f"date={date_value}"
            if self.cfg.partition_by_symbol:
                partition_path = partition_path / f"symbol={ticker}"
                payload = group.drop(columns=["date", "symbol"])
            else:
                payload = group.drop(columns=["date"])
            yield partition_path, payload

    def _write_one(self, df: pd.DataFrame, file_path: Path) -> None:
        """Write a single parquet file with the configured codec & row-group size."""
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(
            table,
            file_path,
            compression=self.cfg.compression,
            compression_level=self.cfg.compression_level,
            row_group_size=self.cfg.row_group_size,
            use_dictionary=True,
            data_page_size=1 << 20,  # 1 MB pages
        )
