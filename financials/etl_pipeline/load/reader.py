"""ML-friendly reader over the Hive-partitioned Parquet dataset.

Built on DuckDB so that:
  * partition pruning happens automatically for ``date=`` / ``symbol=``
    filters in WHERE clauses,
  * predicates push down into Parquet row-group statistics (no need to
    read columns we don't request),
  * results can be materialised as pandas, Polars, or PyArrow without
    extra copies.

Three convenience APIs:

  * :meth:`MLDataLoader.load_pandas` — the simple "give me a dataframe"
    path, perfect for XGBoost / sklearn cross-sectional training.
  * :meth:`MLDataLoader.iter_sequences` — yields aligned (X, y) windows
    per symbol for LSTM / Transformer training.
  * :meth:`MLDataLoader.to_torch_dataset` — wraps :meth:`iter_sequences`
    in a ``torch.utils.data.Dataset`` (only imported lazily so PyTorch
    is an optional dependency).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Sequence

import duckdb
import pandas as pd

from financials.etl_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


class MLDataLoader:
    """Read features from the Parquet dataset with predicate pushdown."""

    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path).resolve()
        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        # One persistent in-process connection. DuckDB is single-process by
        # default which is what we want for training scripts.
        self._con = duckdb.connect(":memory:")
        self._con.execute("PRAGMA threads=8")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def available_dates(self) -> list[str]:
        """Sorted list of dates present in the dataset (from partition names)."""
        dates = sorted({p.name.split("=", 1)[1] for p in self.dataset_path.glob("date=*")})
        return dates

    def available_symbols(self) -> list[str]:
        """Sorted list of symbols present in the dataset (from partition names)."""
        symbols: set[str] = set()
        for sym_dir in self.dataset_path.glob("date=*/symbol=*"):
            symbols.add(sym_dir.name.split("=", 1)[1])
        if symbols:
            return sorted(symbols)

        # Dataset wasn't partitioned by symbol -- fall back to a SQL query.
        try:
            df = self._con.execute(
                f"SELECT DISTINCT symbol FROM '{self._glob_path()}' ORDER BY symbol"
            ).fetchdf()
            return df["symbol"].tolist()
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Pandas / tabular reads
    # ------------------------------------------------------------------

    def load_pandas(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        columns: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Load a slice of the dataset as a pandas DataFrame.

        Parameters
        ----------
        symbols:
            Subset of tickers to read. ``None`` ⇒ all.
        start:
            Inclusive start date filter (pushed down to partition pruning).
        end:
            Inclusive end date filter (pushed down to partition pruning).
        columns:
            Subset of columns to project. ``None`` ⇒ ``SELECT *``.
            Always include the ``timestamp`` column or you'll lose the
            ordering needed for sequence training.
        limit:
            Optional ``LIMIT`` clause -- useful for sanity checks.
        """
        sql, params = self._build_select(
            symbols=symbols, start=start, end=end, columns=columns, limit=limit,
        )
        logger.debug("DuckDB query: {}", sql)
        df = self._con.execute(sql, params).fetchdf()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values(["symbol", "timestamp"]) if "symbol" in df.columns else df.sort_values("timestamp")
            df = df.reset_index(drop=True)

        logger.info("Loaded {} rows from {}", len(df), self.dataset_path)
        return df

    def load_arrow(self, **kwargs: Any):
        """Same arguments as :meth:`load_pandas` but returns a PyArrow Table."""
        sql, params = self._build_select(**kwargs)
        return self._con.execute(sql, params).arrow()

    # ------------------------------------------------------------------
    # Sequence iteration (LSTM / Transformer)
    # ------------------------------------------------------------------

    def iter_sequences(
        self,
        *,
        sequence_length: int,
        feature_columns: Sequence[str],
        target_column: str,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        stride: int = 1,
    ) -> Iterator[tuple[str, pd.Timestamp, "Any", float]]:
        """Yield ``(symbol, timestamp, X_window, y)`` tuples per symbol.

        ``X_window`` is a NumPy array of shape ``(sequence_length, len(feature_columns))``
        and ``y`` is the target value at the *last* timestamp of the window.

        The iteration happens **per symbol**, so each yielded window is
        contiguous in time and contains no cross-stock leakage.
        """
        import numpy as np

        cols = list(dict.fromkeys(["timestamp", "symbol", target_column, *feature_columns]))
        target_symbols = symbols or self.available_symbols()

        for sym in target_symbols:
            df = self.load_pandas(
                symbols=[sym], start=start, end=end, columns=cols,
            )
            if len(df) <= sequence_length:
                logger.debug("Symbol {} has only {} rows -- skipped", sym, len(df))
                continue

            features = df[list(feature_columns)].to_numpy(dtype="float32")
            targets = df[target_column].to_numpy(dtype="float32")
            timestamps = df["timestamp"].to_numpy()

            for i in range(sequence_length, len(df), stride):
                window = features[i - sequence_length : i]
                yield sym, timestamps[i - 1], window, float(targets[i - 1])

    # ------------------------------------------------------------------
    # PyTorch convenience
    # ------------------------------------------------------------------

    def to_torch_dataset(
        self,
        *,
        sequence_length: int,
        feature_columns: Sequence[str],
        target_column: str,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        stride: int = 1,
    ):
        """Materialise an in-memory ``torch.utils.data.Dataset``.

        Suitable for medium datasets (fits in RAM). For larger datasets,
        write a streaming ``IterableDataset`` around :meth:`iter_sequences`.
        """
        import numpy as np
        import torch
        from torch.utils.data import Dataset

        sequences: list[np.ndarray] = []
        targets: list[float] = []
        for _, _, x, y in self.iter_sequences(
            sequence_length=sequence_length,
            feature_columns=feature_columns,
            target_column=target_column,
            symbols=symbols,
            start=start,
            end=end,
            stride=stride,
        ):
            sequences.append(x)
            targets.append(y)

        if not sequences:
            raise ValueError("No sequences produced -- check filters and data availability")

        X = torch.from_numpy(np.stack(sequences))
        y = torch.tensor(targets, dtype=torch.float32)

        class _ParquetSequenceDataset(Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y

            def __len__(self) -> int:
                return self.X.shape[0]

            def __getitem__(self, idx: int):
                return self.X[idx], self.y[idx]

        return _ParquetSequenceDataset(X, y)

    # ------------------------------------------------------------------
    # SQL builder
    # ------------------------------------------------------------------

    def _glob_path(self) -> str:
        # ``hive_partitioning=1`` lets DuckDB infer ``date`` and ``symbol``
        # columns from the directory names automatically.
        return f"{self.dataset_path.as_posix()}/**/*.parquet"

    def _build_select(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        columns: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> tuple[str, list[Any]]:
        cols_sql = "*" if not columns else ", ".join(_quote(c) for c in columns)
        params: list[Any] = []
        where_parts: list[str] = []

        if start is not None:
            where_parts.append("date >= ?")
            params.append(_to_iso_date(start))
        if end is not None:
            where_parts.append("date <= ?")
            params.append(_to_iso_date(end))
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            where_parts.append(f"symbol IN ({placeholders})")
            params.extend(symbols)

        where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        limit_sql = f"LIMIT {int(limit)}" if limit else ""

        sql = (
            f"SELECT {cols_sql} "
            f"FROM read_parquet('{self._glob_path()}', hive_partitioning=1) "
            f"{where_sql} {limit_sql}".strip()
        )
        return sql, params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quote(col: str) -> str:
    # DuckDB treats double-quoted identifiers literally -- safe even when
    # a column name happens to be a reserved word.
    if col == "*":
        return col
    return '"' + col.replace('"', '""') + '"'


def _to_iso_date(value: str | pd.Timestamp) -> str:
    if isinstance(value, str):
        return value[:10]
    return pd.Timestamp(value).strftime("%Y-%m-%d")
