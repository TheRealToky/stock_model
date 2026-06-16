"""ML-friendly reader over the Hive-partitioned Parquet dataset.

Built on DuckDB so that:
  * partition pruning happens automatically for ``date=`` / ``symbol=``
    filters in WHERE clauses,
  * predicates push down into Parquet row-group statistics (no need to
    read columns we don't request),
  * results can be materialised as pandas, Polars, or PyArrow without
    extra copies.

Four convenience APIs:

  * :meth:`MLDataLoader.load_pandas` — the simple "give me a dataframe"
    path, perfect for XGBoost / sklearn cross-sectional training.
  * :meth:`MLDataLoader.iter_sequences` — yields aligned (X, y) windows
    per symbol for LSTM / Transformer training.
  * :meth:`MLDataLoader.to_torch_dataset` — wraps :meth:`iter_sequences`
    in a ``torch.utils.data.Dataset`` (only imported lazily so PyTorch
    is an optional dependency).  Materialises the full set in RAM —
    suitable only for small datasets.
  * :class:`StreamingSequenceDataset` — a ``torch.utils.data.IterableDataset``
    that streams windows symbol-by-symbol with a bounded shuffle buffer
    and never holds the full training set in memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import duckdb
import pandas as pd

from financials.etl_pipeline.utils.logging import get_logger

logger = get_logger(__name__)

# Inherit from torch.utils.data.IterableDataset when torch is available so
# DataLoader workers receive proper get_worker_info() integration.  Falls
# back to a no-op base when torch isn't installed (e.g. CPU-only ETL runs).
try:  # pragma: no cover - import guard
    from torch.utils.data import IterableDataset as _IterableDatasetBase
except ImportError:  # pragma: no cover - torch optional
    class _IterableDatasetBase:  # type: ignore[no-redef]
        pass


def _short_sql(sql: str, limit: int = 200) -> str:
    sql = " ".join(sql.split())
    if len(sql) <= limit:
        return sql
    head = limit * 2 // 3
    tail = limit - head - 5
    return f"{sql[:head]} ... {sql[-tail:]}"


class MLDataLoader:
    """Read features from the Parquet dataset with predicate pushdown.

    Parameters
    ----------
    dataset_path:
        Root of the Hive-partitioned Parquet store.
    memory_limit:
        DuckDB ``memory_limit`` pragma.  Caps the buffer manager so a long
        sweep of per-symbol queries (e.g. fitting a scaler over a 1-min
        OHLCV store) cannot consume the container's RAM.  ``None`` keeps
        DuckDB's default (80% of physical RAM).
    threads:
        DuckDB ``threads`` pragma.  More threads = more concurrent buffer
        allocation per query; 4 is a reasonable trade-off for the
        single-symbol queries the streaming pipeline issues.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        memory_limit: str | None = "1GB",
        threads: int = 4,
    ) -> None:
        self.dataset_path = Path(dataset_path).resolve()
        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        # One persistent in-process connection. DuckDB is single-process by
        # default which is what we want for training scripts.
        self._con = duckdb.connect(":memory:")
        self._con.execute(f"PRAGMA threads={int(threads)}")
        if memory_limit is not None:
            # Without this DuckDB will retain parquet pages across queries up
            # to ~80% of host RAM, which is fatal in a memory-bounded container.
            self._con.execute(f"PRAGMA memory_limit='{memory_limit}'")
        self._memory_limit = memory_limit
        self._threads = int(threads)

    def reset_cache(self) -> None:
        """Drop and reopen the DuckDB connection to release cached pages.

        Use between major phases of a training run (e.g. scaler fit → window
        streaming) when the buffer manager has accumulated unused pages.
        """
        self._con.close()
        self._con = duckdb.connect(":memory:")
        self._con.execute(f"PRAGMA threads={self._threads}")
        if self._memory_limit is not None:
            self._con.execute(f"PRAGMA memory_limit='{self._memory_limit}'")

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
        glob = self._glob_path(
            start=_to_iso_date(start) if start is not None else None,
            end=_to_iso_date(end) if end is not None else None,
            symbols=symbols,
        )
        if not glob:
            logger.debug("No matching partitions found for symbols={} start={} end={}", symbols, start, end)
            return pd.DataFrame(columns=list(columns) if columns else [])

        sql, params = self._build_select(
            symbols=symbols, start=start, end=end, columns=columns, limit=limit,
        )
        logger.debug("DuckDB query: {}", _short_sql(sql))
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

    def _glob_path(
        self,
        start: str | None = None,
        end: str | None = None,
        symbols: Sequence[str] | None = None,
    ) -> str:
        """``hive_partitioning=1`` lets DuckDB infer ``date`` and ``symbol``
        columns from the directory names automatically.

        Enumerate matching date dirs from the filesystem so DuckDB only
        walks the relevant subset instead of the full /**/*.parquet tree.
        Without this, DuckDB must list every file before partition pruning,
        which is slow on large stores (especially on Windows).
        """
        date_dirs = self._matching_date_dirs(start, end)
        base = self.dataset_path.as_posix()
        if date_dirs is None:
            return f"{base}/**/*.parquet"
        if symbols and len(symbols) <= 20:
            paths = [
                f"{base}/date={d}/symbol={s}/*.parquet"
                for d in date_dirs
                for s in symbols
                if (self.dataset_path / f"date={d}" / f"symbol={s}").is_dir()
            ]
        else:
            paths = [
                f"{base}/date={d}/**/*.parquet"
                for d in date_dirs
                if (self.dataset_path / f"date={d}").is_dir()
            ]
        if not paths:
            return ""
        # DuckDB accepts a list literal: read_parquet(['a.parquet','b.parquet'])
        return "[" + ", ".join(f"'{p}'" for p in paths) + "]"

    def _matching_date_dirs(
        self,
        start: str | None,
        end: str | None,
    ) -> list[str] | None:
        """Return sorted date partition names that fall within [start, end].

        Returns ``None`` when no date filter is set (caller should fall back
        to the full recursive glob).
        """
        if start is None and end is None:
            return None
        all_dates = sorted(
            p.name.split("=", 1)[1] for p in self.dataset_path.glob("date=*")
        )
        return [
            d for d in all_dates
            if (start is None or d >= start) and (end is None or d <= end)
        ]

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

        iso_start = _to_iso_date(start) if start is not None else None
        iso_end = _to_iso_date(end) if end is not None else None

        if iso_start is not None:
            where_parts.append("date >= ?")
            params.append(iso_start)
        if iso_end is not None:
            where_parts.append("date <= ?")
            params.append(iso_end)
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            where_parts.append(f"symbol IN ({placeholders})")
            params.extend(symbols)

        glob = self._glob_path(start=iso_start, end=iso_end, symbols=symbols)
        # List globs use bracket syntax; single globs use quoted string syntax.
        from_clause = (
            f"read_parquet({glob}, hive_partitioning=1)"
            if glob.startswith("[")
            else f"read_parquet('{glob}', hive_partitioning=1)"
        )

        where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        limit_sql = f"LIMIT {int(limit)}" if limit else ""

        sql = (
            f"SELECT {cols_sql} "
            f"FROM {from_clause} "
            f"{where_sql} {limit_sql}".strip()
        )
        return sql, params


# ---------------------------------------------------------------------------
# Streaming dataset (IterableDataset) — never materialises the full set
# ---------------------------------------------------------------------------


class StreamingSequenceDataset(_IterableDatasetBase):
    """Streaming PyTorch ``IterableDataset`` for sliding-window training.

    Designed for the case where the *full* set of windows would not fit
    in RAM.  For each symbol in turn the dataset:

      1. Loads only that symbol's rows via DuckDB (``date`` / ``symbol``
         partition pruning so we never read more than requested).
      2. Applies the fitted scaler in one ``transform`` call.
      3. Yields sliding windows of length ``sequence_length``.

    Memory budget per worker (peak):

      * ``shuffle_buffer == 0`` (validation / test):
        one symbol's rows + one window in flight.
      * ``shuffle_buffer == N`` (training):
        one symbol's rows + at most ``N`` windows in a reservoir.
        Each window is *copied* on insertion so the per-symbol ``feats``
        array can be released between symbols.

    Time-series ordering is preserved *within* each symbol; symbols are
    processed sequentially so a window never spans across tickers.

    Parameters
    ----------
    loader:
        A configured :class:`MLDataLoader` pointing at the feature store.
    scaler:
        A **fitted** sklearn-compatible scaler (e.g. ``StandardScaler``).
        Applied to every symbol's feature matrix before windowing.
    feature_columns:
        Ordered list of feature column names that exist in the Parquet
        files.  Order must match the scaler's training order.
    target_fn:
        Callable ``(df: pd.DataFrame) -> pd.Series`` deriving the target
        from a symbol's DataFrame (e.g. next-bar direction).  Rows where
        the target is NaN are dropped.  Mutually exclusive with
        ``target_column``.
    target_column:
        Name of a target column already present in the Parquet files.
        Mutually exclusive with ``target_fn``.
    symbols:
        Subset of tickers to iterate over.  ``None`` ⇒ all symbols.
    start / end:
        Inclusive date-range filter (pushed down to partition pruning).
    sequence_length:
        Number of bars per window.
    stride:
        Step between consecutive windows (1 = fully overlapping).
    shuffle_buffer:
        Reservoir size in windows.  ``0`` disables shuffling and yields
        in time order — required for validation / test loaders.
    seed:
        Base RNG seed.  Each DataLoader worker derives its own seed as
        ``seed + worker_id`` so workers produce different orderings.
        Pass ``None`` to draw from system entropy (different shuffle
        every epoch).
    """

    def __init__(
        self,
        loader: "MLDataLoader",
        scaler: Any,
        feature_columns: Sequence[str],
        *,
        target_fn: Callable[[pd.DataFrame], pd.Series] | None = None,
        target_column: str | None = None,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        sequence_length: int = 60,
        stride: int = 1,
        shuffle_buffer: int = 0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if (target_fn is None) == (target_column is None):
            raise ValueError("Provide exactly one of target_fn or target_column.")

        self._loader = loader
        self._scaler = scaler
        self._feature_columns = list(feature_columns)
        self._target_fn = target_fn
        self._target_column = target_column
        self._symbols = list(symbols) if symbols is not None else None
        self._start = start
        self._end = end
        self._sequence_length = int(sequence_length)
        self._stride = max(int(stride), 1)
        self._shuffle_buffer = max(int(shuffle_buffer), 0)
        self._seed = seed

    # IterableDataset protocol — torch is imported lazily so this module
    # remains importable in CPU-only ETL contexts.
    def __iter__(self):
        import numpy as np
        import torch

        try:
            from torch.utils.data import get_worker_info
            worker = get_worker_info()
        except ImportError:  # pragma: no cover - torch optional
            worker = None

        target_symbols = self._symbols or self._loader.available_symbols()

        # Shard symbols across DataLoader workers so windows are never
        # duplicated.  Each symbol is owned by exactly one worker.
        if worker is not None and worker.num_workers > 1:
            target_symbols = [
                s for i, s in enumerate(target_symbols)
                if i % worker.num_workers == worker.id
            ]

        seed = self._seed
        if seed is not None and worker is not None:
            seed = seed + worker.id
        rng = np.random.default_rng(seed)

        if self._target_fn is not None:
            extra_cols: list[str] = []
        else:
            extra_cols = [self._target_column]  # type: ignore[list-item]
        load_cols = list(dict.fromkeys(
            ["timestamp", "symbol", *extra_cols, *self._feature_columns]
        ))

        buf_X: list["np.ndarray"] = []
        buf_y: list[float] = []

        for sym in target_symbols:
            df = (
                self._loader.load_pandas(
                    symbols=[sym],
                    start=self._start,
                    end=self._end,
                    columns=load_cols,
                )
                .dropna(subset=self._feature_columns)
                .reset_index(drop=True)
            )

            if self._target_fn is not None:
                df = df.copy()
                df["_target"] = self._target_fn(df)
                df = df.dropna(subset=["_target"]).reset_index(drop=True)
                target_arr = df["_target"].to_numpy(dtype="float32")
            else:
                df = df.dropna(subset=[self._target_column]).reset_index(drop=True)
                target_arr = df[self._target_column].to_numpy(dtype="float32")

            if len(df) <= self._sequence_length:
                continue

            # Scale the symbol once; the result is bounded by one ticker's
            # row count, not the full training set.
            feats = self._scaler.transform(
                df[self._feature_columns].values
            ).astype("float32", copy=False)

            for i in range(self._sequence_length, len(df), self._stride):
                # .copy() detaches the window from `feats` so it can be
                # garbage-collected as soon as we move to the next symbol.
                # Without this, every window would be a view holding the
                # whole symbol's feature matrix alive in memory.
                window = feats[i - self._sequence_length : i].copy()
                target = float(target_arr[i - 1])

                if self._shuffle_buffer == 0:
                    yield (
                        torch.from_numpy(window),
                        torch.tensor(target, dtype=torch.float32),
                    )
                    continue

                buf_X.append(window)
                buf_y.append(target)
                if len(buf_X) >= self._shuffle_buffer:
                    # Swap-with-tail-and-pop keeps each emit O(1).
                    idx = int(rng.integers(0, len(buf_X)))
                    buf_X[idx], buf_X[-1] = buf_X[-1], buf_X[idx]
                    buf_y[idx], buf_y[-1] = buf_y[-1], buf_y[idx]
                    yield (
                        torch.from_numpy(buf_X.pop()),
                        torch.tensor(buf_y.pop(), dtype=torch.float32),
                    )

            # Release per-symbol arrays before the next iteration.
            del feats, target_arr, df

        # Drain the reservoir in random order.  Bounded by shuffle_buffer.
        while buf_X:
            idx = int(rng.integers(0, len(buf_X)))
            buf_X[idx], buf_X[-1] = buf_X[-1], buf_X[idx]
            buf_y[idx], buf_y[-1] = buf_y[-1], buf_y[idx]
            yield (
                torch.from_numpy(buf_X.pop()),
                torch.tensor(buf_y.pop(), dtype=torch.float32),
            )


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
