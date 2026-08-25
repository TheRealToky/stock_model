"""Feature computation engine for the quant-lab pipeline.

The :class:`FeatureEngine` ties together the feature registry, the OHLCV data
loader, and the database persistence layer so that features can be computed,
stored, and retrieved with a single method call.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Sequence

import pandas as pd
from loguru import logger
from sqlalchemy import text

from financials.config.settings import settings
from financials.database.connection import get_engine, get_session

if TYPE_CHECKING:  # pragma: no cover - static analysers only
    from financials.data_pipeline.loader import DataLoader  # noqa: F401
from financials.features.registry import FEATURE_REGISTRY, get_feature_func, list_features


class FeatureEngine:
    """Compute, persist, and retrieve features for one or many instruments.

    Parameters
    ----------
    loader:
        An optional :class:`DataLoader` instance.  When ``None`` one is created
        on first use -- not at construction.  :meth:`compute_features` works on
        an in-memory DataFrame and needs no database at all, so building a
        DB-backed loader (and pulling in a market-data client with it) just to
        compute an RSI would be a needless dependency.  Only the DB-backed
        methods trigger it.
    """

    def __init__(self, loader: "DataLoader | None" = None) -> None:
        self._loader = loader

    @property
    def loader(self) -> "DataLoader":
        """The DB-backed loader, constructed on first access."""
        if self._loader is None:
            from financials.data_pipeline.loader import DataLoader as _DataLoader

            self._loader = _DataLoader()
        return self._loader

    @loader.setter
    def loader(self, value: "DataLoader | None") -> None:
        self._loader = value

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_features(
        self,
        df: pd.DataFrame,
        feature_names: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Compute features on an in-memory OHLCV DataFrame.

        Parameters
        ----------
        df:
            Must contain at least ``open``, ``high``, ``low``, ``close``,
            ``volume`` columns.
        feature_names:
            An iterable of registered feature names.  When ``None`` **all**
            registered features are computed.

        Returns
        -------
        pd.DataFrame
            A copy of the input with additional feature columns appended.
            Composite features (e.g. MACD) are flattened to individual columns
            with a ``<feature_name>_<sub_column>`` naming convention.
        """
        names = feature_names or list_features()
        result = df.copy()

        for name in names:
            try:
                func, default_params = get_feature_func(name)
                output = func(df, **default_params)

                if isinstance(output, pd.DataFrame):
                    # Flatten composite features
                    for col in output.columns:
                        result[f"{name}_{col}"] = output[col].values
                else:
                    result[name] = output.values
            except Exception:
                logger.exception("Failed to compute feature '{}'", name)

        logger.info(
            "Computed {} feature(s) -- result has {} columns",
            len(names),
            result.shape[1],
        )
        return result

    # ------------------------------------------------------------------
    # Compute + persist
    # ------------------------------------------------------------------

    def compute_and_store(
        self,
        ticker: str,
        feature_names: Sequence[str] | None = None,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from the database, compute features, and write them
        back to the ``features`` table.

        Parameters
        ----------
        ticker:
            Instrument ticker symbol (e.g. ``"AAPL"``).
        feature_names:
            Feature names to compute.  Defaults to all registered.
        start_date / end_date:
            Optional date filters forwarded to the data loader.

        Returns
        -------
        pd.DataFrame
            The feature DataFrame that was persisted.
        """
        logger.info("compute_and_store: ticker={}", ticker)

        # 1. Load OHLCV --------------------------------------------------
        ohlcv = self.loader.load_ohlcv(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
        if ohlcv.empty:
            logger.warning("No OHLCV data for ticker '{}' -- skipping", ticker)
            return ohlcv

        # 2. Compute features --------------------------------------------
        features_df = self.compute_features(ohlcv, feature_names)

        # 3. Resolve instrument_id ----------------------------------------
        instrument_id = self._resolve_instrument_id(ticker)
        if instrument_id is None:
            logger.error("Instrument '{}' not found in DB -- cannot store", ticker)
            return features_df

        # 4. Persist in long format ---------------------------------------
        self._store_features(instrument_id, features_df, ohlcv.columns)

        return features_df

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def compute_batch(
        self,
        tickers: Sequence[str],
        feature_names: Sequence[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Compute and store features for multiple tickers.

        Returns a mapping ``{ticker: feature_df}``.
        """
        results: dict[str, pd.DataFrame] = {}
        total = len(tickers)

        for idx, ticker in enumerate(tickers, start=1):
            logger.info("Processing {}/{}: {}", idx, total, ticker)
            try:
                results[ticker] = self.compute_and_store(
                    ticker, feature_names=feature_names
                )
            except Exception:
                logger.exception("Error processing ticker '{}'", ticker)

        logger.info(
            "Batch complete: {}/{} tickers succeeded",
            len(results),
            total,
        )
        return results

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_feature_matrix(
        self,
        ticker: str,
        feature_names: Sequence[str],
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """Load previously computed features from the ``features`` table and
        pivot them into a wide DataFrame (one column per feature).

        Parameters
        ----------
        ticker:
            Instrument ticker symbol.
        feature_names:
            Which features to retrieve.
        start_date / end_date:
            Optional date range filter.

        Returns
        -------
        pd.DataFrame
            Index = ``timestamp``, columns = feature names.
        """
        instrument_id = self._resolve_instrument_id(ticker)
        if instrument_id is None:
            logger.warning("Instrument '{}' not found -- returning empty DF", ticker)
            return pd.DataFrame()

        query_parts = [
            "SELECT timestamp, feature_name, feature_value "
            "FROM features "
            "WHERE instrument_id = :instrument_id "
            "AND feature_name IN :feature_names"
        ]
        params: dict = {
            "instrument_id": instrument_id,
            "feature_names": tuple(feature_names),
        }

        if start_date is not None:
            query_parts.append("AND timestamp >= :start_date")
            params["start_date"] = start_date
        if end_date is not None:
            query_parts.append("AND timestamp <= :end_date")
            params["end_date"] = end_date

        query_parts.append("ORDER BY timestamp")
        query = text(" ".join(query_parts))

        engine = get_engine()
        long_df = pd.read_sql(query, engine, params=params)

        if long_df.empty:
            logger.warning("No features found for '{}' with names {}", ticker, feature_names)
            return pd.DataFrame()

        wide_df = long_df.pivot(
            index="timestamp",
            columns="feature_name",
            values="feature_value",
        )
        wide_df.columns.name = None
        wide_df.index.name = "timestamp"

        logger.info(
            "Loaded feature matrix for '{}': {} rows x {} features",
            ticker,
            wide_df.shape[0],
            wide_df.shape[1],
        )
        return wide_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_instrument_id(self, ticker: str) -> int | None:
        """Look up the ``instruments.id`` for a given ticker."""
        session = get_session()
        try:
            row = session.execute(
                text("SELECT id FROM instruments WHERE ticker = :ticker"),
                {"ticker": ticker},
            ).fetchone()
            return row[0] if row else None
        finally:
            session.close()

    def _store_features(
        self,
        instrument_id: int,
        features_df: pd.DataFrame,
        ohlcv_columns: pd.Index,
    ) -> None:
        """Persist computed feature columns to the ``features`` table.

        Only columns that are **not** part of the original OHLCV data are
        written.  The data is inserted using an upsert-style approach
        (``ON CONFLICT ... DO UPDATE``) so re-running is idempotent.
        """
        # Determine which columns are actual features (not raw OHLCV)
        feature_cols = [c for c in features_df.columns if c not in ohlcv_columns]

        if not feature_cols:
            logger.warning("No feature columns to store")
            return

        # Build a long-format list of rows
        records: list[dict] = []
        for _, row in features_df.iterrows():
            ts = row.get("timestamp") or row.name
            for col in feature_cols:
                value = row[col]
                # Skip NaN values to keep the table clean
                if pd.isna(value):
                    continue
                records.append({
                    "instrument_id": instrument_id,
                    "timestamp": ts,
                    "feature_name": col,
                    "feature_value": float(value),
                })

        if not records:
            logger.warning("All feature values are NaN -- nothing to store")
            return

        logger.info(
            "Storing {} feature records for instrument_id={}",
            len(records),
            instrument_id,
        )

        upsert_sql = text(
            "INSERT INTO features (instrument_id, timestamp, feature_name, feature_value) "
            "VALUES (:instrument_id, :timestamp, :feature_name, :feature_value) "
            "ON CONFLICT (instrument_id, timestamp, feature_name) "
            "DO UPDATE SET feature_value = EXCLUDED.feature_value"
        )

        session = get_session()
        try:
            # Batch insert in chunks for memory efficiency
            batch_size = settings.pipeline.batch_size * 100  # features produce many rows
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                session.execute(upsert_sql, batch)
            session.commit()
            logger.info("Feature storage complete")
        except Exception:
            session.rollback()
            logger.exception("Failed to store features")
            raise
        finally:
            session.close()
