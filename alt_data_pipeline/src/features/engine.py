"""Feature engine: discover, compute, and persist flight features."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import pandas as pd

from alt_data_pipeline.src.config.settings import alt_settings
from alt_data_pipeline.src.features.registry import (
    FLIGHT_FEATURE_REGISTRY,
    get_feature_func,
    list_features,
)
from alt_data_pipeline.src.storage.loader import FlightLoader
from alt_data_pipeline.src.storage.repository import FlightRepository
from alt_data_pipeline.src.utils.logging import get_logger

logger = get_logger(__name__)


class FlightFeatureEngine:
    """Compute, persist, and retrieve flight-data features.

    Parameters
    ----------
    loader:
        Optional :class:`FlightLoader`.  Defaults to a fresh instance.
    repository:
        Optional :class:`FlightRepository`.  Defaults to a fresh instance.
    """

    def __init__(
        self,
        loader: FlightLoader | None = None,
        repository: FlightRepository | None = None,
    ) -> None:
        self.loader = loader or FlightLoader()
        self.repository = repository or FlightRepository()

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def compute_features(
        self,
        df: pd.DataFrame,
        feature_names: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Run all requested features on an in-memory cleaned DataFrame.

        Config (``config/features.yaml``) can disable features
        globally; explicit ``feature_names`` override that.
        """
        if df.empty:
            logger.warning("compute_features: empty input DataFrame")
            return _empty_long()

        if feature_names is None:
            toggles = alt_settings.feature_toggles()
            feature_names = [
                name
                for name in list_features()
                if toggles.get(name, True)
            ]

        results: list[pd.DataFrame] = []
        for name in feature_names:
            try:
                func, default_params = get_feature_func(name)
                out = func(df, **default_params)
                if out is None or out.empty:
                    logger.debug("Feature '{}' produced no rows", name)
                    continue
                required = {"airport_icao", "timestamp", "feature_name", "feature_value"}
                if not required.issubset(out.columns):
                    logger.error(
                        "Feature '{}' returned wrong schema: {}",
                        name, list(out.columns),
                    )
                    continue
                results.append(out)
            except Exception:
                logger.exception("Failed to compute feature '{}'", name)

        if not results:
            return _empty_long()

        long = pd.concat(results, ignore_index=True)
        logger.info(
            "Computed {} feature(s) -> {} total rows",
            len(feature_names),
            len(long),
        )
        return long

    # ------------------------------------------------------------------
    # Compute + persist
    # ------------------------------------------------------------------

    def compute_and_store(
        self,
        airport_icao: str,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Load cleaned flights -> compute features -> write features."""
        flights = self.loader.load_flights(
            airport_icao=airport_icao,
            start_date=start_date,
            end_date=end_date,
        )
        if flights.empty:
            logger.warning("No flight data for {} in range", airport_icao)
            return _empty_long()

        # loader returns flights indexed by first_seen_dt; the feature
        # functions expect that column as a regular column.
        flights = flights.reset_index()
        features = self.compute_features(flights, feature_names)
        if features.empty:
            return features

        self.repository.save_features(features)
        return features

    def compute_batch(
        self,
        airports: Sequence[str],
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Compute features for many airports; skip failures with a log."""
        results: dict[str, pd.DataFrame] = {}
        for airport in airports:
            try:
                results[airport] = self.compute_and_store(
                    airport, start_date, end_date, feature_names
                )
            except Exception:
                logger.exception("Feature computation failed for {}", airport)
        return results


def _empty_long() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["airport_icao", "timestamp", "feature_name", "feature_value"]
    )
