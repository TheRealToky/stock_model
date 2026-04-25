"""Bulk-insert repository for raw / cleaned flights and features.

Each method is idempotent: re-running an ingestion over an existing
range upserts the overlapping rows instead of failing on conflict.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd
from sqlalchemy import text

from alt_data_pipeline.src.config.settings import alt_settings
from alt_data_pipeline.src.storage.connection import get_alt_session
from alt_data_pipeline.src.utils.logging import get_logger

logger = get_logger(__name__)


class FlightRepository:
    """Persist and upsert flight-pipeline outputs to the alt-data DB."""

    def __init__(self, batch_size: int | None = None) -> None:
        self.batch_size = batch_size or alt_settings.pipeline.batch_size

    # ------------------------------------------------------------------
    # Raw + cleaned flights
    # ------------------------------------------------------------------

    def save_raw(self, df: pd.DataFrame) -> int:
        """Upsert a raw-flight DataFrame into ``raw_flights``."""
        if df.empty:
            logger.info("save_raw: empty DataFrame, nothing to do")
            return 0

        records = self._raw_records(df)
        sql = text(
            """
            INSERT INTO raw_flights (
                icao24, first_seen, direction, airport_icao,
                callsign, last_seen,
                est_departure_airport, est_arrival_airport,
                est_departure_airport_horiz_distance,
                est_departure_airport_vert_distance,
                est_arrival_airport_horiz_distance,
                est_arrival_airport_vert_distance,
                departure_airport_candidates_count,
                arrival_airport_candidates_count
            ) VALUES (
                :icao24, :first_seen, :direction, :airport_icao,
                :callsign, :last_seen,
                :est_departure_airport, :est_arrival_airport,
                :est_departure_airport_horiz_distance,
                :est_departure_airport_vert_distance,
                :est_arrival_airport_horiz_distance,
                :est_arrival_airport_vert_distance,
                :departure_airport_candidates_count,
                :arrival_airport_candidates_count
            )
            ON CONFLICT (icao24, first_seen, direction, airport_icao)
            DO UPDATE SET
                callsign  = EXCLUDED.callsign,
                last_seen = EXCLUDED.last_seen,
                est_departure_airport = EXCLUDED.est_departure_airport,
                est_arrival_airport   = EXCLUDED.est_arrival_airport
            """
        )
        return self._bulk_upsert(sql, records, "raw_flights")

    def save_clean(self, df: pd.DataFrame) -> int:
        """Upsert a cleaned-flight DataFrame into ``cleaned_flights``."""
        if df.empty:
            logger.info("save_clean: empty DataFrame, nothing to do")
            return 0

        records = self._clean_records(df)
        sql = text(
            """
            INSERT INTO cleaned_flights (
                icao24, first_seen, direction, airport_icao,
                callsign, last_seen, first_seen_dt, last_seen_dt,
                est_departure_airport, est_arrival_airport,
                est_departure_airport_horiz_distance,
                est_arrival_airport_horiz_distance,
                departure_airport_candidates_count,
                arrival_airport_candidates_count
            ) VALUES (
                :icao24, :first_seen, :direction, :airport_icao,
                :callsign, :last_seen, :first_seen_dt, :last_seen_dt,
                :est_departure_airport, :est_arrival_airport,
                :est_departure_airport_horiz_distance,
                :est_arrival_airport_horiz_distance,
                :departure_airport_candidates_count,
                :arrival_airport_candidates_count
            )
            ON CONFLICT (icao24, first_seen, direction, airport_icao)
            DO UPDATE SET
                callsign       = EXCLUDED.callsign,
                last_seen      = EXCLUDED.last_seen,
                first_seen_dt  = EXCLUDED.first_seen_dt,
                last_seen_dt   = EXCLUDED.last_seen_dt,
                est_departure_airport = EXCLUDED.est_departure_airport,
                est_arrival_airport   = EXCLUDED.est_arrival_airport
            """
        )
        return self._bulk_upsert(sql, records, "cleaned_flights")

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def save_features(self, df: pd.DataFrame) -> int:
        """Upsert features (long format: airport, ts, name, value)."""
        if df.empty:
            logger.info("save_features: empty DataFrame, nothing to do")
            return 0

        required = {"airport_icao", "timestamp", "feature_name", "feature_value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"save_features: missing columns {missing}; got {list(df.columns)}"
            )

        records = []
        for _, row in df.iterrows():
            value = row["feature_value"]
            if pd.isna(value):
                continue
            records.append(
                {
                    "airport_icao": str(row["airport_icao"]).upper(),
                    "timestamp": _coerce_dt(row["timestamp"]),
                    "feature_name": str(row["feature_name"]),
                    "feature_value": float(value),
                }
            )

        sql = text(
            """
            INSERT INTO flight_features
                (airport_icao, timestamp, feature_name, feature_value)
            VALUES
                (:airport_icao, :timestamp, :feature_name, :feature_value)
            ON CONFLICT (airport_icao, timestamp, feature_name)
            DO UPDATE SET feature_value = EXCLUDED.feature_value
            """
        )
        return self._bulk_upsert(sql, records, "flight_features")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _bulk_upsert(
        self,
        sql,
        records: list[dict],
        table: str,
    ) -> int:
        if not records:
            logger.info("{}: 0 records, skipping", table)
            return 0

        session = get_alt_session()
        upserted = 0
        try:
            for start in range(0, len(records), self.batch_size):
                batch = records[start : start + self.batch_size]
                session.execute(sql, batch)
                session.commit()
                upserted += len(batch)
            logger.info("Upserted {} rows into {}", upserted, table)
            return upserted
        except Exception:
            session.rollback()
            logger.exception("Bulk upsert into {} failed", table)
            raise
        finally:
            session.close()

    @staticmethod
    def _raw_records(df: pd.DataFrame) -> list[dict]:
        keep = [
            "icao24", "first_seen", "direction", "airport_icao", "callsign",
            "last_seen", "est_departure_airport", "est_arrival_airport",
            "est_departure_airport_horiz_distance",
            "est_departure_airport_vert_distance",
            "est_arrival_airport_horiz_distance",
            "est_arrival_airport_vert_distance",
            "departure_airport_candidates_count",
            "arrival_airport_candidates_count",
        ]
        out: list[dict] = []
        for _, row in df.iterrows():
            rec = {col: _scalar(row.get(col)) for col in keep}
            rec["first_seen"] = int(rec["first_seen"])
            rec["last_seen"] = int(rec["last_seen"])
            out.append(rec)
        return out

    @staticmethod
    def _clean_records(df: pd.DataFrame) -> list[dict]:
        keep = [
            "icao24", "first_seen", "direction", "airport_icao",
            "callsign", "last_seen", "first_seen_dt", "last_seen_dt",
            "est_departure_airport", "est_arrival_airport",
            "est_departure_airport_horiz_distance",
            "est_arrival_airport_horiz_distance",
            "departure_airport_candidates_count",
            "arrival_airport_candidates_count",
        ]
        out: list[dict] = []
        for _, row in df.iterrows():
            rec = {col: _scalar(row.get(col)) for col in keep}
            rec["first_seen"] = int(rec["first_seen"])
            rec["last_seen"] = int(rec["last_seen"])
            rec["first_seen_dt"] = _coerce_dt(row["first_seen_dt"])
            rec["last_seen_dt"] = _coerce_dt(row["last_seen_dt"])
            out.append(rec)
        return out


def _scalar(v):
    """Convert pandas NA / numpy NaN to Python None for DB insertion."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if v is pd.NA:
        return None
    return v


def _coerce_dt(v) -> datetime:
    if isinstance(v, datetime):
        return v
    ts = pd.Timestamp(v)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()
