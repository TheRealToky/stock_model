"""SQLAlchemy ORM models for the alternative-data (flight) database.

Three tables mirror the pipeline stages:

* ``raw_flights``     - immutable record of everything we fetched
* ``cleaned_flights`` - deduped, normalized, validated
* ``flight_features`` - long-format feature values per flight/airport/day

All time-series tables use a composite primary key so a single SERIAL
isn't required and the tables can be converted to TimescaleDB
hypertables by the migration.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase


class AltBase(DeclarativeBase):
    """Declarative base for the alt-data schema."""


# ---------------------------------------------------------------------------
# raw_flights -- one row per (icao24, first_seen, direction, airport)
# ---------------------------------------------------------------------------


class RawFlight(AltBase):
    __tablename__ = "raw_flights"

    icao24 = Column(String(12), primary_key=True, nullable=False)
    first_seen = Column(BigInteger, primary_key=True, nullable=False)
    direction = Column(String(10), primary_key=True, nullable=False)
    airport_icao = Column(String(8), primary_key=True, nullable=False)

    callsign = Column(String(16), nullable=True)
    last_seen = Column(BigInteger, nullable=False)
    est_departure_airport = Column(String(8), nullable=True)
    est_arrival_airport = Column(String(8), nullable=True)
    est_departure_airport_horiz_distance = Column(Integer, nullable=True)
    est_departure_airport_vert_distance = Column(Integer, nullable=True)
    est_arrival_airport_horiz_distance = Column(Integer, nullable=True)
    est_arrival_airport_vert_distance = Column(Integer, nullable=True)
    departure_airport_candidates_count = Column(Integer, nullable=True)
    arrival_airport_candidates_count = Column(Integer, nullable=True)
    ingested_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_raw_flights_airport_first_seen", "airport_icao", "first_seen"),
        Index("ix_raw_flights_direction", "direction"),
    )


# ---------------------------------------------------------------------------
# cleaned_flights -- same PK, post-cleaning values, tz-aware timestamps
# ---------------------------------------------------------------------------


class CleanedFlight(AltBase):
    __tablename__ = "cleaned_flights"

    icao24 = Column(String(12), primary_key=True, nullable=False)
    first_seen = Column(BigInteger, primary_key=True, nullable=False)
    direction = Column(String(10), primary_key=True, nullable=False)
    airport_icao = Column(String(8), primary_key=True, nullable=False)

    callsign = Column(String(16), nullable=True, index=True)
    last_seen = Column(BigInteger, nullable=False)
    first_seen_dt = Column(DateTime(timezone=True), nullable=False, index=True)
    last_seen_dt = Column(DateTime(timezone=True), nullable=False)
    est_departure_airport = Column(String(8), nullable=True, index=True)
    est_arrival_airport = Column(String(8), nullable=True, index=True)
    est_departure_airport_horiz_distance = Column(Integer, nullable=True)
    est_arrival_airport_horiz_distance = Column(Integer, nullable=True)
    departure_airport_candidates_count = Column(Integer, nullable=True)
    arrival_airport_candidates_count = Column(Integer, nullable=True)
    cleaned_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint(
            "icao24", "first_seen", "direction", "airport_icao",
            name="uq_cleaned_flights_natural_key",
        ),
        Index(
            "ix_cleaned_flights_airport_time",
            "airport_icao", "first_seen_dt",
        ),
        Index(
            "ix_cleaned_flights_route",
            "est_departure_airport", "est_arrival_airport",
        ),
    )


# ---------------------------------------------------------------------------
# flight_features -- long format, one row per (airport, day, feature)
# ---------------------------------------------------------------------------


class FlightFeature(AltBase):
    __tablename__ = "flight_features"

    airport_icao = Column(String(8), primary_key=True, nullable=False)
    timestamp = Column(
        DateTime(timezone=True), primary_key=True, nullable=False, index=True
    )
    feature_name = Column(String(100), primary_key=True, nullable=False)

    feature_value = Column(Float, nullable=True)
    feature_meta = Column(JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint(
            "airport_icao", "timestamp", "feature_name",
            name="uq_flight_features_natural_key",
        ),
        Index(
            "ix_flight_features_airport_name_ts",
            "airport_icao", "feature_name", "timestamp",
        ),
    )
