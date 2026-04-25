"""Shared fixtures for alt-data pipeline tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest


@pytest.fixture
def sample_api_payload() -> list[dict]:
    """Minimal OpenSky ``/flights/arrival`` response fixture."""
    return [
        {
            "icao24": "abc123",
            "firstSeen": 1_704_067_200,  # 2024-01-01 00:00:00 UTC
            "estDepartureAirport": "EGLL",
            "lastSeen": 1_704_078_000,   # ~3 hours later
            "estArrivalAirport": "KJFK",
            "callsign": "  UAL100 ",
            "estDepartureAirportHorizDistance": 50,
            "estDepartureAirportVertDistance": 5,
            "estArrivalAirportHorizDistance": 100,
            "estArrivalAirportVertDistance": 10,
            "departureAirportCandidatesCount": 1,
            "arrivalAirportCandidatesCount": 1,
        },
        {
            "icao24": "def456",
            "firstSeen": 1_704_070_800,
            "estDepartureAirport": "EDDF",
            "lastSeen": 1_704_085_000,
            "estArrivalAirport": "KJFK",
            "callsign": "LHA200",
            "estDepartureAirportHorizDistance": 30,
            "estArrivalAirportHorizDistance": 80,
        },
    ]


@pytest.fixture
def cleaned_flight_df() -> pd.DataFrame:
    """A small cleaned-flight DataFrame for feature tests."""
    rows = []
    base = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
    airport = "KJFK"
    # 10 arrivals + 10 departures across 2 days, 5 unique aircraft
    for day_offset in range(2):
        for i in range(10):
            ts_first = base + day_offset * 86400 + i * 600
            ts_last = ts_first + 1800  # 30 min duration
            rows.append(
                {
                    "icao24": f"ac{i % 5:03d}",
                    "callsign": f"UAL{i:03d}",
                    "first_seen": ts_first,
                    "last_seen": ts_last,
                    "direction": "arrival" if i % 2 == 0 else "departure",
                    "airport_icao": airport,
                    "est_departure_airport": "EGLL" if i % 2 == 0 else airport,
                    "est_arrival_airport": airport if i % 2 == 0 else "EDDF",
                    "est_departure_airport_horiz_distance": 50,
                    "est_arrival_airport_horiz_distance": 60,
                    "departure_airport_candidates_count": 1,
                    "arrival_airport_candidates_count": 1,
                    "first_seen_dt": datetime.fromtimestamp(ts_first, tz=timezone.utc),
                    "last_seen_dt": datetime.fromtimestamp(ts_last, tz=timezone.utc),
                }
            )
    return pd.DataFrame(rows)
