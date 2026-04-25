"""FlightData: typed dataclass for a single OpenSky arrival/departure record.

The OpenSky ``/flights/arrival`` and ``/flights/departure`` endpoints
return the same schema per row.  We model that schema here so the rest
of the pipeline can use dot-access (``f.icao24``) instead of string
keys.  Using a plain ``@dataclass`` keeps the object hashable, printable,
and trivially convertible to a DataFrame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class FlightData:
    """Single flight observation from OpenSky (arrival *or* departure).

    Attributes follow the OpenSky API schema verbatim, with two additions:
    ``direction`` (``"arrival" | "departure"``) and ``airport_icao``
    (the airport we queried).  The ``first_seen`` / ``last_seen`` Unix
    timestamps are also exposed as tz-aware datetimes for convenience.
    """

    # --- Identity ------------------------------------------------------
    icao24: str
    callsign: str | None

    # --- Timing (Unix seconds, UTC) -----------------------------------
    first_seen: int
    last_seen: int

    # --- Airports ------------------------------------------------------
    est_departure_airport: str | None
    est_arrival_airport: str | None

    # --- Distance / confidence fields ---------------------------------
    est_departure_airport_horiz_distance: int | None
    est_departure_airport_vert_distance: int | None
    est_arrival_airport_horiz_distance: int | None
    est_arrival_airport_vert_distance: int | None

    # --- Counts --------------------------------------------------------
    departure_airport_candidates_count: int | None
    arrival_airport_candidates_count: int | None

    # --- Context we add ourselves -------------------------------------
    direction: str                    # "arrival" | "departure"
    airport_icao: str                 # the queried airport

    # -------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------
    @classmethod
    def from_api_payload(
        cls,
        payload: dict[str, Any],
        direction: str,
        airport_icao: str,
    ) -> "FlightData":
        """Build a FlightData from one row of an OpenSky JSON response."""
        return cls(
            icao24=str(payload.get("icao24", "")).strip().lower(),
            callsign=(payload.get("callsign") or None),
            first_seen=int(payload["firstSeen"]),
            last_seen=int(payload["lastSeen"]),
            est_departure_airport=payload.get("estDepartureAirport"),
            est_arrival_airport=payload.get("estArrivalAirport"),
            est_departure_airport_horiz_distance=payload.get(
                "estDepartureAirportHorizDistance"
            ),
            est_departure_airport_vert_distance=payload.get(
                "estDepartureAirportVertDistance"
            ),
            est_arrival_airport_horiz_distance=payload.get(
                "estArrivalAirportHorizDistance"
            ),
            est_arrival_airport_vert_distance=payload.get(
                "estArrivalAirportVertDistance"
            ),
            departure_airport_candidates_count=payload.get(
                "departureAirportCandidatesCount"
            ),
            arrival_airport_candidates_count=payload.get(
                "arrivalAirportCandidatesCount"
            ),
            direction=direction,
            airport_icao=airport_icao.upper(),
        )

    # -------------------------------------------------------------------
    # Convenience properties
    # -------------------------------------------------------------------
    @property
    def first_seen_dt(self) -> datetime:
        return datetime.fromtimestamp(self.first_seen, tz=timezone.utc)

    @property
    def last_seen_dt(self) -> datetime:
        return datetime.fromtimestamp(self.last_seen, tz=timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
