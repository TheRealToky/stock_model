"""DataFetcher: high-level ingestion entry point for OpenSky.

Splits a user date range into daily UTC windows, calls the OpenSky
arrivals + departures endpoints for each day, normalizes the payloads
into :class:`FlightData` objects, and aggregates the whole thing into
a single pandas DataFrame ready for the cleaning stage.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from alt_data_pipeline.src.ingestion.flight_data import FlightData
from alt_data_pipeline.src.ingestion.opensky_client import OpenSkyClient
from alt_data_pipeline.src.utils.logging import get_logger
from alt_data_pipeline.src.utils.time_utils import (
    daily_utc_intervals,
    to_unix_seconds,
)

logger = get_logger(__name__)


class DataFetcher:
    """Fetch arrivals + departures for an airport over a date range.

    Parameters
    ----------
    client:
        Optional pre-configured :class:`OpenSkyClient`.  When ``None``
        a new one is built from :mod:`alt_settings`.
    """

    def __init__(self, client: OpenSkyClient | None = None) -> None:
        self.client = client or OpenSkyClient()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def fetch(
        self,
        airport_icao: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch arrivals + departures for *airport_icao* over a date range.

        The date range is split into daily UTC intervals; each day we
        call the arrivals and departures endpoints, normalize both
        payloads into :class:`FlightData`, and concatenate.

        Args:
            airport_icao: ICAO-4 airport code (e.g. ``"KJFK"``).
            start_date:   Range start (inclusive).  Naive datetimes are
                          assumed UTC.
            end_date:     Range end   (inclusive).

        Returns:
            A single aggregated DataFrame indexed by row number with one
            row per observed flight.  Columns match
            :class:`FlightData` fields plus a ``airport_icao`` column.
            Returns an empty DataFrame when no flights were found.
        """
        airport_icao = airport_icao.upper()
        logger.info(
            "Fetching flights for {} [{} -> {}]",
            airport_icao,
            start_date.isoformat(),
            end_date.isoformat(),
        )

        all_flights: list[FlightData] = []
        failed_windows: list[tuple[str, datetime, datetime]] = []

        for day_start, day_end in daily_utc_intervals(start_date, end_date):
            begin = to_unix_seconds(day_start)
            end = to_unix_seconds(day_end)

            # --- arrivals -------------------------------------------------
            try:
                arrivals_payload = self.client.get_arrivals_by_airport(
                    airport_icao, begin, end
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "arrivals failed for {} on {}: {}",
                    airport_icao, day_start.date(), exc,
                )
                failed_windows.append(("arrival", day_start, day_end))
                arrivals_payload = []

            # --- departures ----------------------------------------------
            try:
                departures_payload = self.client.get_departures_by_airport(
                    airport_icao, begin, end
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "departures failed for {} on {}: {}",
                    airport_icao, day_start.date(), exc,
                )
                failed_windows.append(("departure", day_start, day_end))
                departures_payload = []

            all_flights.extend(
                self._payload_to_flights(arrivals_payload, "arrival", airport_icao)
            )
            all_flights.extend(
                self._payload_to_flights(departures_payload, "departure", airport_icao)
            )

            logger.debug(
                "{} [{}] arr={} dep={}",
                airport_icao,
                day_start.date(),
                len(arrivals_payload),
                len(departures_payload),
            )

        if failed_windows:
            logger.warning(
                "{} of {} request windows failed -- returning partial data",
                len(failed_windows),
                sum(1 for _ in daily_utc_intervals(start_date, end_date)) * 2,
            )

        if not all_flights:
            logger.info("No flights returned for {}", airport_icao)
            return pd.DataFrame(columns=list(FlightData.__annotations__.keys()))

        df = pd.DataFrame([f.to_dict() for f in all_flights])
        df = self._normalize_dataframe(df)
        logger.info(
            "Aggregated {} flight rows for {} ({} arrivals, {} departures)",
            len(df),
            airport_icao,
            (df["direction"] == "arrival").sum(),
            (df["direction"] == "departure").sum(),
        )
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _payload_to_flights(
        payload: list[dict[str, Any]],
        direction: str,
        airport_icao: str,
    ) -> list[FlightData]:
        """Convert a raw JSON list into FlightData objects, skipping bad rows."""
        out: list[FlightData] = []
        for row in payload:
            try:
                out.append(
                    FlightData.from_api_payload(row, direction, airport_icao)
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("Skipping malformed row: {} ({})", exc, row)
        return out

    @staticmethod
    def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Add ``first_seen_dt`` / ``last_seen_dt`` columns and sort."""
        df = df.copy()
        df["first_seen_dt"] = pd.to_datetime(df["first_seen"], unit="s", utc=True)
        df["last_seen_dt"] = pd.to_datetime(df["last_seen"], unit="s", utc=True)
        df = df.sort_values("first_seen_dt").reset_index(drop=True)
        return df
