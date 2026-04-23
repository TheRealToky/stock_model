"""Cleaning stage for the flight-data pipeline.

Takes a raw DataFrame from :class:`DataFetcher`, dedupes, normalizes
strings and datetimes, filters low-confidence observations, and
returns a validated DataFrame ready to be persisted as ``cleaned_flights``.
"""

from __future__ import annotations

import pandas as pd

from alt_data_pipeline.src.config.settings import alt_settings
from alt_data_pipeline.src.utils.logging import get_logger

logger = get_logger(__name__)

# Columns the downstream tables require.  Cleaning fails loudly if any
# are missing from the incoming DataFrame.
REQUIRED_COLUMNS = (
    "icao24",
    "callsign",
    "first_seen",
    "last_seen",
    "est_departure_airport",
    "est_arrival_airport",
    "direction",
    "airport_icao",
)


class FlightDataCleaner:
    """Stateless cleaner for raw OpenSky flight DataFrames."""

    def __init__(
        self,
        max_stand_distance_m: int | None = None,
    ) -> None:
        self.max_stand_distance_m = (
            max_stand_distance_m
            if max_stand_distance_m is not None
            else alt_settings.pipeline.max_stand_distance_m
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full cleaning pipeline.

        Steps (in order):
            1. Validate schema.
            2. Drop duplicate (icao24, first_seen, direction, airport_icao) rows.
            3. Normalize callsigns -- strip whitespace, uppercase, NaN if empty.
            4. Coerce unix timestamps to tz-aware UTC datetimes.
            5. Drop rows where first_seen/last_seen is NaN.
            6. Drop low-confidence rows based on horizontal-distance fields.
            7. Drop rows with last_seen < first_seen.

        Returns a cleaned copy; the input is not mutated.
        """
        if df.empty:
            logger.warning("FlightDataCleaner received empty DataFrame")
            return df.copy()

        self._validate_schema(df)
        original_len = len(df)
        result = df.copy()

        # --- dedup -----------------------------------------------------
        dup_keys = ["icao24", "first_seen", "direction", "airport_icao"]
        dup_mask = result.duplicated(subset=dup_keys, keep="last")
        if dup_mask.any():
            n = int(dup_mask.sum())
            logger.info("Removing {} duplicate flight rows", n)
            result = result[~dup_mask]

        # --- normalize callsign --------------------------------------
        result["callsign"] = (
            result["callsign"]
            .astype("string")
            .str.strip()
            .str.upper()
            .replace({"": pd.NA})
        )

        # --- datetimes ------------------------------------------------
        result["first_seen_dt"] = pd.to_datetime(
            result["first_seen"], unit="s", utc=True, errors="coerce"
        )
        result["last_seen_dt"] = pd.to_datetime(
            result["last_seen"], unit="s", utc=True, errors="coerce"
        )
        bad_ts = result["first_seen_dt"].isna() | result["last_seen_dt"].isna()
        if bad_ts.any():
            logger.info("Dropping {} rows with invalid timestamps", int(bad_ts.sum()))
            result = result[~bad_ts]

        # --- monotonicity --------------------------------------------
        bad_order = result["last_seen_dt"] < result["first_seen_dt"]
        if bad_order.any():
            logger.info(
                "Dropping {} rows where last_seen < first_seen",
                int(bad_order.sum()),
            )
            result = result[~bad_order]

        # --- low-confidence filter -----------------------------------
        result = self._filter_low_confidence(result)

        # --- normalize airport fields -------------------------------
        for col in ("est_departure_airport", "est_arrival_airport", "airport_icao"):
            result[col] = (
                result[col].astype("string").str.strip().str.upper().replace({"": pd.NA})
            )

        result = result.sort_values("first_seen_dt").reset_index(drop=True)

        final_len = len(result)
        logger.info(
            "Cleaning reduced rows from {} to {} (removed {})",
            original_len,
            final_len,
            original_len - final_len,
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_schema(df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"FlightDataCleaner: missing required columns: {missing}"
            )

    def _filter_low_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows whose estimated-airport horizontal distance is too high.

        OpenSky provides ``estDepartureAirportHorizDistance`` and
        ``estArrivalAirportHorizDistance`` (meters from the airport
        reference point).  Large values mean the airport could not be
        confidently inferred.  We keep rows where the field is null
        (the airport was a scheduled known stop) and drop rows where
        the relevant distance exceeds ``max_stand_distance_m``.
        """
        distance_col = {
            "arrival": "est_arrival_airport_horiz_distance",
            "departure": "est_departure_airport_horiz_distance",
        }
        before = len(df)
        masks = []
        for direction, col in distance_col.items():
            if col not in df.columns:
                continue
            mask = (df["direction"] == direction) & (
                df[col].notna() & (df[col] > self.max_stand_distance_m)
            )
            masks.append(mask)

        if not masks:
            return df

        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m

        if combined.any():
            logger.info(
                "Dropping {} low-confidence rows (> {}m from airport)",
                int(combined.sum()),
                self.max_stand_distance_m,
            )
            df = df[~combined]
        logger.debug(
            "Low-confidence filter: {} -> {}", before, len(df),
        )
        return df
