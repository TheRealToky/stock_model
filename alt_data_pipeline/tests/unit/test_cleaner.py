"""Unit tests for FlightDataCleaner."""

import pandas as pd
import pytest

from alt_data_pipeline.src.cleaning.cleaner import FlightDataCleaner


def _raw_df_from_cleaned(cleaned_flight_df):
    """Drop the derived _dt columns so we simulate a fresh raw DataFrame."""
    return cleaned_flight_df.drop(columns=["first_seen_dt", "last_seen_dt"]).copy()


class TestCleaning:
    def test_empty_input_returns_empty(self):
        cleaner = FlightDataCleaner()
        out = cleaner.clean(pd.DataFrame())
        assert out.empty

    def test_missing_required_columns_raises(self):
        bad = pd.DataFrame({"icao24": ["abc"], "first_seen": [1]})
        with pytest.raises(ValueError):
            FlightDataCleaner().clean(bad)

    def test_callsign_normalization(self, cleaned_flight_df):
        raw = _raw_df_from_cleaned(cleaned_flight_df)
        raw.loc[0, "callsign"] = "  ual0  "
        out = FlightDataCleaner().clean(raw)
        assert out.loc[out.index[0], "callsign"] == "UAL0"

    def test_duplicate_rows_are_removed(self, cleaned_flight_df):
        raw = _raw_df_from_cleaned(cleaned_flight_df)
        dupes = pd.concat([raw, raw.iloc[:2]], ignore_index=True)
        out = FlightDataCleaner().clean(dupes)
        assert len(out) == len(raw)

    def test_low_confidence_rows_filtered(self, cleaned_flight_df):
        raw = _raw_df_from_cleaned(cleaned_flight_df)
        raw.loc[0, "est_arrival_airport_horiz_distance"] = 99_999_999
        cleaner = FlightDataCleaner(max_stand_distance_m=1000)
        out = cleaner.clean(raw)
        assert len(out) == len(raw) - 1

    def test_datetime_columns_created(self, cleaned_flight_df):
        raw = _raw_df_from_cleaned(cleaned_flight_df)
        out = FlightDataCleaner().clean(raw)
        assert "first_seen_dt" in out.columns
        assert str(out["first_seen_dt"].dt.tz) == "UTC"

    def test_bad_monotonicity_dropped(self, cleaned_flight_df):
        raw = _raw_df_from_cleaned(cleaned_flight_df)
        raw.loc[0, "last_seen"] = raw.loc[0, "first_seen"] - 1000
        out = FlightDataCleaner().clean(raw)
        # Row 0 should have been dropped
        assert len(out) == len(raw) - 1
