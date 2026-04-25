"""Unit tests for the DataFetcher orchestration (no real HTTP)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd

from alt_data_pipeline.src.ingestion.fetcher import DataFetcher


class TestFetcher:
    def test_fetch_aggregates_arrivals_and_departures(self, sample_api_payload):
        client = MagicMock()
        client.get_arrivals_by_airport.return_value = sample_api_payload
        client.get_departures_by_airport.return_value = []

        fetcher = DataFetcher(client=client)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 23, 59, 59, tzinfo=timezone.utc)

        df = fetcher.fetch("KJFK", start, end)
        assert len(df) == 2
        assert set(df["direction"]) == {"arrival"}
        assert "first_seen_dt" in df.columns
        assert (df["airport_icao"] == "KJFK").all()

    def test_fetch_splits_range_into_days(self, sample_api_payload):
        client = MagicMock()
        client.get_arrivals_by_airport.return_value = []
        client.get_departures_by_airport.return_value = []

        fetcher = DataFetcher(client=client)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 3, 23, 59, 59, tzinfo=timezone.utc)
        fetcher.fetch("KJFK", start, end)

        assert client.get_arrivals_by_airport.call_count == 3
        assert client.get_departures_by_airport.call_count == 3

    def test_fetch_continues_on_partial_failure(self, sample_api_payload):
        client = MagicMock()

        def arrivals_side_effect(airport, begin, end):
            # Fail the second day only
            if begin >= 1_704_153_600:
                raise RuntimeError("transient failure")
            return sample_api_payload

        client.get_arrivals_by_airport.side_effect = arrivals_side_effect
        client.get_departures_by_airport.return_value = []

        fetcher = DataFetcher(client=client)
        df = fetcher.fetch(
            "KJFK",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, 23, 59, 59, tzinfo=timezone.utc),
        )
        # Day 1 succeeded -> 2 rows
        assert len(df) == 2

    def test_empty_api_returns_empty_df(self):
        client = MagicMock()
        client.get_arrivals_by_airport.return_value = []
        client.get_departures_by_airport.return_value = []

        fetcher = DataFetcher(client=client)
        df = fetcher.fetch(
            "KJFK",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 23, tzinfo=timezone.utc),
        )
        assert df.empty
