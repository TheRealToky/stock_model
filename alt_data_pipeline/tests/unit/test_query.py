"""Unit tests for the cross-dataset query layer.

The DataHub tests are split into:
 - pure helpers (time_align) which have no DB dependency
 - AssetMapper behavior when mappings are injected
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from alt_data_pipeline.src.query.mappings import AssetMapper
from alt_data_pipeline.src.query.time_align import (
    align_on_trading_day,
    resample_daily,
)


class TestAssetMapper:
    def test_tickers_for_airport(self):
        m = AssetMapper(airport_to_tickers={"KJFK": ["DAL", "UAL"]})
        assert m.tickers_for_airport("KJFK") == ["DAL", "UAL"]
        assert m.tickers_for_airport("kjfk") == ["DAL", "UAL"]
        assert m.tickers_for_airport("KZZZ") == []

    def test_airports_for_ticker_reverse_lookup(self):
        m = AssetMapper(
            airport_to_tickers={
                "KJFK": ["DAL", "UAL"],
                "KLAX": ["UAL"],
            }
        )
        assert m.airports_for_ticker("UAL") == ["KJFK", "KLAX"]

    def test_region_lookup(self):
        m = AssetMapper(
            airport_to_tickers={"KJFK": ["DAL"]},
            airport_to_region={"KJFK": "US-NE"},
        )
        assert m.region_for_airport("KJFK") == "US-NE"
        assert m.region_for_airport("KZZZ") is None

    def test_route_lookup(self):
        m = AssetMapper(
            airport_to_tickers={},
            route_to_tickers={"KJFK->EGLL": ["DAL"]},
        )
        assert m.tickers_for_route("KJFK", "EGLL") == ["DAL"]


class TestResampleDaily:
    def test_sum_collapses_intra_day_rows(self):
        idx = pd.DatetimeIndex(
            [
                "2024-01-01 10:00",
                "2024-01-01 11:00",
                "2024-01-02 09:00",
            ],
            tz="UTC",
        )
        df = pd.DataFrame({"flights": [3, 2, 5]}, index=idx)
        out = resample_daily(df, how="sum")
        assert out.loc[pd.Timestamp("2024-01-01", tz="UTC"), "flights"] == 5
        assert out.loc[pd.Timestamp("2024-01-02", tz="UTC"), "flights"] == 5

    def test_empty_returns_empty(self):
        assert resample_daily(pd.DataFrame()).empty

    def test_naive_index_is_localized(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="1h")
        df = pd.DataFrame({"v": [1, 1, 1]}, index=idx)
        out = resample_daily(df)
        assert str(out.index.tz) == "UTC"


class TestAlignOnTradingDay:
    def _flight_df(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
        return pd.DataFrame({"net_flow": [1, 2, 3, 4, 5]}, index=idx)

    def _ohlcv_df(self):
        # Skip a weekend to simulate a market calendar
        idx = pd.DatetimeIndex(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            tz="UTC",
        )
        return pd.DataFrame({"close": [100, 101, 99, 102, 103]}, index=idx)

    def test_inner_join_produces_intersection(self):
        out = align_on_trading_day(self._flight_df(), self._ohlcv_df(), how="inner")
        assert len(out) == 5
        assert "net_flow" in out.columns
        assert "close" in out.columns

    def test_shift_days_applies_lag(self):
        out = align_on_trading_day(
            self._flight_df(), self._ohlcv_df(), shift_days=1
        )
        # shifting +1 day drops the first flight row
        assert pd.isna(out.iloc[0]["net_flow"]) or len(out) == 4

    def test_empty_sides_produce_empty(self):
        assert align_on_trading_day(pd.DataFrame(), self._ohlcv_df()).empty
        assert align_on_trading_day(self._flight_df(), pd.DataFrame()).empty
