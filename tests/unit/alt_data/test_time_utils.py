"""Unit tests for time helpers."""

from datetime import datetime, timezone

import pytest

from alt_data.utils.time_utils import (
    daily_utc_intervals,
    to_unix_seconds,
    utc_day_bounds,
)


class TestUtcDayBounds:
    def test_returns_midnight_to_235959(self):
        d = datetime(2024, 3, 15, 10, 0, tzinfo=timezone.utc)
        start, end = utc_day_bounds(d)
        assert start == datetime(2024, 3, 15, 0, 0, 0, tzinfo=timezone.utc)
        assert end == datetime(2024, 3, 15, 23, 59, 59, tzinfo=timezone.utc)

    def test_normalizes_non_utc_input(self):
        import datetime as _dt
        tz_plus2 = _dt.timezone(_dt.timedelta(hours=2))
        d = datetime(2024, 3, 15, 1, 0, tzinfo=tz_plus2)   # == 2024-03-14 23:00 UTC
        start, end = utc_day_bounds(d)
        assert start.date() == _dt.date(2024, 3, 14)
        assert end.date() == _dt.date(2024, 3, 14)


class TestDailyUtcIntervals:
    def test_single_day(self):
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc)
        intervals = list(daily_utc_intervals(start, end))
        assert len(intervals) == 1
        assert intervals[0][0] == start
        assert intervals[0][1] == end

    def test_three_day_span(self):
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 3, 23, 59, 59, tzinfo=timezone.utc)
        intervals = list(daily_utc_intervals(start, end))
        assert len(intervals) == 3
        # Middle interval is a full UTC day
        assert intervals[1][0].time().hour == 0
        assert intervals[1][1].time().hour == 23

    def test_end_before_start_raises(self):
        start = datetime(2024, 1, 5, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            list(daily_utc_intervals(start, end))

    def test_naive_inputs_assumed_utc(self):
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 1, 12)
        intervals = list(daily_utc_intervals(start, end))
        assert intervals[0][0].tzinfo is timezone.utc
        assert intervals[0][1].tzinfo is timezone.utc


class TestToUnixSeconds:
    def test_roundtrip(self):
        dt = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        assert to_unix_seconds(dt) == 1_704_067_200

    def test_naive_input(self):
        dt = datetime(2024, 1, 1)
        assert to_unix_seconds(dt) == 1_704_067_200
