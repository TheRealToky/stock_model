"""Tests for research.lib.resample.

The failure mode worth guarding: clock-anchored bins. US regular hours run
14:30-21:00 UTC, so a naive ``df.resample("1h")`` does not merge two sessions
-- but it buckets from 14:00, leaving each session's opening "hour" only 30
minutes long and every later bin offset from the session structure. Bar n then
means something different from bar n the next day. Those bars look perfectly
ordinary downstream.
"""

import numpy as np
import pandas as pd
import pytest

from research.lib import resample as R

BARS_PER_SESSION = 390  # US regular trading hours, 09:30-16:00 ET


def session_frame(day: str, bars: int = BARS_PER_SESSION, base: float = 100.0):
    """One session of 1-minute OHLCV bars starting at 14:30 UTC."""
    index = pd.date_range(f"{day} 14:30", periods=bars, freq="1min", tz="UTC")
    close = base + np.arange(bars) * 0.01
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close + 0.01,
            "volume": np.full(bars, 1000.0),
        },
        index=index,
    )


def two_sessions(bars: int = BARS_PER_SESSION):
    return pd.concat(
        [
            session_frame("2024-01-02", bars, base=100.0),
            session_frame("2024-01-03", bars, base=150.0),
        ]
    )


class TestBarsPerPeriod:
    @pytest.mark.parametrize(
        "interval,expected",
        [("1min", 1), ("5min", 5), ("15min", 15), ("30min", 30), ("1d", 390)],
    )
    def test_known_conversions(self, interval, expected):
        assert R.bars_per_period(interval) == expected

    def test_upsampling_is_refused(self):
        with pytest.raises(ValueError, match="Cannot upsample"):
            R.bars_per_period("1min", source_interval="1h")


class TestResampleOhlcv:
    def test_ohlcv_aggregation_is_exact(self):
        df = two_sessions()
        out = R.resample_ohlcv(df, "15min")
        first = df.iloc[:15]
        assert out["open"].iloc[0] == pytest.approx(first["open"].iloc[0])
        assert out["high"].iloc[0] == pytest.approx(first["high"].max())
        assert out["low"].iloc[0] == pytest.approx(first["low"].min())
        assert out["close"].iloc[0] == pytest.approx(first["close"].iloc[-1])
        assert out["volume"].iloc[0] == pytest.approx(first["volume"].sum())

    @pytest.mark.parametrize(
        "interval,expected_bars", [("5min", 78 * 2), ("15min", 26 * 2), ("1d", 2)]
    )
    def test_bar_counts_for_evenly_dividing_intervals(self, interval, expected_bars):
        out = R.resample_ohlcv(two_sessions(), interval)
        assert len(out) == expected_bars

    def test_no_bin_spans_the_overnight_gap(self):
        """The core guarantee."""
        out = R.resample_ohlcv(two_sessions(), "1h")
        assert (out["n_bars"] <= 60).all()
        # Every aggregated bar must fall inside exactly one source session.
        assert out.index.normalize().nunique() == 2

    def test_each_session_opens_a_fresh_bin(self):
        out = R.resample_ohlcv(two_sessions(), "1h")
        opens = out.groupby(out.index.normalize()).head(1).index
        assert set(opens.strftime("%H:%M")) == {"14:30"}

    def test_hourly_session_yields_six_full_bins_and_one_half(self):
        """390 minutes is 6.5 hours -- the tail bin is real, not a bug."""
        out = R.resample_ohlcv(session_frame("2024-01-02"), "1h")
        assert len(out) == 7
        assert list(out["n_bars"]) == [60] * 6 + [30]

    def test_drop_partial_bins_removes_the_ragged_tail(self):
        out = R.resample_ohlcv(session_frame("2024-01-02"), "1h", drop_partial_bins=True)
        assert len(out) == 6
        assert (out["n_bars"] == 60).all()

    def test_one_minute_is_a_passthrough(self):
        df = session_frame("2024-01-02", bars=10)
        out = R.resample_ohlcv(df, "1min")
        assert len(out) == len(df)
        assert (out["n_bars"] == 1).all()
        assert np.allclose(out["close"], df["close"])

    def test_session_unaware_mode_misaligns_bins_to_the_clock(self):
        """Contrast case -- shows the session anchoring is doing real work.

        For US RTH in UTC a clock-floored bin does not actually merge two
        sessions; what it does is start bucketing at 14:00 so the session's
        first bin holds only 30 of its 60 minutes, and every later bin is
        offset from the session structure.
        """
        df = two_sessions()
        aware = R.resample_ohlcv(df, "1h", session_aware=True)
        naive = R.resample_ohlcv(df, "1h", session_aware=False)

        def first_bin_times(frame):
            return set(
                frame.groupby(frame.index.normalize()).head(1).index.strftime("%H:%M")
            )

        assert first_bin_times(aware) == {"14:30"}
        assert first_bin_times(naive) == {"14:00"}
        # Session-anchored bins are full hours except one tail bin per session.
        assert sorted(aware["n_bars"].unique()) == [30, 60]
        assert naive["n_bars"].iloc[0] == 30, "clock bucketing truncates the opening hour"

    def test_clock_bucketing_splits_a_session_that_crosses_utc_midnight(self):
        """Where clock bucketing genuinely destroys session structure."""
        index = pd.date_range("2024-01-02 23:00", periods=180, freq="1min", tz="UTC")
        close = 100 + np.arange(180) * 0.01
        df = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close,
             "volume": np.full(180, 1.0)},
            index=index,
        )
        aware = R.resample_ohlcv(df, "1d", session_aware=True)
        naive = R.resample_ohlcv(df, "1d", session_aware=False)
        assert len(naive) == 2, "one session torn across two daily bars"
        assert len(aware) == 2, "UTC date still defines the session boundary here"
        # The anchored version keeps each piece aligned to its own first bar.
        assert aware.index[0] == index[0]

    def test_extra_numeric_columns_are_carried_as_last(self):
        df = two_sessions(bars=30)
        df["rsi"] = np.arange(len(df), dtype=float)
        out = R.resample_ohlcv(df, "15min")
        assert "rsi" in out.columns
        assert out["rsi"].iloc[0] == pytest.approx(14.0)

    def test_min_bars_filter(self):
        out = R.resample_ohlcv(session_frame("2024-01-02"), "1h", min_bars=45)
        assert (out["n_bars"] >= 45).all()
        assert len(out) == 6

    def test_volume_is_conserved(self):
        df = two_sessions()
        out = R.resample_ohlcv(df, "15min")
        assert out["volume"].sum() == pytest.approx(df["volume"].sum())

    def test_result_is_sorted(self):
        out = R.resample_ohlcv(two_sessions(), "5min")
        assert out.index.is_monotonic_increasing

    def test_empty_frame_round_trips(self):
        empty = two_sessions(bars=5).iloc[:0]
        assert R.resample_ohlcv(empty, "5min").empty

    def test_missing_ohlc_raises(self):
        df = two_sessions(bars=10)[["close"]]
        with pytest.raises(ValueError, match="Missing required OHLC columns"):
            R.resample_ohlcv(df, "5min")

    def test_non_datetime_index_raises(self):
        df = two_sessions(bars=10).reset_index(drop=True)
        with pytest.raises(ValueError, match="requires a DatetimeIndex"):
            R.resample_ohlcv(df, "5min")

    def test_unsupported_interval_raises(self):
        with pytest.raises(ValueError, match="Unknown interval"):
            R.resample_ohlcv(two_sessions(bars=10), "3fortnights")

    def test_weekly_is_known_but_unsupported_here(self):
        with pytest.raises(ValueError, match="not in"):
            R.resample_ohlcv(two_sessions(bars=10), "1wk")


class TestAnnualisationWiring:
    """Resampled bars must carry the right annualisation factor downstream."""

    @pytest.mark.parametrize(
        "interval,expected",
        [("1min", 98_280), ("5min", 19_656), ("15min", 6_552), ("1h", 1_638), ("1d", 252)],
    )
    def test_periods_per_year_matches_the_backtesting_constants(self, interval, expected):
        assert R.periods_per_year(interval) == expected

    def test_bars_per_period_and_periods_per_year_are_consistent(self):
        for interval in R.SUPPORTED_INTERVALS:
            assert (
                R.bars_per_period(interval) * R.periods_per_year(interval)
                == pytest.approx(98_280, rel=0.02)
            )
