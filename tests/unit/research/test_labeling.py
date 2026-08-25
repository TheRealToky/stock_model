"""Tests for research.lib.labeling.

The central risk in this module is a label that quietly encodes the future in
a way the model could never have acted on -- an overnight gap, or a barrier
touched after the close. These pin that behaviour down.
"""

import numpy as np
import pandas as pd
import pytest

from research.lib import labeling as lb


def two_sessions(bars_per_day: int = 5) -> pd.DatetimeIndex:
    """Two consecutive RTH sessions of *bars_per_day* one-minute bars."""
    parts = [
        pd.date_range(f"{day} 14:30", periods=bars_per_day, freq="1min", tz="UTC")
        for day in ("2024-01-02", "2024-01-03")
    ]
    return parts[0].append(parts[1])


def ramp(index: pd.DatetimeIndex, step: float = 0.001, start: float = 100.0):
    """Monotone price ramp with OHLC columns."""
    close = start * (1 + np.arange(len(index)) * step)
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close}, index=index
    )


class TestSessionHelpers:
    def test_session_ids_are_ordinal_per_day(self):
        idx = two_sessions()
        assert list(lb.session_ids(idx)) == [0] * 5 + [1] * 5

    def test_session_ids_rejects_non_datetime_index(self):
        with pytest.raises(TypeError, match="DatetimeIndex"):
            lb.session_ids(pd.Index([1, 2, 3]))

    def test_same_session_forward_excludes_the_overnight_hop(self):
        idx = two_sessions()
        mask = lb.same_session_forward(idx, 1)
        # Last bar of each session has no same-session successor.
        assert mask[4] is np.False_ or not mask[4]
        assert not mask[9]
        assert mask[0] and mask[3]

    def test_same_session_forward_rejects_zero_horizon(self):
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            lb.same_session_forward(two_sessions(), 0)

    def test_session_end_positions(self):
        assert list(lb.session_end_positions(two_sessions())) == [4] * 5 + [9] * 5


class TestForwardReturn:
    def test_masks_cross_session_returns(self):
        idx = two_sessions()
        df = ramp(idx)
        fwd = lb.forward_return(df, 1)
        assert np.isnan(fwd.iloc[4]), "overnight label must be masked"
        assert np.isnan(fwd.iloc[9]), "final bar has no successor"
        assert np.isfinite(fwd.iloc[0])

    def test_unmasked_keeps_the_overnight_return(self):
        df = ramp(two_sessions())
        fwd = lb.forward_return(df, 1, mask_cross_session=False)
        assert np.isfinite(fwd.iloc[4])

    def test_value_matches_manual_computation(self):
        df = ramp(two_sessions())
        fwd = lb.forward_return(df, 2)
        expected = df["close"].iloc[2] / df["close"].iloc[0] - 1.0
        assert abs(fwd.iloc[0] - expected) < 1e-12

    def test_log_returns(self):
        df = ramp(two_sessions())
        fwd = lb.forward_return(df, 1, log=True)
        expected = np.log(df["close"].iloc[1] / df["close"].iloc[0])
        assert abs(fwd.iloc[0] - expected) < 1e-12

    def test_missing_column_raises(self):
        with pytest.raises(KeyError, match="vwap"):
            lb.forward_return(ramp(two_sessions()), 1, price_col="vwap")


class TestFixedHorizonLabel:
    def test_binary_labels_are_zero_one_or_nan(self):
        y = lb.fixed_horizon_label(ramp(two_sessions()), 1)
        assert set(np.unique(y.dropna())) <= {0.0, 1.0}
        assert y.isna().sum() == 2  # one per session boundary

    def test_ternary_threshold_creates_a_dead_zone(self):
        df = ramp(two_sessions(), step=0.0001)  # 1 bp per bar
        y = lb.fixed_horizon_label(df, 1, threshold=0.01, binary=False)
        assert set(np.unique(y.dropna())) == {0.0}, "moves inside the band are no-trade"

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold must be >= 0"):
            lb.fixed_horizon_label(ramp(two_sessions()), 1, threshold=-0.1)


class TestVolatility:
    def test_overnight_gap_is_excluded_from_the_estimate(self):
        idx = two_sessions(bars_per_day=40)
        close = pd.Series(
            np.concatenate([np.full(40, 100.0), np.full(40, 200.0)]), index=idx
        )
        masked = lb.ewm_volatility(close, span=10)
        unmasked = lb.ewm_volatility(close, span=10, mask_cross_session=False)
        # Prices are flat within each session; only the gap has any variance.
        assert np.nanmax(np.nan_to_num(masked.to_numpy(), nan=0.0)) < 1e-12
        assert np.nanmax(unmasked.to_numpy()) > 0.1

    def test_realized_volatility_runs(self):
        idx = two_sessions(bars_per_day=40)
        rng = np.random.default_rng(0)
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.001, 80)), index=idx)
        v = lb.realized_volatility(close, window=20)
        assert v.notna().sum() > 0
        assert (v.dropna() >= 0).all()


class TestTripleBarrier:
    def _long_run(self, n=60):
        return pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")

    def test_monotone_rise_hits_profit_take(self):
        df = ramp(self._long_run(), step=0.001)
        r = lb.triple_barrier_labels(df, horizon_bars=30, pt=1, sl=1, target=0.005)
        assert r.barrier.iloc[0] == "pt"
        assert r.label.iloc[0] == 1.0
        assert r.ret.iloc[0] == pytest.approx(0.005)

    def test_monotone_fall_hits_stop_loss(self):
        df = ramp(self._long_run(), step=-0.001)
        r = lb.triple_barrier_labels(df, horizon_bars=30, pt=1, sl=1, target=0.005)
        assert r.barrier.iloc[0] == "sl"
        assert r.label.iloc[0] == -1.0
        assert r.ret.iloc[0] == pytest.approx(-0.005)

    def test_flat_price_hits_the_vertical_barrier(self):
        idx = self._long_run()
        df = pd.DataFrame(
            {c: np.full(len(idx), 100.0) for c in ("open", "high", "low", "close")},
            index=idx,
        )
        r = lb.triple_barrier_labels(
            df, horizon_bars=10, pt=1, sl=1, target=0.005, zero_on_vertical=True
        )
        assert r.barrier.iloc[0] == "vertical"
        assert r.label.iloc[0] == 0.0
        assert r.holding_bars.iloc[0] == 10

    def test_holding_never_exceeds_the_horizon(self):
        rng = np.random.default_rng(3)
        idx = self._long_run(400)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.002, len(idx)))
        df = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close}, index=idx
        )
        r = lb.triple_barrier_labels(df, horizon_bars=20, pt=2, sl=2, target=0.01)
        assert r.holding_bars.max() <= 20

    def test_label_never_holds_past_the_session_close(self):
        """The barrier must stop at 16:00 -- holding overnight is another trade."""
        idx = two_sessions(bars_per_day=30)
        rng = np.random.default_rng(1)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.0005, len(idx)))
        df = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close}, index=idx
        )
        r = lb.triple_barrier_labels(
            df, horizon_bars=50, pt=5, sl=5, target=0.01, stop_at_session_end=True
        )
        ends = lb.session_end_positions(idx)
        live = r.touch_idx[r.touch_idx >= 0]
        for pos, touch in zip(np.flatnonzero(r.touch_idx.to_numpy() >= 0), live):
            assert touch <= ends[pos], "barrier ran past the session close"

    def test_intrabar_high_low_touch_is_detected(self):
        """With high/low supplied, a wick that pierces the barrier counts."""
        idx = self._long_run(10)
        close = np.full(10, 100.0)
        high = close.copy()
        high[3] = 101.0  # wick up, close unchanged
        df = pd.DataFrame(
            {"open": close, "high": high, "low": close, "close": close}, index=idx
        )
        wick = lb.triple_barrier_labels(
            df, horizon_bars=8, pt=1, sl=1, target=0.005,
            high_col="high", low_col="low",
        )
        close_only = lb.triple_barrier_labels(
            df, horizon_bars=8, pt=1, sl=1, target=0.005
        )
        assert wick.barrier.iloc[0] == "pt"
        assert close_only.barrier.iloc[0] == "vertical"

    def test_both_barriers_disabled_raises(self):
        with pytest.raises(ValueError, match="At least one of pt/sl"):
            lb.triple_barrier_labels(
                ramp(self._long_run()), horizon_bars=5, pt=0, sl=0, target=0.01
            )

    def test_zero_horizon_raises(self):
        with pytest.raises(ValueError, match="horizon_bars must be >= 1"):
            lb.triple_barrier_labels(ramp(self._long_run()), horizon_bars=0)

    def test_misaligned_target_raises(self):
        df = ramp(self._long_run())
        bad = pd.Series([0.01] * 3)
        with pytest.raises(ValueError, match="target must share"):
            lb.triple_barrier_labels(df, horizon_bars=5, target=bad)

    def test_to_frame_round_trips_every_field(self):
        df = ramp(self._long_run(), step=0.001)
        out = lb.triple_barrier_labels(
            df, horizon_bars=10, pt=1, sl=1, target=0.005
        ).to_frame()
        assert {"label", "ret", "touch_idx", "barrier", "holding_bars"} <= set(out.columns)
        assert len(out) == len(df)


class TestMetaLabels:
    def _frame(self, n=40):
        idx = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
        return ramp(idx, step=0.001)

    def test_correct_side_is_labelled_profitable(self):
        df = self._frame()
        side = pd.Series(1.0, index=df.index)  # long into a rising market
        r = lb.meta_labels(df, side, horizon_bars=20, pt=1, sl=1, target=0.005)
        assert r.label.iloc[0] == 1.0
        assert set(np.unique(r.label.dropna())) <= {0.0, 1.0}

    def test_wrong_side_is_labelled_unprofitable(self):
        df = self._frame()
        side = pd.Series(-1.0, index=df.index)  # short into a rising market
        r = lb.meta_labels(df, side, horizon_bars=20, pt=1, sl=1, target=0.005)
        assert r.label.iloc[0] == 0.0

    def test_flat_side_produces_no_label(self):
        df = self._frame()
        side = pd.Series(0.0, index=df.index)
        r = lb.meta_labels(df, side, horizon_bars=10, pt=1, sl=1, target=0.005)
        assert r.label.isna().all(), "no signal means no trade to evaluate"

    def test_misaligned_side_raises(self):
        df = self._frame()
        with pytest.raises(ValueError, match="side must share"):
            lb.meta_labels(df, pd.Series([1.0, -1.0]), horizon_bars=5)


class TestDailyAndCoarserBars:
    """Session masking must not erase the series when each bar IS a session.

    On daily bars every consecutive pair belongs to a different session, so a
    naive "same session?" mask is false everywhere -- which would NaN out every
    label and silently make the daily horizon unusable.
    """

    def daily(self, n: int = 300) -> pd.DataFrame:
        idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
        close = 100 * np.cumprod(
            1 + np.random.default_rng(0).normal(0.0003, 0.012, n)
        )
        return pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close},
            index=idx,
        )

    def test_is_intraday_distinguishes_the_two_regimes(self):
        assert lb.is_intraday(two_sessions()) is True
        assert lb.is_intraday(self.daily().index) is False

    def test_forward_return_survives_on_daily_bars(self):
        fwd = lb.forward_return(self.daily(), 1)
        assert fwd.notna().sum() == 299, "daily forward returns must not be masked away"

    def test_volatility_is_finite_on_daily_bars(self):
        vol = lb.ewm_volatility(self.daily()["close"], span=100)
        assert vol.notna().sum() > 100
        assert (vol.dropna() > 0).all()

    def test_session_end_does_not_cap_daily_barriers_at_zero(self):
        idx = self.daily().index
        assert list(lb.session_end_positions(idx)[:3]) == [len(idx) - 1] * 3

    def test_triple_barrier_produces_labels_on_daily_bars(self):
        df = self.daily()
        vol = lb.ewm_volatility(df["close"], span=100)
        r = lb.triple_barrier_labels(
            df, horizon_bars=5, pt=2, sl=2, target=vol,
            high_col="high", low_col="low",
        )
        assert r.label.notna().sum() > 100, "daily triple-barrier labels vanished"
        assert set(np.unique(r.barrier[r.barrier.notna()])) <= {"pt", "sl", "vertical"}

    def test_fixed_horizon_label_works_on_daily_bars(self):
        y = lb.fixed_horizon_label(self.daily(), 5)
        assert y.notna().sum() > 250
