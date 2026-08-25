"""Tests for financials.backtesting.ml_adapter.

The adapter is what turns "53% accurate" into "loses 94% of capital at retail
costs". Two properties carry most of the weight:

* positions must be shifted before they earn, or every result is look-ahead;
* costs must scale with turnover, or high-frequency strategies look free.
"""

import numpy as np
import pandas as pd
import pytest

from financials.backtesting import ml_adapter as A


def bars_frame(n: int = 500, seed: int = 0, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(drift, 3e-4, n))
    return pd.DataFrame(
        {"open": close, "high": close * 1.0002, "low": close * 0.9998,
         "close": close, "volume": np.full(n, 1000.0)},
        index=idx,
    )


class TestProbabilitiesToPositions:
    def test_binary_sizing_is_all_or_nothing(self):
        p = pd.Series([0.1, 0.49, 0.51, 0.9])
        pos = A.probabilities_to_positions(p, threshold=0.5, sizing="binary")
        assert list(pos) == [0.0, 0.0, 1.0, 1.0]

    def test_threshold_is_respected(self):
        p = pd.Series([0.55, 0.65])
        pos = A.probabilities_to_positions(p, threshold=0.6)
        assert list(pos) == [0.0, 1.0]

    def test_proportional_sizing_scales_with_conviction(self):
        p = pd.Series([0.5, 0.75, 1.0])
        pos = A.probabilities_to_positions(p, threshold=0.5, sizing="proportional")
        assert pos.iloc[0] == pytest.approx(0.0)
        assert pos.iloc[1] == pytest.approx(0.5)
        assert pos.iloc[2] == pytest.approx(1.0)

    def test_confidence_sizing_ignores_threshold(self):
        p = pd.Series([0.5, 0.75, 1.0])
        pos = A.probabilities_to_positions(p, sizing="confidence", threshold=0.9)
        assert pos.iloc[0] == pytest.approx(0.0)
        assert pos.iloc[1] == pytest.approx(0.5)
        assert pos.iloc[2] == pytest.approx(1.0)

    def test_long_only_by_default(self):
        pos = A.probabilities_to_positions(pd.Series([0.01, 0.99]))
        assert (pos >= 0).all()

    def test_allow_short_goes_negative(self):
        pos = A.probabilities_to_positions(
            pd.Series([0.05, 0.95]), threshold=0.6, allow_short=True
        )
        assert pos.iloc[0] == -1.0
        assert pos.iloc[1] == 1.0

    def test_max_position_caps_exposure(self):
        pos = A.probabilities_to_positions(pd.Series([0.99]), max_position=0.5)
        assert pos.iloc[0] == pytest.approx(0.5)

    def test_nan_probability_yields_nan_position(self):
        pos = A.probabilities_to_positions(pd.Series([np.nan, 0.9]))
        assert np.isnan(pos.iloc[0])
        assert pos.iloc[1] == 1.0

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError, match="sizing must be one of"):
            A.probabilities_to_positions(pd.Series([0.5]), sizing="kelly")
        with pytest.raises(ValueError, match=r"threshold must be in \(0, 1\)"):
            A.probabilities_to_positions(pd.Series([0.5]), threshold=1.0)
        with pytest.raises(ValueError, match="max_position must be > 0"):
            A.probabilities_to_positions(pd.Series([0.5]), max_position=0.0)


class TestSimulatePositions:
    def test_positions_are_shifted_so_signals_cannot_be_front_run(self):
        """A position known at bar t may only earn from bar t+1.

        ``up_now[t]`` describes the return INTO bar t, i.e. exactly
        ``asset_ret[t]``. Held unshifted it is pure foresight and prints
        money; shifted by one bar it is stale and should not. That gap is the
        look-ahead the default ``shift=1`` exists to close.
        """
        bars = bars_frame(400)
        up_now = (bars["close"] > bars["close"].shift(1)).astype(float)
        kwargs = dict(commission=0, slippage=0, risk_free_rate=0, interval="1min")

        cheat = A.simulate_positions(bars, up_now, shift=0, **kwargs)
        honest = A.simulate_positions(bars, up_now, shift=1, **kwargs)

        assert cheat["equity"].iloc[-1] > 100_000, "unshifted foresight must pay"
        assert honest["equity"].iloc[-1] < 100_000, "shifted, the signal is stale"
        assert cheat["equity"].iloc[-1] > honest["equity"].iloc[-1]

    def test_flat_position_earns_the_risk_free_rate(self):
        bars = bars_frame(200)
        flat = pd.Series(0.0, index=bars.index)
        sim = A.simulate_positions(
            bars, flat, commission=0, slippage=0,
            risk_free_rate=0.04, interval="1min",
        )
        expected = (1 + 0.04 / 98_280) ** len(bars)
        assert sim["equity"].iloc[-1] / 100_000 == pytest.approx(expected, rel=1e-9)

    def test_turnover_is_charged_per_unit_of_exposure_change(self):
        bars = bars_frame(10)
        alternating = pd.Series([0.0, 1.0] * 5, index=bars.index)
        sim = A.simulate_positions(
            bars, alternating, commission=0.001, slippage=0.0005,
            risk_free_rate=0, interval="1min",
        )
        # 10 alternating targets, but shift(1) consumes the first flip at the
        # boundary, leaving 8 exposure changes of magnitude 1.
        assert sim["turnover"].sum() == pytest.approx(8.0)

    def test_holding_costs_nothing(self):
        bars = bars_frame(100)
        held = pd.Series(1.0, index=bars.index)
        sim = A.simulate_positions(
            bars, held, commission=0.01, slippage=0.01,
            risk_free_rate=0, interval="1min",
        )
        assert sim["turnover"].sum() == pytest.approx(1.0), "one entry, then free"

    def test_matches_reslib_simulate_exposure_exactly(self):
        """Chains this simulator to the engine reslib was validated against."""
        reslib = pytest.importorskip("research.lib.reslib")
        rng = np.random.default_rng(0)
        bars = bars_frame(2000)
        exposure = pd.Series(rng.random(len(bars)).round(1), index=bars.index)

        mine = A.simulate_positions(
            bars, exposure, commission=0.001, slippage=0.0005,
            interval="1min", risk_free_rate=0.04,
        )
        theirs = reslib.simulate_exposure(
            bars, exposure, commission=0.001, slippage=0.0005,
            rf=0.04, periods=reslib.PERIODS_1MIN,
        )
        assert np.allclose(
            mine["equity"].to_numpy(), theirs["equity"].to_numpy(), rtol=0, atol=1e-9
        )

    def test_invalid_arguments_raise(self):
        bars = bars_frame(10)
        pos = pd.Series(1.0, index=bars.index)
        with pytest.raises(ValueError, match="shift must be >= 0"):
            A.simulate_positions(bars, pos, shift=-1, interval="1min")
        with pytest.raises(ValueError, match="non-negative"):
            A.simulate_positions(bars, pos, commission=-0.1, interval="1min")
        with pytest.raises(ValueError, match="must share the bars index"):
            A.simulate_positions(bars, pd.Series([1.0, 0.0]), interval="1min")


class TestBacktestPredictions:
    def test_annualisation_follows_the_interval(self):
        bars = bars_frame(1000)
        proba = pd.Series(0.9, index=bars.index)
        minute = A.backtest_predictions(bars, proba, interval="1min")
        daily = A.backtest_predictions(bars, proba, interval="1d")
        assert minute.strategy["periods_per_year"] == 98_280
        assert daily.strategy["periods_per_year"] == 252
        assert minute.strategy["sharpe_ratio"] != daily.strategy["sharpe_ratio"]

    def test_interval_is_inferred_from_the_index(self):
        bars = bars_frame(500)
        result = A.backtest_predictions(bars, pd.Series(0.9, index=bars.index))
        assert result.strategy["periods_per_year"] == 98_280

    def test_always_long_reproduces_the_benchmark(self):
        bars = bars_frame(500)
        always = pd.Series(1.0, index=bars.index)
        result = A.backtest_predictions(bars, always, interval="1min", threshold=0.5)
        assert result.strategy["sharpe_ratio"] == pytest.approx(
            result.benchmark["sharpe_ratio"]
        )
        assert result.sharpe_edge == pytest.approx(0.0)

    def test_never_long_holds_no_exposure(self):
        bars = bars_frame(300)
        never = pd.Series(0.0, index=bars.index)
        result = A.backtest_predictions(bars, never, interval="1min")
        assert result.strategy["exposure"] == pytest.approx(0.0)
        assert result.strategy["n_entries"] == 0

    def test_higher_costs_never_help(self):
        rng = np.random.default_rng(1)
        bars = bars_frame(2000)
        proba = pd.Series(rng.random(len(bars)), index=bars.index)
        cheap = A.backtest_predictions(bars, proba, interval="1min",
                                       commission=0.0, slippage=0.0)
        dear = A.backtest_predictions(bars, proba, interval="1min",
                                      commission=0.001, slippage=0.0005)
        assert dear.strategy["sharpe_ratio"] < cheap.strategy["sharpe_ratio"]

    def test_beats_benchmark_requires_drawdown_too(self):
        bars = bars_frame(500)
        result = A.backtest_predictions(bars, pd.Series(1.0, index=bars.index),
                                        interval="1min")
        # Identical to the benchmark: edge is zero, so it does not "beat" it.
        assert result.beats_benchmark is False

    def test_row_is_flat_and_json_friendly(self):
        bars = bars_frame(300)
        row = A.backtest_predictions(
            bars, pd.Series(0.8, index=bars.index), interval="1min",
            meta={"symbol": "TEST"},
        ).row()
        assert row["symbol"] == "TEST"
        assert "strat_sharpe_ratio" in row and "bh_sharpe_ratio" in row
        assert isinstance(row["sharpe_edge"], float)
        assert all(not isinstance(v, (list, dict)) for v in row.values())

    def test_misaligned_probabilities_raise(self):
        bars = bars_frame(100)
        with pytest.raises(ValueError, match="must share the bars index"):
            A.backtest_predictions(bars, pd.Series([0.5, 0.5]), interval="1min")


class TestCostSweep:
    def test_covers_every_threshold_and_cost_pair(self):
        bars = bars_frame(600)
        rng = np.random.default_rng(2)
        proba = pd.Series(rng.random(len(bars)), index=bars.index)
        table = A.cost_sweep(bars, proba, interval="1min", thresholds=(0.5, 0.6))
        assert len(table) == 2 * len(A.DEFAULT_COST_GRID)
        assert set(table["cost_label"]) == {c[2] for c in A.DEFAULT_COST_GRID}

    def test_sharpe_decreases_as_friction_rises(self):
        bars = bars_frame(3000)
        rng = np.random.default_rng(3)
        proba = pd.Series(rng.random(len(bars)), index=bars.index)
        table = A.cost_sweep(bars, proba, interval="1min", thresholds=(0.5,))
        ordered = table.sort_values("round_trip_bps")["strat_sharpe_ratio"].to_numpy()
        assert np.all(np.diff(ordered) <= 1e-9), "more cost must never help"


class TestTuneThreshold:
    def test_returns_a_candidate_and_a_full_table(self):
        bars = bars_frame(1000, drift=2e-5)
        rng = np.random.default_rng(4)
        proba = pd.Series(rng.random(len(bars)), index=bars.index)
        best, table = A.tune_threshold(
            bars, proba, interval="1min", candidates=(0.5, 0.55, 0.6)
        )
        assert best in (0.5, 0.55, 0.6)
        assert len(table) == 3

    def test_objective_choice_changes_the_ranking_key(self):
        bars = bars_frame(800)
        rng = np.random.default_rng(5)
        proba = pd.Series(rng.random(len(bars)), index=bars.index)
        _, by_edge = A.tune_threshold(bars, proba, interval="1min", objective="sharpe_edge")
        _, by_sharpe = A.tune_threshold(bars, proba, interval="1min", objective="sharpe")
        assert "sharpe_edge" in by_edge.columns
        assert "strat_sharpe_ratio" in by_sharpe.columns

    def test_invalid_arguments_raise(self):
        bars = bars_frame(100)
        proba = pd.Series(0.5, index=bars.index)
        with pytest.raises(ValueError, match="objective must be"):
            A.tune_threshold(bars, proba, interval="1min", objective="calmar")
        with pytest.raises(ValueError, match="must not be empty"):
            A.tune_threshold(bars, proba, interval="1min", candidates=())
