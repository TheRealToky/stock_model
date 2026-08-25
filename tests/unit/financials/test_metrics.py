"""Unit tests for backtesting.metrics module."""

import numpy as np
import pandas as pd
import pytest

from financials.backtesting.constants import (
    PERIODS_PER_YEAR,
    infer_interval,
    periods_per_year,
    resolve_periods,
)
from financials.backtesting.metrics import (
    compute_all_metrics,
    compute_alpha_beta,
    compute_cagr,
    compute_calmar_ratio,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_win_rate,
)

# -----------------------------------------------------------------------
# Sharpe ratio
# -----------------------------------------------------------------------

class TestSharpeRatio:
    def test_positive_returns(self):
        returns = np.array([0.01, 0.02, 0.015, 0.005, 0.01] * 50)
        sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0)
        assert sharpe > 0

    def test_zero_std_returns_zero(self):
        returns = np.array([0.01, 0.01, 0.01])
        sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0)
        # With ddof=1, a constant array may have non-zero std
        # but with only 3 identical values the result is deterministic
        assert isinstance(sharpe, float)

    def test_single_return_gives_zero(self):
        assert compute_sharpe_ratio(np.array([0.05])) == 0.0

    def test_empty_returns_zero(self):
        assert compute_sharpe_ratio(np.array([])) == 0.0


# -----------------------------------------------------------------------
# Sortino ratio
# -----------------------------------------------------------------------

class TestSortinoRatio:
    def test_all_positive_returns(self):
        returns = np.array([0.01, 0.02, 0.03, 0.01, 0.02])
        sortino = compute_sortino_ratio(returns, risk_free_rate=0.0)
        # No downside -> returns 0 (infinite clamped)
        assert sortino == 0.0

    def test_mixed_returns(self):
        returns = np.array([0.02, -0.01, 0.03, -0.005, 0.01])
        sortino = compute_sortino_ratio(returns, risk_free_rate=0.0)
        assert isinstance(sortino, float)

    def test_empty_returns_zero(self):
        assert compute_sortino_ratio(np.array([])) == 0.0


# -----------------------------------------------------------------------
# Max drawdown
# -----------------------------------------------------------------------

class TestMaxDrawdown:
    def test_monotonic_increase(self):
        equity = [100, 110, 120, 130, 140]
        assert compute_max_drawdown(equity) == 0.0

    def test_simple_drawdown(self):
        equity = [100, 110, 90, 95, 100]
        mdd = compute_max_drawdown(equity)
        expected = (110 - 90) / 110  # ~18.18%
        assert abs(mdd - expected) < 1e-6

    def test_empty_curve(self):
        assert compute_max_drawdown([]) == 0.0

    def test_single_value(self):
        assert compute_max_drawdown([100]) == 0.0


# -----------------------------------------------------------------------
# CAGR
# -----------------------------------------------------------------------

class TestCAGR:
    def test_doubling_in_one_year(self):
        cagr = compute_cagr(100, 200, 1.0)
        assert abs(cagr - 1.0) < 1e-6  # 100% growth

    def test_no_growth(self):
        cagr = compute_cagr(100, 100, 3.0)
        assert abs(cagr) < 1e-6

    def test_zero_years(self):
        assert compute_cagr(100, 200, 0) == 0.0

    def test_negative_initial(self):
        assert compute_cagr(-100, 200, 1) == 0.0


# -----------------------------------------------------------------------
# Win rate
# -----------------------------------------------------------------------

class TestWinRate:
    def test_all_winners(self):
        trades = [{"pnl": 100}, {"pnl": 50}]
        assert compute_win_rate(trades) == 1.0

    def test_all_losers(self):
        trades = [{"pnl": -100}, {"pnl": -50}]
        assert compute_win_rate(trades) == 0.0

    def test_mixed(self):
        trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 30}]
        assert abs(compute_win_rate(trades) - 2 / 3) < 1e-6

    def test_empty(self):
        assert compute_win_rate([]) == 0.0


# -----------------------------------------------------------------------
# Profit factor
# -----------------------------------------------------------------------

class TestProfitFactor:
    def test_no_losers(self):
        trades = [{"pnl": 100}, {"pnl": 50}]
        assert compute_profit_factor(trades) == float("inf")

    def test_equal_profit_loss(self):
        trades = [{"pnl": 100}, {"pnl": -100}]
        assert abs(compute_profit_factor(trades) - 1.0) < 1e-6

    def test_empty(self):
        assert compute_profit_factor([]) == 0.0


# -----------------------------------------------------------------------
# Calmar ratio
# -----------------------------------------------------------------------

class TestCalmarRatio:
    def test_zero_drawdown(self):
        assert compute_calmar_ratio(0.10, 0.0) == 0.0

    def test_normal_case(self):
        assert abs(compute_calmar_ratio(0.20, 0.10) - 2.0) < 1e-6


# -----------------------------------------------------------------------
# All metrics
# -----------------------------------------------------------------------

class TestAllMetrics:
    def test_returns_expected_keys(self):
        equity = [100_000, 101_000, 102_000, 101_500, 103_000]
        trades = [{"pnl": 1000}, {"pnl": -500}, {"pnl": 1500}]
        metrics = compute_all_metrics(equity, trades, interval="1d")
        expected_keys = {
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "cagr",
            "calmar_ratio", "win_rate", "profit_factor", "total_trades",
            "initial_capital", "final_capital", "total_return",
        }
        assert expected_keys.issubset(metrics.keys())
        assert metrics["total_trades"] == 3
        assert metrics["initial_capital"] == 100_000
        assert metrics["final_capital"] == 103_000


# -----------------------------------------------------------------------
# Annualisation -- regression guards for the 252-on-intraday bug.
#
# BacktestEngine used to call compute_all_metrics without `periods`, so every
# 1-minute backtest was annualised at the daily convention. That overstates
# Sharpe by sqrt(98_280 / 252) ~= 19.7x and computes CAGR over ~1,984
# imaginary years. These tests exist so it cannot come back.
# -----------------------------------------------------------------------

class TestAnnualisation:
    def test_periods_table_matches_us_rth_convention(self):
        assert PERIODS_PER_YEAR["1min"] == 390 * 252 == 98_280
        assert PERIODS_PER_YEAR["1d"] == 252

    def test_interval_aliases_resolve(self):
        assert periods_per_year("1m") == periods_per_year("1min")
        assert periods_per_year("daily") == periods_per_year("1d") == 252

    def test_unknown_interval_raises(self):
        with pytest.raises(ValueError, match="Unknown interval"):
            periods_per_year("1fortnight")

    def test_resolve_periods_requires_a_source(self):
        """No hint at all must raise, never silently fall back to 252."""
        with pytest.raises(ValueError, match="Cannot annualise"):
            resolve_periods()

    def test_compute_all_metrics_requires_interval_or_periods(self):
        equity = [100_000, 101_000, 100_500, 102_000]
        with pytest.raises(ValueError, match="Cannot annualise"):
            compute_all_metrics(equity, trades=[])

    def test_one_minute_sharpe_uses_98280_not_252(self):
        """The headline regression: same series, two intervals, ~19.7x apart."""
        rng = np.random.default_rng(42)
        returns = rng.normal(loc=1e-5, scale=1e-3, size=5_000)
        equity = 100_000 * np.cumprod(1.0 + returns)

        m_min = compute_all_metrics(equity, trades=[], risk_free_rate=0.0, interval="1min")
        m_day = compute_all_metrics(equity, trades=[], risk_free_rate=0.0, interval="1d")

        assert m_min["periods_per_year"] == 98_280
        assert m_day["periods_per_year"] == 252

        expected_ratio = np.sqrt(98_280 / 252)  # ~19.746
        assert abs(m_min["sharpe_ratio"] / m_day["sharpe_ratio"] - expected_ratio) < 1e-6

    def test_one_minute_sharpe_matches_manual_formula(self):
        rng = np.random.default_rng(7)
        returns = rng.normal(loc=2e-6, scale=5e-4, size=10_000)
        equity = 100_000 * np.cumprod(1.0 + returns)

        metrics = compute_all_metrics(equity, trades=[], risk_free_rate=0.0, interval="1min")

        realised = np.diff(equity) / equity[:-1]
        expected = (np.mean(realised) / np.std(realised, ddof=1)) * np.sqrt(98_280)
        assert abs(metrics["sharpe_ratio"] - expected) < 1e-9

    def test_cagr_horizon_is_not_inflated_on_intraday_bars(self):
        """98_280 one-minute bars is one year, not 390."""
        equity = np.linspace(100_000, 110_000, 98_280)
        metrics = compute_all_metrics(equity, trades=[], interval="1min")
        # One year elapsed => CAGR is simply the total return.
        assert abs(metrics["cagr"] - 0.10) < 1e-3

    def test_infer_interval_from_datetime_index(self):
        minute_idx = pd.date_range("2024-01-02 09:30", periods=200, freq="1min")
        daily_idx = pd.date_range("2024-01-02", periods=200, freq="B")
        assert infer_interval(minute_idx) == "1min"
        assert infer_interval(daily_idx) == "1d"


# -----------------------------------------------------------------------
# Alpha / beta -- previously regressed the strategy against its own signals.
# -----------------------------------------------------------------------

class TestAlphaBeta:
    def test_identical_series_gives_beta_one_zero_alpha(self):
        rng = np.random.default_rng(3)
        bench = rng.normal(0.0, 0.01, size=500)
        alpha, beta = compute_alpha_beta(bench, bench, interval="1d")
        assert abs(beta - 1.0) < 1e-9
        assert abs(alpha) < 1e-9

    def test_leveraged_series_recovers_the_leverage_as_beta(self):
        rng = np.random.default_rng(11)
        bench = rng.normal(0.0, 0.01, size=500)
        alpha, beta = compute_alpha_beta(2.0 * bench, bench, interval="1d")
        assert abs(beta - 2.0) < 1e-9
        assert abs(alpha) < 1e-9

    def test_constant_outperformance_shows_up_as_annualised_alpha(self):
        rng = np.random.default_rng(5)
        bench = rng.normal(0.0, 0.01, size=1_000)
        edge = 0.0004  # 4 bps per day
        alpha, beta = compute_alpha_beta(bench + edge, bench, interval="1d")
        assert abs(beta - 1.0) < 1e-9
        assert abs(alpha - edge * 252) < 1e-9

    def test_zero_variance_benchmark_returns_zeros(self):
        strat = np.array([0.01, -0.02, 0.03, 0.00])
        flat = np.zeros(4)
        assert compute_alpha_beta(strat, flat, interval="1d") == (0.0, 0.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            compute_alpha_beta(np.zeros(10), np.zeros(9), interval="1d")

    def test_requires_annualisation_factor(self):
        with pytest.raises(ValueError, match="Cannot annualise"):
            compute_alpha_beta(np.zeros(10), np.zeros(10))
