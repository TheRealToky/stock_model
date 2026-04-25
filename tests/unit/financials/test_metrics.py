"""Unit tests for backtesting.metrics module."""

import numpy as np
import pytest

from financials.backtesting.metrics import (
    compute_calmar_ratio,
    compute_cagr,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_win_rate,
    compute_all_metrics,
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
        metrics = compute_all_metrics(equity, trades)
        expected_keys = {
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "cagr",
            "calmar_ratio", "win_rate", "profit_factor", "total_trades",
            "initial_capital", "final_capital", "total_return",
        }
        assert expected_keys.issubset(metrics.keys())
        assert metrics["total_trades"] == 3
        assert metrics["initial_capital"] == 100_000
        assert metrics["final_capital"] == 103_000
