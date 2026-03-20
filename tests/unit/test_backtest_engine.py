"""Unit tests for backtesting.engine module."""

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestEngine, BacktestResult
from strategies.ema_crossover import EMA_Crossover
from strategies.momentum import Momentum


def _make_trending_df(n: int = 200) -> pd.DataFrame:
    """Create a DataFrame with a clear uptrend then downtrend."""
    up = np.linspace(100, 150, n // 2)
    down = np.linspace(150, 100, n // 2)
    prices = np.concatenate([up, down])
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": [1_000_000] * n,
        },
        index=dates,
    )


class TestBacktestEngine:
    def test_run_returns_result(self):
        df = _make_trending_df()
        engine = BacktestEngine(initial_capital=100_000, commission=0.001, slippage=0.0)
        strategy = EMA_Crossover(fast_period=5, slow_period=20)
        result = engine.run(strategy, df, ticker="TEST")
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "ema_crossover"
        assert result.ticker == "TEST"
        assert len(result.equity_curve) == len(df)

    def test_metrics_present(self):
        df = _make_trending_df()
        engine = BacktestEngine(initial_capital=100_000)
        strategy = EMA_Crossover(fast_period=5, slow_period=20)
        result = engine.run(strategy, df)
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics
        assert "cagr" in result.metrics
        assert "win_rate" in result.metrics
        assert "total_trades" in result.metrics

    def test_initial_capital_preserved_with_no_signals(self):
        # A short DataFrame with no crossovers
        prices = [100] * 10
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {"open": prices, "high": prices, "low": prices, "close": prices, "volume": [1e6] * 10},
            index=dates,
        )
        engine = BacktestEngine(initial_capital=50_000, commission=0, slippage=0)
        strategy = EMA_Crossover(fast_period=3, slow_period=5)
        result = engine.run(strategy, df)
        # No trades should occur with constant prices
        assert result.equity_curve[-1] == pytest.approx(50_000, abs=1)

    def test_empty_df_raises(self):
        engine = BacktestEngine()
        strategy = EMA_Crossover()
        with pytest.raises(ValueError, match="empty"):
            engine.run(strategy, pd.DataFrame())

    def test_missing_close_raises(self):
        df = pd.DataFrame({"open": [100], "volume": [1e6]})
        engine = BacktestEngine()
        strategy = EMA_Crossover()
        with pytest.raises(ValueError, match="close"):
            engine.run(strategy, df)

    def test_commission_reduces_returns(self):
        df = _make_trending_df()
        no_fee = BacktestEngine(initial_capital=100_000, commission=0, slippage=0)
        with_fee = BacktestEngine(initial_capital=100_000, commission=0.01, slippage=0)
        strategy = EMA_Crossover(fast_period=5, slow_period=20)

        r1 = no_fee.run(strategy, df)
        r2 = with_fee.run(strategy, df)

        # With fees, final capital should be lower
        assert r2.equity_curve[-1] <= r1.equity_curve[-1]
