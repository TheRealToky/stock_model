"""Unit tests for trading strategies."""

import numpy as np
import pandas as pd
import pytest

from strategies.ema_crossover import EMACrossover
from strategies.mean_reversion import MeanReversion
from strategies.momentum import Momentum


def _make_price_df(prices: list[float]) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame from a list of close prices."""
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1_000_000] * len(prices),
        },
        index=dates,
    )


# -----------------------------------------------------------------------
# EMA Crossover
# -----------------------------------------------------------------------

class TestEMACrossover:
    def test_valid_init(self):
        s = EMACrossover(fast_period=5, slow_period=20)
        assert s.fast_period == 5
        assert s.slow_period == 20

    def test_invalid_periods(self):
        with pytest.raises(ValueError):
            EMACrossover(fast_period=20, slow_period=10)

    def test_signal_values(self):
        prices = list(range(100, 200)) + list(range(200, 100, -1))
        df = _make_price_df(prices)
        s = EMACrossover(fast_period=5, slow_period=20)
        signals = s.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})
        assert len(signals) == len(df)

    def test_parameters(self):
        s = EMACrossover(fast_period=10, slow_period=30)
        params = s.get_parameters()
        assert params == {"fast_period": 10, "slow_period": 30}

    def test_dependencies(self):
        s = EMACrossover()
        assert "close" in s.get_dependencies()


# -----------------------------------------------------------------------
# Mean Reversion
# -----------------------------------------------------------------------

class TestMeanReversion:
    def test_valid_init(self):
        s = MeanReversion(lookback=30, entry_std=2.5, exit_std=0.3)
        assert s.lookback == 30

    def test_invalid_lookback(self):
        with pytest.raises(ValueError):
            MeanReversion(lookback=1)

    def test_invalid_exit_greater_than_entry(self):
        with pytest.raises(ValueError):
            MeanReversion(entry_std=2.0, exit_std=2.5)

    def test_signal_values(self):
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 2)
        df = _make_price_df(prices.tolist())
        s = MeanReversion(lookback=20, entry_std=2.0, exit_std=0.5)
        signals = s.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_parameters(self):
        s = MeanReversion(lookback=25, entry_std=1.5, exit_std=0.3)
        params = s.get_parameters()
        assert params["lookback"] == 25
        assert params["entry_std"] == 1.5
        assert params["exit_std"] == 0.3


# -----------------------------------------------------------------------
# Momentum
# -----------------------------------------------------------------------

class TestMomentum:
    def test_valid_init(self):
        s = Momentum(lookback=10, top_pct=0.3)
        assert s.lookback == 10

    def test_invalid_top_pct(self):
        with pytest.raises(ValueError):
            Momentum(top_pct=0.0)
        with pytest.raises(ValueError):
            Momentum(top_pct=1.0)

    def test_signal_values(self):
        prices = list(range(50, 150))
        df = _make_price_df(prices)
        s = Momentum(lookback=10, top_pct=0.2)
        signals = s.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_parameters(self):
        s = Momentum(lookback=15, top_pct=0.25)
        params = s.get_parameters()
        assert params == {"lookback": 15, "top_pct": 0.25}
