"""Unit tests for features.technical module."""

import numpy as np
import pandas as pd
import pytest

from financials.features.technical import (
    compute_returns,
    compute_log_returns,
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_volatility,
    compute_atr,
    compute_rolling_stats,
    compute_volume_features,
    compute_price_features,
)


@pytest.fixture
def sample_df():
    """Create a sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 1.5)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.5,
            "high": close + abs(np.random.randn(n)) * 2,
            "low": close - abs(np.random.randn(n)) * 2,
            "close": close,
            "volume": np.random.randint(500_000, 2_000_000, n),
        },
        index=dates,
    )


class TestReturns:
    def test_simple_returns_length(self, sample_df):
        ret = compute_returns(sample_df)
        assert len(ret) == len(sample_df)
        # First value should be NaN
        assert pd.isna(ret.iloc[0])

    def test_log_returns_length(self, sample_df):
        lr = compute_log_returns(sample_df)
        assert len(lr) == len(sample_df)


class TestMovingAverages:
    def test_sma_length(self, sample_df):
        sma = compute_sma(sample_df, window=20)
        assert len(sma) == len(sample_df)
        # First 19 should be NaN
        assert sma.iloc[:19].isna().all()
        assert not pd.isna(sma.iloc[19])

    def test_ema_length(self, sample_df):
        ema = compute_ema(sample_df, span=20)
        assert len(ema) == len(sample_df)
        # EMA with adjust=False starts from first value
        assert not pd.isna(ema.iloc[0])


class TestRSI:
    def test_rsi_range(self, sample_df):
        rsi = compute_rsi(sample_df, window=14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_length(self, sample_df):
        rsi = compute_rsi(sample_df, window=14)
        assert len(rsi) == len(sample_df)


class TestMACD:
    def test_macd_columns(self, sample_df):
        result = compute_macd(sample_df)
        assert "macd" in result.columns
        assert "signal" in result.columns
        assert "histogram" in result.columns

    def test_macd_length(self, sample_df):
        result = compute_macd(sample_df)
        assert len(result) == len(sample_df)


class TestBollingerBands:
    def test_bb_columns(self, sample_df):
        result = compute_bollinger_bands(sample_df)
        assert "upper" in result.columns
        assert "middle" in result.columns
        assert "lower" in result.columns

    def test_upper_above_lower(self, sample_df):
        result = compute_bollinger_bands(sample_df)
        valid = result.dropna()
        assert (valid["upper"] >= valid["lower"]).all()


class TestVolatility:
    def test_volatility_non_negative(self, sample_df):
        vol = compute_volatility(sample_df)
        valid = vol.dropna()
        assert (valid >= 0).all()


class TestATR:
    def test_atr_non_negative(self, sample_df):
        atr = compute_atr(sample_df)
        valid = atr.dropna()
        assert (valid >= 0).all()


class TestRollingStats:
    def test_rolling_stats_columns(self, sample_df):
        result = compute_rolling_stats(sample_df)
        assert "rolling_mean" in result.columns
        assert "rolling_std" in result.columns


class TestVolumeFeatures:
    def test_volume_features_columns(self, sample_df):
        result = compute_volume_features(sample_df)
        assert "volume_sma" in result.columns
        assert "volume_ratio" in result.columns


class TestPriceFeatures:
    def test_price_features_columns(self, sample_df):
        result = compute_price_features(sample_df)
        assert "high_low_range" in result.columns
        assert "close_open_range" in result.columns
