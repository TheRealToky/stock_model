"""Lightweight data-quality checks for ETL chunks.

These are intentionally cheap so they can run on every chunk without
becoming the bottleneck. They guard against silent corruption (bad joins,
schema drift, NaN-filled feature columns) rather than provide a full
data-quality framework.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class OHLCVValidationError(ValueError):
    """Raised when an OHLCV chunk fails one of the required invariants."""


def validate_ohlcv_chunk(df: pd.DataFrame, *, ticker: str) -> None:
    """Cheap structural checks on a freshly-extracted OHLCV chunk.

    Raises :class:`OHLCVValidationError` on any failure. The caller decides
    whether to skip the chunk or abort the run.
    """
    if df.empty:
        # Empty is legitimate (e.g. no new bars on a holiday).
        return

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise OHLCVValidationError(
            f"[{ticker}] OHLCV chunk missing columns: {missing}"
        )

    # Index must be a sorted, unique, tz-aware DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise OHLCVValidationError(f"[{ticker}] index is not DatetimeIndex")
    if df.index.tz is None:
        raise OHLCVValidationError(f"[{ticker}] index is timezone-naive")
    if not df.index.is_monotonic_increasing:
        raise OHLCVValidationError(f"[{ticker}] index is not sorted ascending")
    if df.index.duplicated().any():
        n_dupes = int(df.index.duplicated().sum())
        raise OHLCVValidationError(f"[{ticker}] {n_dupes} duplicate timestamps")

    # Numeric sanity
    closes = df["close"].astype(float)
    if (closes <= 0).any():
        raise OHLCVValidationError(f"[{ticker}] non-positive close prices detected")

    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    if (highs < lows).any():
        raise OHLCVValidationError(f"[{ticker}] high < low on at least one bar")

    if df["volume"].astype(float).lt(0).any():
        raise OHLCVValidationError(f"[{ticker}] negative volume")


def validate_feature_frame(
    df: pd.DataFrame,
    *,
    expected_features: Iterable[str],
    ticker: str,
    max_nan_fraction: float = 0.5,
) -> None:
    """Confirm computed features are present and not pathologically empty."""
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        raise OHLCVValidationError(
            f"[{ticker}] feature DataFrame missing expected columns: {missing}"
        )

    feature_cols = [c for c in expected_features if c in df.columns]
    if not feature_cols:
        return

    nan_fractions = df[feature_cols].isna().mean()
    bad = nan_fractions[nan_fractions > max_nan_fraction]
    if not bad.empty:
        # Don't raise -- the warm-up region is allowed to be heavy on NaNs.
        # Caller can decide to drop those rows. We just ensure no column is
        # *entirely* NaN, which would indicate a broken indicator.
        all_nan = nan_fractions[nan_fractions == 1.0]
        if not all_nan.empty:
            raise OHLCVValidationError(
                f"[{ticker}] features all-NaN: {list(all_nan.index)}"
            )


def cast_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 columns to float32 in-place (returns same df)."""
    float_cols = df.select_dtypes(include=[np.float64]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype(np.float32)
    return df
