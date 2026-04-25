"""Data cleaning and validation utilities for OHLCV market data."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans, validates, and repairs OHLCV DataFrames."""

    def clean_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full cleaning pipeline to an OHLCV DataFrame.

        Steps:
            1. Remove duplicate index entries (keep last).
            2. Sort by timestamp.
            3. Forward-fill missing values, then drop any remaining NaNs.
            4. Normalize the index timezone to UTC.
            5. Validate OHLCV constraints (high >= low, volume >= 0).

        Args:
            df: DataFrame indexed by timestamp with columns
                ``open, high, low, close, volume`` (and optionally
                ``adjusted_close``).

        Returns:
            Cleaned copy of the DataFrame.
        """
        if df.empty:
            logger.warning("clean_ohlcv received an empty DataFrame")
            return df.copy()

        original_len = len(df)
        result = df.copy()

        # 1. Remove duplicate timestamps
        duplicates = result.index.duplicated(keep="last")
        if duplicates.any():
            n_dupes = duplicates.sum()
            logger.info("Removing %d duplicate timestamps", n_dupes)
            result = result[~duplicates]

        # 2. Sort by timestamp
        result = result.sort_index()

        # 3. Handle missing values: forward-fill then drop remaining NaNs
        n_missing_before = result.isna().sum().sum()
        if n_missing_before > 0:
            result = result.ffill()
            n_remaining = result.isna().sum().sum()
            if n_remaining > 0:
                logger.info(
                    "Dropping %d rows with NaN values that could not be forward-filled",
                    result.isna().any(axis=1).sum(),
                )
                result = result.dropna()
            logger.info(
                "Handled %d missing values (%d forward-filled, %d dropped)",
                n_missing_before,
                n_missing_before - n_remaining,
                n_remaining,
            )

        # 4. Normalize timezone to UTC
        result = self._normalize_timezone(result)

        # 5. Validate OHLCV constraints
        result = self._validate_ohlcv_constraints(result)

        final_len = len(result)
        if final_len < original_len:
            logger.info(
                "Cleaning reduced rows from %d to %d (removed %d)",
                original_len, final_len, original_len - final_len,
            )

        return result

    def detect_outliers(
        self,
        df: pd.DataFrame,
        column: str = "close",
        n_std: float = 3.0,
    ) -> pd.Series:
        """Flag rows where *column* deviates more than *n_std* standard
        deviations from the rolling mean.

        Uses a 20-period rolling window to compute local statistics so that
        long-term trends do not inflate the standard deviation.

        Args:
            df: OHLCV DataFrame.
            column: Column to check.
            n_std: Number of standard deviations for the threshold.

        Returns:
            Boolean Series aligned to *df* where ``True`` marks an outlier.
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        series = df[column]
        rolling_mean = series.rolling(window=20, min_periods=1).mean()
        rolling_std = series.rolling(window=20, min_periods=1).std().fillna(0)

        deviation = (series - rolling_mean).abs()
        is_outlier = deviation > (n_std * rolling_std)

        n_outliers = is_outlier.sum()
        if n_outliers > 0:
            logger.info(
                "Detected %d outliers in '%s' (threshold=%.1f std)",
                n_outliers, column, n_std,
            )

        return is_outlier

    def remove_gaps(
        self,
        df: pd.DataFrame,
        max_gap_days: int = 5,
    ) -> pd.DataFrame:
        """Split the DataFrame at large gaps and keep only the longest
        contiguous segment.

        A *gap* is defined as a stretch of more than *max_gap_days* calendar
        days between consecutive timestamps.

        Args:
            df: OHLCV DataFrame sorted by timestamp.
            max_gap_days: Maximum allowed gap in calendar days.

        Returns:
            The longest contiguous segment of the DataFrame.
        """
        if df.empty or len(df) < 2:
            return df.copy()

        result = df.sort_index().copy()
        time_diffs = result.index.to_series().diff()
        gap_threshold = pd.Timedelta(days=max_gap_days)

        gap_mask = time_diffs > gap_threshold
        n_gaps = gap_mask.sum()

        if n_gaps == 0:
            return result

        logger.info(
            "Found %d gaps exceeding %d days", n_gaps, max_gap_days,
        )

        # Assign each row a segment id, then keep the largest segment.
        segment_ids = gap_mask.cumsum()
        segment_sizes = segment_ids.value_counts()
        largest_segment = segment_sizes.idxmax()

        result = result[segment_ids == largest_segment]
        logger.info(
            "Kept segment with %d rows (dropped %d rows across %d gaps)",
            len(result), len(df) - len(result), n_gaps,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_timezone(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DatetimeIndex is tz-aware and in UTC."""
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df

    @staticmethod
    def _validate_ohlcv_constraints(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows that violate basic OHLCV invariants.

        Rules:
            * ``high >= low``
            * ``volume >= 0``
        """
        invalid_mask = pd.Series(False, index=df.index)

        if "high" in df.columns and "low" in df.columns:
            bad_hl = df["high"] < df["low"]
            if bad_hl.any():
                logger.warning(
                    "Dropping %d rows where high < low", bad_hl.sum(),
                )
                invalid_mask |= bad_hl

        if "volume" in df.columns:
            bad_vol = df["volume"] < 0
            if bad_vol.any():
                logger.warning(
                    "Dropping %d rows where volume < 0", bad_vol.sum(),
                )
                invalid_mask |= bad_vol

        if invalid_mask.any():
            df = df[~invalid_mask]

        return df
