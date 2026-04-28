"""Transformer protocol -- a pure function from OHLCV → feature frame."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Transformer(Protocol):
    """Apply feature engineering to a single OHLCV chunk."""

    feature_columns: list[str]

    def transform(self, ohlcv: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
        """Return ``ohlcv`` augmented with feature columns."""
        ...
