"""Loader protocol -- accepts feature frames, writes them somewhere durable."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Loader(Protocol):
    """Sink for feature frames produced by the transform layer."""

    def write(self, df: pd.DataFrame, *, ticker: str) -> int:
        """Persist *df* and return the number of rows actually written."""
        ...
