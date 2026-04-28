"""Extractor interface used by the orchestrator.

Defining a Protocol lets the pipeline accept any object that yields chunked
OHLCV DataFrames -- handy for unit tests (in-memory fixture extractor) or
swapping the source (e.g. CSV ingestion) without touching the pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator, Protocol

import pandas as pd


class Extractor(Protocol):
    """Iterable source of OHLCV chunks for a single ticker."""

    def list_tickers(self) -> list[str]:
        """Return all tickers eligible for extraction."""
        ...

    def get_latest_timestamp(self, ticker: str) -> datetime | None:
        """Latest available bar timestamp for *ticker*, or None if absent."""
        ...

    def stream_chunks(
        self,
        ticker: str,
        start: datetime,
        end: datetime | None,
    ) -> Iterator[pd.DataFrame]:
        """Yield OHLCV DataFrames covering ``[start, end]`` for *ticker*."""
        ...
