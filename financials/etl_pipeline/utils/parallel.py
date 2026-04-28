"""Helpers for running per-ticker ETL work in a process pool.

Multiprocessing is only worthwhile when the unit of work is meaningfully
CPU-bound. Feature computation (rolling windows over millions of rows) is
exactly that, so we parallelize at the *ticker* level. Each worker opens
its own DB connection -- SQLAlchemy engines are not fork-safe.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

from financials.etl_pipeline.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def map_tickers(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int | None,
) -> list[R]:
    """Apply *func* to each item, optionally in parallel.

    ``max_workers=1`` (or 0) runs sequentially in the caller's process,
    which is essential for debuggers, profilers, and CI environments
    that don't allow fork.
    """
    items = list(items)
    if not items:
        return []

    workers = max_workers if max_workers is not None else os.cpu_count() or 1
    if workers <= 1 or len(items) == 1:
        return [func(item) for item in items]

    workers = min(workers, len(items))
    logger.info("Running {} tasks across {} workers", len(items), workers)

    results: list[R] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(func, item): item for item in items}
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                results.append(fut.result())
            except Exception:
                logger.exception("Worker failed for item: {}", item)
    return results
