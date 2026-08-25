"""Production ETL pipeline for financial time-series ML training.

Extracts OHLCV bars from TimescaleDB, applies feature engineering using the
existing :mod:`financials.features` module, and writes Hive-partitioned
Parquet datasets that downstream ML training can consume via
:class:`~financials.etl_pipeline.load.reader.MLDataLoader`.

Exports are resolved lazily (PEP 562).  Eagerly importing them here meant
that reading a Parquet file pulled in the write path too -- ``ETLPipeline``
drags in SQLAlchemy, psycopg2 and yfinance by way of
:mod:`financials.features`, and :mod:`financials.database.connection`
builds an engine at import time.  A training or inference process that only
wants :class:`MLDataLoader` should not need a Postgres driver, and unit
tests should not need one to import the reader.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - static analysers only
    from financials.etl_pipeline.config.etl_config import (  # noqa: F401
        ETLConfig,
        load_config,
    )
    from financials.etl_pipeline.load.reader import (  # noqa: F401
        MLDataLoader,
        StreamingSequenceDataset,
    )
    from financials.etl_pipeline.pipeline import ETLPipeline  # noqa: F401

# Attribute name -> module that defines it.
_LAZY_EXPORTS: dict[str, str] = {
    "ETLPipeline": "financials.etl_pipeline.pipeline",
    "ETLConfig": "financials.etl_pipeline.config.etl_config",
    "load_config": "financials.etl_pipeline.config.etl_config",
    "MLDataLoader": "financials.etl_pipeline.load.reader",
    "StreamingSequenceDataset": "financials.etl_pipeline.load.reader",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    """Import an export on first access (PEP 562)."""
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_path), name)
    globals()[name] = value  # cache so __getattr__ runs once per name
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY_EXPORTS])
