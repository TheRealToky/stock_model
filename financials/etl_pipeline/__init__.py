"""Production ETL pipeline for financial time-series ML training.

Extracts OHLCV bars from TimescaleDB, applies feature engineering using the
existing :mod:`financials.features` module, and writes Hive-partitioned
Parquet datasets that downstream ML training can consume via
:class:`~financials.etl_pipeline.load.reader.MLDataLoader`.
"""

from financials.etl_pipeline.pipeline import ETLPipeline
from financials.etl_pipeline.config.etl_config import ETLConfig, load_config
from financials.etl_pipeline.load.reader import MLDataLoader

__all__ = ["ETLPipeline", "ETLConfig", "load_config", "MLDataLoader"]
