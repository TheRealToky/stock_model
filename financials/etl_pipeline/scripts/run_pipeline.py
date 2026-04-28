"""Programmatic example: run the full ETL pipeline.

Equivalent to::

    python -m financials.etl_pipeline.main run --mode full

but lets you tweak the config in Python before running -- handy in
notebooks or in CI tasks that build the dataset before training.
"""

from __future__ import annotations

from financials.etl_pipeline import ETLPipeline, load_config


def main() -> None:
    # 1. Load defaults from pipeline.yaml and override programmatically.
    cfg = load_config(
        overrides={
            "extract": {
                "tickers": ["AAPL", "MSFT", "NVDA"],
                "start_date": "2023-01-01",
                "end_date": None,                  # up to "now"
                "interval": "1min",
                "chunk_days": 30,
            },
            "features": {
                "enabled_features": [
                    "log_returns", "sma", "ema", "rsi",
                    "macd", "bollinger_bands", "atr",
                    "volume_features", "price_features",
                ],
                "feature_params": {
                    "sma": {"window": 50},
                    "ema": {"span": 50},
                },
            },
            "load": {
                "output_path": "data/feature_store",
                "compression": "zstd",
                "partition_by_symbol": True,
            },
            "runtime": {
                "num_workers": 4,
                "log_level": "INFO",
            },
        }
    )

    pipeline = ETLPipeline(cfg)

    # 2. First-time build for these tickers.
    summary_full = pipeline.run_full()
    print("FULL run summary:", summary_full.as_dict())

    # 3. Later (e.g. nightly) -- only fetch new bars since the watermark.
    summary_incr = pipeline.run_incremental()
    print("INCREMENTAL run summary:", summary_incr.as_dict())


if __name__ == "__main__":
    main()
