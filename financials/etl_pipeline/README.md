# ETL Pipeline — Financial Feature Store

Extracts OHLCV bars from TimescaleDB, applies the full feature-engineering stack,
and writes a Hive-partitioned Parquet dataset optimized for ML training.

It is designed for production use: chunked extraction keeps memory flat, a
JSON manifest tracks per-ticker high-water marks so incremental runs only
process new bars, and a process-pool worker model saturates available CPU cores.

## Directory structure

```
financials/etl_pipeline/
├── __init__.py                    # Public API: ETLPipeline, load_config, MLDataLoader
├── main.py                        # CLI entry point (run / status)
├── pipeline.py                    # ETLPipeline orchestrator + RunSummary
│
├── config/
│   ├── etl_config.py              # Typed dataclasses for all config sections
│   └── pipeline.yaml              # Defaults (edit or override via CLI / Python)
│
├── extract/
│   ├── base.py                    # Abstract Extractor interface
│   └── timescale.py               # Server-side cursor streaming from TimescaleDB
│
├── transform/
│   ├── base.py                    # Abstract Transformer interface
│   └── feature_transformer.py     # FeatureEngine wrapper + warmup strip
│
├── load/
│   ├── base.py                    # Abstract Loader interface
│   ├── parquet_writer.py          # Hive-partitioned Parquet writer (zstd)
│   └── reader.py                  # MLDataLoader — read the feature store
│
├── features/
│   └── selector.py                # Feature subsetting / registry resolution
│
├── utils/
│   ├── logging.py                 # Structured JSON-capable logging
│   ├── manifest.py                # High-water mark persistence (_manifest.json)
│   ├── parallel.py                # Process-pool map over ticker list
│   └── validation.py              # OHLCV + feature shape validation
│
└── scripts/
    ├── run_pipeline.py            # Programmatic usage example
    └── load_for_training.py       # Dataset loading example for ML training
```

## Data flow

```
TimescaleDB
    │
    │  server-side cursor, 60-day chunks
    ▼
TimescaleExtractor.stream_chunks(ticker, start, end)
    │
    │  prepend warmup_rows (default 200) so rolling indicators
    │  have data before the requested start date
    ▼
validate_ohlcv_chunk()              ← skip validation with --skip-validation
    │
    ▼
FeatureTransformer.transform()
    │  • computes all enabled features via FeatureEngine
    │  • strips warmup rows (timestamps before warmup_cutoff)
    │  • optionally casts floats to float32 (halves disk use)
    ▼
ParquetLoader.write()
    │
    │  Hive layout: data/feature_store/date=YYYY-MM-DD/symbol=AAPL/part-*.parquet
    ▼
Manifest.update(ticker, last_timestamp, rows_written)
    │  persists to data/feature_store/_manifest.json
    ▼
RunSummary (JSON to stdout)
```

## Configuration

All defaults live in [`config/pipeline.yaml`](config/pipeline.yaml). Every key
can be overridden via:
- `--config path/to/other.yaml` (CLI / programmatic)
- CLI flags (`--tickers`, `--start`, `--workers`, …)
- the `overrides` dict passed to `load_config()`

### `extract` section

| Key | Default | Description |
|---|---|---|
| `interval` | `"1min"` | OHLCV bar resolution to extract |
| `start_date` | `"2021-01-01"` | Earliest bar to include |
| `end_date` | `null` | Latest bar to include (`null` = up to latest in DB) |
| `tickers` | `[]` | Tickers to process (`[]` = all instruments in the DB) |
| `chunk_days` | `60` | Date window per chunk; controls peak memory usage |
| `warmup_rows` | `200` | Bars prepended before `start_date` for rolling indicator warmup |
| `server_fetch_size` | `50000` | Cursor batch size for TimescaleDB streaming |

`warmup_rows` must be ≥ the longest rolling window across all enabled features.
With the default feature params (SMA-20, RSI-14, MACD-26, ATR-14) 200 rows
is a safe margin.

### `features` section

| Key | Default | Description |
|---|---|---|
| `enabled_features` | `null` | Feature list to compute (`null` = all registered features) |
| `feature_params` | see YAML | Per-feature parameter overrides (window sizes, spans, …) |
| `drop_warmup_nans` | `true` | Drop rows with NaN values introduced by the warmup strip |

Default `feature_params`:

```yaml
feature_params:
  sma:            {window: 20}
  ema:            {span: 20}
  rsi:            {window: 14}
  macd:           {fast: 12, slow: 26, signal: 9}
  bollinger_bands: {window: 20, num_std: 2}
  atr:            {window: 14}
  volatility:     {window: 20}
  rolling_stats:  {window: 20}
  volume_features: {window: 20}
```

### `load` section

| Key | Default | Description |
|---|---|---|
| `output_path` | `"data/feature_store"` | Root directory for the Parquet dataset |
| `compression` | `"zstd"` | Parquet compression codec |
| `compression_level` | `3` | Compression level (higher = smaller files, slower writes) |
| `row_group_size` | `200000` | Rows per Parquet row group (affects read-time filtering) |
| `partition_by_symbol` | `true` | Add `symbol=<TICKER>` partition level under each date |
| `cast_float32` | `true` | Cast feature columns to float32 (approximately halves disk use) |

The resulting directory tree looks like:

```
data/feature_store/
├── _manifest.json
├── date=2024-01-02/
│   ├── symbol=AAPL/
│   │   └── part-0.parquet
│   └── symbol=MSFT/
│       └── part-0.parquet
└── date=2024-01-03/
    └── ...
```

### `runtime` section

| Key | Default | Description |
|---|---|---|
| `log_level` | `"INFO"` | Python logging level |
| `log_json` | `false` | Emit structured JSON log lines (for log aggregators) |
| `num_workers` | `null` | Process-pool size (`null` = `os.cpu_count()`, `1` = serial) |
| `manifest_filename` | `"_manifest.json"` | Manifest file name inside `output_path` |
| `skip_validation` | `false` | Skip OHLCV and feature shape validation |

## CLI usage

```bash
# Full rebuild — all tickers in the DB, default date window
python -m financials.etl_pipeline.main run --mode full

# Incremental — only new bars since the last manifest watermark
python -m financials.etl_pipeline.main run --mode incremental

# Incremental with custom parallelism and log level
python -m financials.etl_pipeline.main run \
    --mode incremental \
    --workers 4 \
    --log-level DEBUG

# Subset of tickers, custom date range, custom output path
python -m financials.etl_pipeline.main run \
    --mode full \
    --tickers AAPL MSFT NVDA \
    --start 2023-01-01 \
    --end 2024-01-01 \
    --output /data/custom_store

# Override config file
python -m financials.etl_pipeline.main \
    --config path/to/custom.yaml \
    run --mode full

# Inspect manifest / dataset state (JSON to stdout)
python -m financials.etl_pipeline.main status
```

The `run` command exits with code `0` on success and `1` if any tickers failed.
It prints a JSON `RunSummary` to stdout:

```json
{
  "tickers_attempted": 10,
  "tickers_succeeded": 9,
  "rows_written": 4823100,
  "skipped": [],
  "failed": ["XYZ"]
}
```

## Programmatic usage

```python
from financials.etl_pipeline import ETLPipeline, load_config, MLDataLoader

# --- Build the feature store ---

cfg = load_config(
    overrides={
        "extract": {
            "tickers": ["AAPL", "MSFT", "NVDA"],
            "start_date": "2023-01-01",
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
        "runtime": {"num_workers": 4},
    }
)

pipeline = ETLPipeline(cfg)

# First-time full build
summary = pipeline.run_full()
print(summary.as_dict())

# Nightly incremental update
summary = pipeline.run_incremental()
print(summary.as_dict())

# --- Read back for training ---

loader = MLDataLoader("data/feature_store")

# Load all tickers across a date range
df = loader.load(start="2023-01-01", end="2024-01-01")

# Load specific tickers
df = loader.load(tickers=["AAPL", "MSFT"], start="2023-06-01")
```

See [`scripts/run_pipeline.py`](scripts/run_pipeline.py) for a runnable example.

## Manifest and incremental updates

The manifest (`_manifest.json` inside `output_path`) stores the latest
processed timestamp per ticker. `run_incremental()` reads these watermarks and
starts each ticker's extraction from `last_timestamp + 1µs`, so previously
written data is never re-processed or overwritten.

```json
{
  "AAPL": {"last_timestamp": "2024-06-30T23:59:00+00:00", "rows_written": 480300},
  "MSFT": {"last_timestamp": "2024-06-30T23:59:00+00:00", "rows_written": 479800}
}
```

Tickers that appear in the DB but not in the manifest are treated as new and
processed from `extract.start_date`.

## Parallelism

By default the pipeline spawns one worker process per CPU core. Each worker
owns its own DB connection and Arrow buffers — these are not fork-safe and are
therefore created fresh inside each process rather than inherited from the
parent. The manifest is only written by the parent process after all workers
return.

Set `--workers 1` (or `num_workers: 1` in config) to run serially, which is
useful for debugging or when the DB is the bottleneck.

## Running inside Docker

```bash
# Full build
docker compose exec python-app \
    python -m financials.etl_pipeline.main run --mode full

# Nightly incremental
docker compose exec python-app \
    python -m financials.etl_pipeline.main run --mode incremental --workers 4

# Status
docker compose exec python-app \
    python -m financials.etl_pipeline.main status
```
