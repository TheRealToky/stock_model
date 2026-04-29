# Quant Lab

A modular quantitative research lab for financial strategy development, backtesting, and ML model cataloguing. Built for fast experimentation with trading strategies using historical market data.

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                               QUANT LAB                                      │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐            │
│  │   yfinance   │───▶│   Data       │───▶│   TimescaleDB        │            │
│  │   API        │    │   Pipeline   │    │   (PostgreSQL 16)    │            │
│  └──────────────┘    │              │    │                      │            │
│                      │  • Ingestion │    │  • instruments       │            │
│                      │  • Cleaning  │    │  • ohlcv_data ⊡      │            │
│                      │  • Loader    │    │  • features ⊡        │            │
│                      └──────────────┘    │  • model_registry    │            │
│                                          │  • strategy_registry │            │
│  ┌──────────────┐    ┌──────────────┐    │  • backtest_results  │            │
│  │  Strategy    │    │  Backtest    │◀───│                      │            │
│  │  Framework   │───▶│  Engine      │    │  ⊡ = hypertable      │            │
│  │              │    │              │    └──────────┬───────────┘            │
│  │  • EMA Cross │    │  • Event-    │               │                        │
│  │  • Mean Rev  │    │    driven    │               │                        │
│  │  • Momentum  │    │  • Vectorized│               │                        │
│  └──────────────┘    │  • Metrics   │               ▼                        │
│                      └──────────────┘    ┌──────────────────────┐            │
│                                          │   ETL Pipeline       │            │
│  ┌──────────────┐    ┌──────────────┐    │   (Feature Store)    │            │
│  │  Jupyter     │    │  Reports &   │    │                      │            │
│  │  Notebooks   │───▶│  Comparison  │    │  • Chunked extract   │            │
│  │  (Research)  │    │              │    │  • Feature compute   │            │
│  └──────────────┘    └──────────────┘    │  • Parquet output    │            │
│                                          │  • Manifest / incr.  │            │
│                                          └──────────┬───────────┘            │
│                                                     │                        │
│                                                     ▼                        │
│                                          ┌──────────────────────┐            │
│                                          │   ML Pipeline        │            │
│                                          │                      │            │
│                                          │  • XGBoost           │            │
│                                          │  • Random Forest     │            │
│                                          │  • Trainer           │            │
│                                          │  • Registry          │            │
│                                          └──────────────────────┘            │
└──────────────────────────────────────────────────────────────────────────────┘
             Docker Compose (timescaledb + python-app + jupyter)
```

## Database Schema

```
instruments                    ohlcv_data (hypertable)
┌──────────────────┐          ┌─────────────────────────┐
│ id (UUID, PK)    │◀────────│ instrument_id (FK)       │
│ ticker (unique)  │          │ timestamp (PK, partkey)  │
│ name             │          │ interval (PK)            │
│ exchange         │          │ open / high / low / close│
│ asset_type       │          │ volume / adjusted_close  │
│ sector / currency│          └─────────────────────────┘
│ is_active        │
│ created_at       │          features (hypertable)
└──────────────────┘          ┌─────────────────────────┐
                              │ instrument_id (FK)       │
strategy_registry             │ timestamp (PK, partkey)  │
┌──────────────────┐          │ feature_name (PK)        │
│ id (UUID, PK)    │          │ feature_value            │
│ strategy_name    │          └─────────────────────────┘
│ description      │
│ parameters (JSON)│          model_registry
│ dependencies     │          ┌─────────────────────────┐
│ perf_metrics     │          │ id (UUID, PK)            │
│ last_evaluated_at│          │ model_name (unique)      │
│ created/updated  │          │ model_type               │
└────────┬─────────┘          │ hyperparameters (JSON)   │
         │                    │ training_info (JSON)     │
         │                    │ perf_metrics (JSON)      │
         ▼                    │ model_path               │
backtest_results              │ created/updated          │
┌──────────────────┐          └────────────┬────────────┘
│ id (UUID, PK)    │                       │
│ strategy_id (FK) │───────────────────────┘
│ model_id (FK)    │
│ instrument_ids   │
│ start/end_date   │
│ initial/final $  │
│ sharpe / sortino │
│ max_dd / cagr    │
│ win_rate / trades│
│ equity_curve     │
└──────────────────┘
```

## Project Structure

```
quant-lab/
│
├── financials/                # Financial / OHLCV side of the lab
│   ├── __init__.py
│   ├── config/                # Central configuration
│   │   ├── __init__.py
│   │   └── settings.py        # Database, pipeline, backtest, model settings
│   │
│   ├── database/              # Database layer
│   │   ├── __init__.py
│   │   ├── connection.py      # SQLAlchemy engine & session factory
│   │   ├── schema.py          # ORM models for all tables
│   │   ├── migrate.py         # Migration runner script
│   │   └── migrations/
│   │       └── 001_initial_schema.sql  # Idempotent schema with hypertables
│   │
│   ├── data_pipeline/         # Data ingestion & transformation
│   │   ├── __init__.py
│   │   ├── ingestion.py       # yfinance fetcher + DB upserts
│   │   ├── cleaning.py        # Dedup, fill, timezone normalization
│   │   ├── loader.py          # Load from DB → DataFrames
│   │   └── runner.py          # CLI: ingest / update / status
│   │
│   ├── features/              # Feature engineering
│   │   ├── __init__.py
│   │   ├── technical.py       # Pure indicator functions (RSI, MACD, etc.)
│   │   ├── engine.py          # Compute + store features
│   │   └── registry.py        # Feature function registry
│   │
│   ├── etl_pipeline/          # ML-ready feature store pipeline
│   │   ├── __init__.py        # Public API: ETLPipeline, load_config, MLDataLoader
│   │   ├── main.py            # CLI entry point (run / status)
│   │   ├── pipeline.py        # ETLPipeline orchestrator + RunSummary
│   │   ├── config/
│   │   │   ├── etl_config.py  # Typed dataclass configuration
│   │   │   └── pipeline.yaml  # Default config (extract / features / load / runtime)
│   │   ├── extract/
│   │   │   ├── base.py        # Abstract Extractor interface
│   │   │   └── timescale.py   # Server-side cursor streaming from TimescaleDB
│   │   ├── transform/
│   │   │   ├── base.py        # Abstract Transformer interface
│   │   │   └── feature_transformer.py  # FeatureEngine wrapper with warmup handling
│   │   ├── load/
│   │   │   ├── base.py        # Abstract Loader interface
│   │   │   ├── parquet_writer.py  # Hive-partitioned Parquet output
│   │   │   └── reader.py      # MLDataLoader for reading the feature store
│   │   ├── features/
│   │   │   └── selector.py    # Feature resolution / subsetting logic
│   │   ├── utils/
│   │   │   ├── logging.py     # Structured logging helpers
│   │   │   ├── manifest.py    # High-water mark persistence (_manifest.json)
│   │   │   ├── parallel.py    # Multi-worker process pool orchestration
│   │   │   └── validation.py  # OHLCV + feature shape validation
│   │   └── scripts/
│   │       ├── run_pipeline.py       # Programmatic usage example
│   │       └── load_for_training.py  # Dataset loading example for ML training
│   │
│   └── backtesting/           # Backtesting engine
│       ├── __init__.py
│       ├── engine.py          # Event-driven backtest simulator
│       ├── metrics.py         # Sharpe, Sortino, MaxDD, CAGR, etc.
│       ├── report.py          # Report generation + DB persistence
│       └── vectorized.py      # Fast numpy / vectorbt backtests
│
├── alt_data/                  # Alternative-data pipeline (OpenSky flights)
│   ├── __init__.py
│   ├── README.md
│   ├── requirements.txt
│   ├── run_pipeline.py        # CLI runner
│   ├── config/                # Settings + YAML (features.yaml, mappings.yaml)
│   ├── ingestion/             # OpenSky client, fetcher, schemas
│   ├── cleaning/              # Cleaning logic
│   ├── features/              # Feature registry, engine, technicals
│   ├── database/              # Storage layer + migrations/
│   │   └── migrations/
│   │       └── 001_initial_alt_schema.sql
│   ├── query/                 # Cross-dataset query layer (DataHub, mappings)
│   └── utils/                 # Logging, time utilities
│
├── strategies/                # Trading strategies
│   ├── __init__.py
│   ├── base.py                # Abstract BaseStrategy
│   ├── ema_crossover.py       # EMA crossover strategy
│   ├── mean_reversion.py      # Bollinger band mean reversion
│   ├── momentum.py            # Rolling-return momentum
│   └── registry.py            # Strategy DB registry
│
├── models/                    # ML models
│   ├── __init__.py
│   ├── base.py                # Abstract BaseModel
│   ├── xgboost_model.py       # XGBoost classifier
│   ├── random_forest_model.py # Random Forest classifier
│   ├── trainer.py             # Train / evaluate / cross-validate
│   ├── registry.py            # Model DB registry
│   └── saved/                 # Serialized model artifacts
│
├── notebooks/                 # Jupyter notebooks (research + alt-data)
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_strategy_backtest.ipynb
│   ├── 04_ml_pipeline.ipynb
│   ├── 05_cross_dataset_example.ipynb
│   ├── full_workflow.py       # End-to-end script
│   └── example_cross_dataset.py
│
├── docker/                    # Docker build files
│   ├── Dockerfile.financials  # Python research container (financial side)
│   ├── Dockerfile.alt_data    # Alt-data pipeline container
│   └── Dockerfile.jupyter     # JupyterLab container
│
├── tests/                     # Test suite
│   ├── unit/
│   │   ├── financials/
│   │   │   ├── test_metrics.py
│   │   │   ├── test_strategies.py
│   │   │   ├── test_technical.py
│   │   │   └── test_backtest_engine.py
│   │   └── alt_data/
│   │       ├── conftest.py
│   │       ├── test_cleaner.py
│   │       ├── test_features.py
│   │       ├── test_fetcher.py
│   │       ├── test_flight_data.py
│   │       ├── test_query.py
│   │       └── test_time_utils.py
│   └── integration/
│
├── docker-compose.yml         # Full environment orchestration
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project metadata & tool config
├── .env.example               # Environment variable template
└── .gitignore
```

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url> quant-lab && cd quant-lab
cp .env.example .env
# Edit .env if you want to change defaults
```

### 2. Start the environment

```bash
docker compose up -d
```

This launches three containers:
- **TimescaleDB** on port 5432 (with health check)
- **Python app** container for pipelines and scripts
- **JupyterLab** on port 8888 (token: `quantlab`)

### 3. Run database migrations

```bash
docker compose exec python-app python -m financials.database.migrate
```

### 4. Ingest market data

```bash
# Full ingestion (default tickers from 2010)
docker compose exec python-app python -m financials.data_pipeline.runner ingest

# Specific tickers
docker compose exec python-app python -m financials.data_pipeline.runner ingest \
    --tickers AAPL MSFT GOOGL --start 2020-01-01

# Incremental update (fetch only new data)
docker compose exec python-app python -m financials.data_pipeline.runner update

# Check what's in the database
docker compose exec python-app python -m financials.data_pipeline.runner status
```

### 5. Build the feature store (optional, for ML training)

The ETL pipeline extracts OHLCV bars from TimescaleDB, computes technical features, and writes a Hive-partitioned Parquet dataset optimized for ML training.

```bash
# Full build (all tickers, full date window)
docker compose exec python-app python -m financials.etl_pipeline.main run --mode full

# Incremental update (only bars since the last run)
docker compose exec python-app python -m financials.etl_pipeline.main run --mode incremental

# Check dataset state
docker compose exec python-app python -m financials.etl_pipeline.main status
```

See [`financials/etl_pipeline/README.md`](financials/etl_pipeline/README.md) for the full configuration reference.

### 6. Open JupyterLab

Navigate to `http://localhost:8888` (token: `quantlab`). Open the research notebooks in order:

1. **01_data_exploration.ipynb** -- Load and visualize OHLCV data
2. **02_feature_engineering.ipynb** -- Compute technical indicators
3. **03_strategy_backtest.ipynb** -- Run and compare strategies
4. **04_ml_pipeline.ipynb** -- Train and evaluate ML models
5. **05_cross_dataset_example.ipynb** -- Join OHLCV data with flight data

## Example Workflows

### Strategy backtesting

```python
from financials.data_pipeline.ingestion import DataFetcher
from financials.data_pipeline.loader import DataLoader
from financials.features.engine import FeatureEngine
from strategies import EMACrossover
from financials.backtesting.engine import BacktestEngine
from financials.backtesting.report import BacktestReport

# 1. Fetch and load data
fetcher = DataFetcher()
fetcher.run_full_ingestion(["AAPL"], start_date="2020-01-01")
loader = DataLoader()
df = loader.load_ohlcv("AAPL", start_date="2020-01-01")

# 2. Generate features
engine = FeatureEngine()
features = engine.compute_features(df)

# 3. Run a strategy backtest
strategy = EMACrossover(fast_period=12, slow_period=26)
bt = BacktestEngine(initial_capital=100_000)
result = bt.run(strategy, df, ticker="AAPL")

report = BacktestReport()
print(report.generate(result))  # Sharpe, Sortino, MaxDD, CAGR, win rate, etc.
report.save_to_db(result)
```

### ML training via the feature store

The ETL pipeline produces a Parquet feature store that is designed for fast, memory-efficient ML training. Use it when training over large date ranges or many tickers.

```python
from financials.etl_pipeline import ETLPipeline, load_config, MLDataLoader
from models import XGBoostModel, ModelTrainer

# 1. Build (or refresh) the feature store
cfg = load_config(overrides={
    "extract": {"tickers": ["AAPL", "MSFT", "NVDA"], "start_date": "2022-01-01"},
    "features": {"enabled_features": ["log_returns", "sma", "ema", "rsi", "macd"]},
    "runtime": {"num_workers": 4},
})
pipeline = ETLPipeline(cfg)
pipeline.run_full()       # first-time build
# pipeline.run_incremental()  # subsequent nightly updates

# 2. Load the feature store for training
loader = MLDataLoader("data/feature_store")
df = loader.load(tickers=["AAPL"], start="2022-01-01", end="2024-01-01")

# 3. Train an ML model
trainer = ModelTrainer()
X_tr, X_te, y_tr, y_te = trainer.prepare_data(
    ticker="AAPL",
    feature_names=["log_returns", "rsi", "macd"],
    start_date="2022-01-01",
)
model = XGBoostModel(name="xgb_aapl_v1", n_estimators=200)
trainer.train_model(model, X_tr, y_tr)
metrics = trainer.evaluate_model(model, X_te, y_te)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **TimescaleDB** | Native hypertable partitioning for time-series queries; full PostgreSQL compatibility |
| **UUID primary keys** | Safe for distributed inserts; no sequence contention |
| **JSONB columns** | Flexible schema for hyperparameters, metrics, and config without migrations |
| **Composite PK on OHLCV** | `(instrument_id, timestamp, interval)` naturally prevents duplicates |
| **Abstract base classes** | Uniform interface for strategies and models enables plug-and-play extensibility |
| **Feature registry** | Decorator-based registration makes adding new indicators trivial |
| **Time-series CV** | `TimeSeriesSplit` prevents look-ahead bias in model evaluation |
| **Vectorized + event-driven** | Fast screening via numpy, accurate simulation via the event engine |
| **ETL feature store** | Parquet + manifest pattern decouples data prep from training; incremental runs keep it fresh without full rebuilds |
| **Chunked extraction** | 60-day windows with server-side cursors keep memory flat regardless of history length |
| **Warmup rows** | Extra bars prepended before each chunk let rolling indicators stabilize; these rows are stripped before writing |

## Extending the System

### Add a new strategy

```python
# strategies/my_strategy.py
from strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    name = "my_strategy"
    description = "Description of what it does"

    def __init__(self, param1=10):
        self.param1 = param1

    def generate_signals(self, df):
        signals = pd.Series(0, index=df.index, dtype=int)
        # ... your logic ...
        return signals

    def get_parameters(self):
        return {"param1": self.param1}

    def get_dependencies(self):
        return ["close"]
```

### Add a new ML model

```python
# models/my_model.py
from models.base import BaseModel

class MyModel(BaseModel):
    name = "my_model"
    model_type = "custom"

    def train(self, X, y): ...
    def predict(self, X): ...
    def predict_proba(self, X): ...
    def get_hyperparameters(self): ...
    def save(self, path): ...
    def load(cls, path): ...
```

### Add a new feature

```python
# In financials/features/technical.py
def compute_my_indicator(df, window=14):
    return df["close"].rolling(window).apply(my_func)

# In financials/features/registry.py
@register_feature("my_indicator")
def _my_indicator(df, window=14):
    return compute_my_indicator(df, window)
```

## Running Tests

```bash
# Inside the container
docker compose exec python-app pytest tests/ -v

# With coverage
docker compose exec python-app pytest tests/ --cov=. --cov-report=term-missing

# Locally (requires DB running)
pytest tests/unit/ -v
```

## Performance Metrics

The backtesting system computes and stores:

| Metric | Description |
|---|---|
| **Sharpe Ratio** | Risk-adjusted return (annualized) |
| **Sortino Ratio** | Downside-risk-adjusted return |
| **Max Drawdown** | Largest peak-to-trough decline |
| **CAGR** | Compound annual growth rate |
| **Calmar Ratio** | CAGR / max drawdown |
| **Win Rate** | Percentage of profitable trades |
| **Profit Factor** | Gross profit / gross loss |
| **Total Return** | Overall percentage return |

## Tech Stack

- **Python 3.12** -- Core language
- **TimescaleDB** (PostgreSQL 16) -- Time-series database with hypertable partitioning
- **SQLAlchemy 2.0** -- ORM and database toolkit
- **yfinance** -- Market data ingestion
- **pandas / numpy** -- Data manipulation
- **scikit-learn / XGBoost** -- Machine learning
- **vectorbt / Backtrader** -- Backtesting frameworks
- **Docker Compose** -- Environment orchestration
- **JupyterLab** -- Interactive research
- **pytest** -- Testing framework
