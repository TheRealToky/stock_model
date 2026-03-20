# Quant Lab - Documentation

**Quantitative Research Lab** for stock market analysis, ML-based prediction, and strategy backtesting.

**Stack:** Python 3.11+, PostgreSQL/TimescaleDB, SQLAlchemy, yfinance, scikit-learn, XGBoost

---

## Project Structure

```
quant-lab/
├── config/                 # Centralized configuration
│   └── settings.py
├── database/               # Database schema, connection, migrations
│   ├── connection.py
│   ├── schema.py
│   └── migrate.py
├── data_pipeline/          # Data fetching, cleaning, loading
│   ├── ingestion.py
│   ├── cleaning.py
│   ├── loader.py
│   └── runner.py
├── features/               # Feature engineering
│   ├── technical.py
│   ├── registry.py
│   └── engine.py
├── models/                 # ML model implementations
│   ├── base.py
│   ├── xgboost_model.py
│   ├── random_forest_model.py
│   ├── trainer.py
│   └── registry.py
├── strategies/             # Trading strategies
│   ├── base.py
│   ├── ema_crossover.py
│   ├── momentum.py
│   ├── mean_reversion.py
│   └── registry.py
├── backtesting/            # Backtesting engines and metrics
│   ├── engine.py
│   ├── metrics.py
│   ├── report.py
│   └── vectorized.py
├── research/               # Example workflows
│   └── full_workflow.py
├── tests/                  # Unit and integration tests
│   └── unit/
│       ├── test_backtest_engine.py
│       ├── test_metrics.py
│       ├── test_strategies.py
│       └── test_technical.py
├── docker/                 # Docker configuration
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

## Data Flow

```
yfinance API
     │
     ▼
DataFetcher  ──►  TimescaleDB (OHLCV + Instruments)
                       │
                       ▼
                  DataLoader  ──►  FeatureEngine  ──►  Features Table
                                                            │
                                        ┌───────────────────┤
                                        ▼                   ▼
                                  ModelTrainer        BacktestEngine
                                  (XGBoost/RF)        (+ Strategies)
                                        │                   │
                                        ▼                   ▼
                                  ModelRegistry       BacktestReport ──► DB
```

---

## Module Reference

---

### `config/settings.py`

Centralized configuration for all components.

#### Class: `DatabaseConfig`
PostgreSQL/TimescaleDB connection settings.

| Property | Description |
|----------|-------------|
| `host` | DB host (default: `localhost`) |
| `port` | DB port (default: `5432`) |
| `database` | Database name (default: `quant_lab`) |
| `user` | DB user (default: `quant_user`) |
| `password` | DB password (default: `quant_pass`) |
| `url` | Constructed SQLAlchemy connection URL |

#### Class: `DataPipelineConfig`
Data ingestion settings.

| Property | Description |
|----------|-------------|
| `default_start_date` | Historical start date (default: `2020-01-01`) |
| `default_interval` | OHLCV interval (default: `1d`) |
| `batch_size` | Tickers per batch (default: `10`) |
| `max_retries` | Retry count for failed fetches (default: `3`) |
| `retry_delay` | Delay between retries in seconds (default: `5`) |
| `default_tickers` | Default ticker list |

#### Class: `BacktestConfig`
Backtest simulation parameters.

| Property | Description |
|----------|-------------|
| `initial_capital` | Starting capital (default: `100000`) |
| `commission` | Commission per trade (default: `0.001`) |
| `slippage` | Slippage per trade (default: `0.0005`) |
| `risk_free_rate` | Annual risk-free rate (default: `0.05`) |

#### Class: `ModelConfig`
ML model storage settings.

| Property | Description |
|----------|-------------|
| `model_dir` | Directory for saved models |
| `test_size` | Test set fraction (default: `0.2`) |

#### Class: `Settings`
Aggregates all config classes. Singleton accessor via `settings` property.

---

### `database/connection.py`

Database connection and session management.

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_engine` | `(echo=False)` | Creates SQLAlchemy engine with connection pooling |
| `get_session` | `()` | Returns a new database session |

| Object | Description |
|--------|-------------|
| `SessionLocal` | Session factory bound to the default engine |

---

### `database/schema.py`

SQLAlchemy ORM models for TimescaleDB.

#### Class: `Base`
SQLAlchemy `DeclarativeBase`.

#### Class: `Instrument`
Stock/ETF metadata.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment ID |
| `ticker` | String | Unique ticker symbol |
| `name` | String | Instrument name |
| `exchange` | String | Exchange name |
| `asset_type` | String | e.g. `stock`, `etf` |
| `sector` | String | Sector classification |
| `currency` | String | Trading currency |
| `is_active` | Boolean | Active status |

#### Class: `OHLCVData`
Time-series price data (TimescaleDB hypertable on `timestamp`).

| Column | Type | Description |
|--------|------|-------------|
| `instrument_id` | Integer (FK) | Reference to Instrument |
| `timestamp` | DateTime | Bar timestamp |
| `interval` | String | Time interval (e.g. `1d`) |
| `open`, `high`, `low`, `close` | Float | Price data |
| `volume` | BigInteger | Volume |

#### Class: `Feature`
Computed features in long format.

| Column | Type | Description |
|--------|------|-------------|
| `instrument_id` | Integer (FK) | Reference to Instrument |
| `timestamp` | DateTime | Feature timestamp |
| `feature_name` | String | Name of the feature |
| `feature_value` | Float | Computed value |

#### Class: `ModelRegistry`
Trained model metadata.

| Column | Type | Description |
|--------|------|-------------|
| `model_name` | String | Unique model name |
| `model_type` | String | Model class type |
| `hyperparameters` | JSON | Model hyperparameters |
| `training_dataset_info` | JSON | Training data details |
| `performance_metrics` | JSON | Evaluation metrics |
| `model_path` | String | Path to serialized model |

#### Class: `StrategyRegistry`
Strategy metadata.

| Column | Type | Description |
|--------|------|-------------|
| `strategy_name` | String | Unique strategy name |
| `description` | String | Strategy description |
| `parameters` | JSON | Strategy parameters |
| `dependencies` | JSON | Required features/columns |
| `performance_metrics` | JSON | Backtest results |

#### Class: `BacktestResult`
Backtest execution results.

| Column | Type | Description |
|--------|------|-------------|
| `strategy_id` | Integer (FK) | Reference to StrategyRegistry |
| `model_id` | Integer (FK) | Reference to ModelRegistry |
| `start_date`, `end_date` | DateTime | Backtest date range |
| `initial_capital`, `final_capital` | Float | Capital values |
| `metrics` | JSON | Performance metrics |
| `trades` | JSON | Trade log |
| `equity_curve` | JSON | Equity over time |

---

### `database/migrate.py`

SQL migration runner.

| Function | Signature | Description |
|----------|-----------|-------------|
| `_get_engine` | `()` | Build engine from settings |
| `_ensure_migrations_table` | `()` | Create `schema_migrations` tracking table |
| `_applied_versions` | `(conn)` | Query already-applied migration versions |
| `_discover_migrations` | `()` | Find all `.sql` migration files |
| `_version_from_path` | `(path)` | Extract version string from filename |
| `show_status` | `()` | Display applied/pending migration status |
| `run_migrations` | `()` | Apply all pending migrations in order |
| `main` | `()` | CLI entry point |

---

### `data_pipeline/ingestion.py`

#### Class: `DataFetcher`
Fetches market data from yfinance and stores it in the database.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(max_retries=None, retry_delay=None)` | Initialize with retry settings |
| `fetch_ohlcv` | `(ticker, start_date, end_date, interval)` | Fetch OHLCV for a single ticker with retry logic |
| `fetch_batch` | `(tickers, start_date, end_date, interval)` | Fetch OHLCV for multiple tickers |
| `upsert_instrument` | `(ticker)` | Insert or retrieve an instrument record |
| `store_ohlcv` | `(instrument_id, df, interval)` | Upsert OHLCV data using `ON CONFLICT` |
| `run_full_ingestion` | `(tickers, start_date, end_date, interval)` | Full historical data fetch and store |
| `run_incremental_update` | `(tickers, interval)` | Fetch only new data since last stored timestamp |
| `_normalize_ohlcv` | `(df)` | Normalize yfinance DataFrame (columns, timezone) |
| `_fetch_instrument_info` | `(ticker)` | Get instrument metadata from yfinance |
| `_prepare_ohlcv_records` | `(instrument_id, df, interval)` | Convert DataFrame rows to SQL-ready dicts |
| `_get_last_timestamp` | `(instrument_id, interval)` | Get most recent stored timestamp for a ticker |

---

### `data_pipeline/cleaning.py`

#### Class: `DataCleaner`
Cleans and validates OHLCV data.

| Method | Signature | Description |
|--------|-----------|-------------|
| `clean_ohlcv` | `(df)` | Full pipeline: deduplicate, sort, forward-fill, normalize tz, validate |
| `detect_outliers` | `(df, column, n_std)` | Flag outliers using rolling window statistics |
| `remove_gaps` | `(df, max_gap_days)` | Split on large gaps, keep the longest contiguous segment |
| `_normalize_timezone` | `(df)` | Ensure timezone-aware UTC index |
| `_validate_ohlcv_constraints` | `(df)` | Drop rows where `high < low` or `volume < 0` |

---

### `data_pipeline/loader.py`

#### Class: `DataLoader`
Loads data from the database into DataFrames.

| Method | Signature | Description |
|--------|-----------|-------------|
| `load_ohlcv` | `(ticker, start_date, end_date, interval)` | Load OHLCV for a single ticker |
| `load_multiple` | `(tickers, start_date, end_date, interval)` | Load OHLCV for multiple tickers |
| `load_features` | `(ticker, feature_names, start_date, end_date)` | Load computed features, pivot to wide format |
| `get_available_tickers` | `()` | List all tickers with stored data |
| `get_date_range` | `(ticker, interval)` | Return `(min_timestamp, max_timestamp)` for a ticker |
| `_resolve_instrument_id` | `(ticker)` | Look up instrument ID by ticker symbol |
| `_build_ohlcv_query` | `(...)` | Build parameterized OHLCV SQL query |
| `_build_features_query` | `(...)` | Build parameterized features SQL query |
| `_format_ohlcv_result` | `(df)` | Convert raw query result to standard DataFrame format |

---

### `data_pipeline/runner.py`

CLI entry point for the data pipeline.

| Function | Signature | Description |
|----------|-----------|-------------|
| `_setup_logging` | `(level)` | Configure logging output |
| `cmd_ingest` | `(args)` | CLI command: full historical ingestion |
| `cmd_update` | `(args)` | CLI command: incremental update |
| `cmd_status` | `(args)` | CLI command: display database status |
| `main` | `()` | Argument parser with subcommands: `ingest`, `update`, `status` |

---

### `features/technical.py`

Standalone functions for computing technical indicators on OHLCV DataFrames.

| Function | Signature | Description |
|----------|-----------|-------------|
| `_validate_ohlcv` | `(df)` | Validate that required OHLCV columns exist |
| `compute_returns` | `(df, periods)` | Simple returns over given periods |
| `compute_log_returns` | `(df, periods)` | Logarithmic returns over given periods |
| `compute_sma` | `(df, window)` | Simple Moving Average |
| `compute_ema` | `(df, span)` | Exponential Moving Average |
| `compute_rsi` | `(df, window)` | Relative Strength Index (Wilder smoothing) |
| `compute_macd` | `(df, fast, slow, signal)` | MACD line, signal line, histogram |
| `compute_bollinger_bands` | `(df, window, num_std)` | Upper, middle, lower Bollinger Bands |
| `compute_atr` | `(df, window)` | Average True Range |
| `compute_volatility` | `(df, window)` | Rolling standard deviation of returns |
| `compute_rolling_stats` | `(df, window)` | Rolling mean, std, skew, kurtosis |
| `compute_volume_features` | `(df, window)` | Volume SMA, volume ratio, OBV |
| `compute_price_features` | `(df)` | High-low range, close-open range, gap |

---

### `features/registry.py`

Feature registration system.

| Function/Object | Signature | Description |
|-----------------|-----------|-------------|
| `FEATURE_REGISTRY` | `dict` | Global mapping of `feature_name -> (function, default_params)` |
| `register_feature` | `(name, default_params)` | Decorator to register a feature function |
| `get_feature_func` | `(name)` | Look up a feature function by name |
| `list_features` | `()` | Return sorted list of all registered feature names |
| `_register_builtins` | `()` | Populate registry with 12 built-in features |

---

### `features/engine.py`

#### Class: `FeatureEngine`
Orchestrates feature computation and storage.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(loader=None)` | Initialize with optional `DataLoader` |
| `compute_features` | `(df, feature_names)` | Compute features on an OHLCV DataFrame, flatten composite features |
| `compute_and_store` | `(ticker, feature_names, start_date, end_date)` | Load OHLCV, compute features, store to DB |
| `compute_batch` | `(tickers, feature_names)` | Batch compute and store for multiple tickers |
| `get_feature_matrix` | `(ticker, feature_names, start_date, end_date)` | Retrieve and pivot stored features from DB |
| `_resolve_instrument_id` | `(ticker)` | Look up instrument ID |
| `_store_features` | `(instrument_id, features_df, ohlcv_columns)` | Persist features in long format with upsert |

---

### `models/base.py`

#### Class: `BaseModel` (ABC)
Abstract base class for all ML models.

| Method | Signature | Description |
|--------|-----------|-------------|
| `train` | `(X, y)` | **Abstract.** Fit the model |
| `predict` | `(X)` | **Abstract.** Generate predictions |
| `predict_proba` | `(X)` | **Abstract.** Return class probability estimates |
| `get_hyperparameters` | `()` | **Abstract.** Return hyperparameter dict |
| `save` | `(path)` | **Abstract.** Persist model to disk |
| `load` | `(path)` | **Abstract classmethod.** Load model from disk |
| `evaluate` | `(X, y)` | Compute accuracy, precision, recall, F1, confusion matrix |

---

### `models/xgboost_model.py`

#### Class: `XGBoostModel(BaseModel)`
XGBoost wrapper for binary classification.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(name, n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, **kwargs)` | Initialize XGBClassifier |
| `train` | `(X, y)` | Fit XGBClassifier |
| `predict` | `(X)` | Predict class labels |
| `predict_proba` | `(X)` | Predict class probabilities |
| `get_hyperparameters` | `()` | Return XGBoost hyperparameters |
| `save` | `(path)` | Save with joblib |
| `load` | `(path)` | Load with joblib (classmethod) |
| `feature_importances` | *(property)* | Feature importance array |

---

### `models/random_forest_model.py`

#### Class: `RandomForestModel(BaseModel)`
Random Forest wrapper for binary classification.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(name, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=42, **kwargs)` | Initialize RandomForestClassifier |
| `train` | `(X, y)` | Fit RandomForestClassifier |
| `predict` | `(X)` | Predict class labels |
| `predict_proba` | `(X)` | Predict class probabilities |
| `get_hyperparameters` | `()` | Return RF hyperparameters |
| `save` | `(path)` | Save with joblib |
| `load` | `(path)` | Load with joblib (classmethod) |
| `feature_importances` | *(property)* | Feature importance array |

---

### `models/trainer.py`

#### Class: `ModelTrainer`
Handles training, evaluation, and registration of models.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `()` | Initialize with `ModelRegistry` |
| `prepare_data` | `(ticker, feature_names, target_column, start_date, end_date, test_size)` | Load features, create binary target (next-day direction), time-aware train/test split |
| `train_model` | `(model, X_train, y_train)` | Fit a model on training data |
| `evaluate_model` | `(model, X_test, y_test)` | Return evaluation metrics dict |
| `train_and_register` | `(model, ticker, feature_names, start_date, end_date, test_size, description)` | Full pipeline: prepare, train, evaluate, register |
| `cross_validate` | `(model, X, y, n_splits)` | Time-series cross-validation using `TimeSeriesSplit` |

---

### `models/registry.py`

#### Class: `ModelRegistryRow`
SQLAlchemy ORM mapping for the `model_registry` table.

#### Class: `ModelRegistry`
Manages persistence and retrieval of trained models.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `()` | Initialize, ensure table exists |
| `register` | `(model, training_info, metrics, description)` | Save model to disk and register metadata in DB |
| `get` | `(model_name)` | Fetch model metadata by name |
| `load_model` | `(model_name)` | Load a trained model object from disk |
| `list_all` | `()` | List all registered models |
| `compare` | `(model_names)` | Compare metrics of multiple models side-by-side (DataFrame) |
| `delete` | `(model_name)` | Remove model from DB and disk |

| Helper Function | Signature | Description |
|-----------------|-----------|-------------|
| `_make_json_safe` | `(obj)` | Recursively convert numpy types to JSON-serializable primitives |

---

### `strategies/base.py`

#### Class: `BaseStrategy` (ABC)
Abstract base class for all trading strategies.

| Attribute | Description |
|-----------|-------------|
| `name` | Strategy name (class attribute) |
| `description` | Strategy description (class attribute) |

| Method | Signature | Description |
|--------|-----------|-------------|
| `generate_signals` | `(df)` | **Abstract.** Return Series of `1` (buy), `-1` (sell), `0` (hold) |
| `get_parameters` | `()` | **Abstract.** Return dict of strategy parameters |
| `get_dependencies` | `()` | **Abstract.** Return list of required DataFrame columns |
| `validate_dataframe` | `(df)` | Check that all dependency columns are present |
| `__repr__` | `()` | String representation with parameters |

---

### `strategies/ema_crossover.py`

#### Class: `EMA_Crossover(BaseStrategy)`
Dual EMA crossover strategy.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(fast_period=12, slow_period=26)` | Initialize with period validation |
| `generate_signals` | `(df)` | Emit buy on fast crossing above slow, sell on crossing below |
| `get_parameters` | `()` | Return `{fast_period, slow_period}` |
| `get_dependencies` | `()` | Return `['close']` |

---

### `strategies/momentum.py`

#### Class: `Momentum(BaseStrategy)`
Rolling return momentum strategy.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(lookback=20, top_pct=0.2)` | Initialize with lookback window and percentile threshold |
| `generate_signals` | `(df)` | Buy when rolling return is in top percentile, sell when negative |
| `get_parameters` | `()` | Return `{lookback, top_pct}` |
| `get_dependencies` | `()` | Return `['close']` |

---

### `strategies/mean_reversion.py`

#### Class: `MeanReversion(BaseStrategy)`
Bollinger Band-style mean reversion strategy.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(lookback=20, entry_std=2.0, exit_std=0.5)` | Initialize with band parameters |
| `generate_signals` | `(df)` | Buy when price falls `entry_std` below mean, sell when within `exit_std` (stateful) |
| `get_parameters` | `()` | Return `{lookback, entry_std, exit_std}` |
| `get_dependencies` | `()` | Return `['close']` |

---

### `strategies/registry.py`

#### Class: `StrategyRegistry`
Manages strategy metadata in the database.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `()` | Initialize with DB engine |
| `register` | `(strategy, performance_metrics)` | Insert or update strategy metadata |
| `get` | `(strategy_name)` | Fetch strategy metadata by name |
| `list_all` | `()` | Return all strategies as DataFrame |
| `update_metrics` | `(strategy_name, metrics)` | Update performance metrics for a strategy |
| `deregister` | `(strategy_name)` | Remove a strategy from the registry |

---

### `backtesting/engine.py`

#### Dataclass: `BacktestResult`
Container for backtest output.
| Field | Type | Description |
|-------|------|-------------|
| `equity_curve` | list | Equity values over time |
| `trades` | list[dict] | Trade log with entry/exit details |
| `metrics` | dict | Performance metrics |
| `parameters` | dict | Strategy parameters used |
| `strategy_name` | str | Name of the strategy |
| `ticker` | str | Ticker symbol |
| `start_date` | str | Backtest start date |
| `end_date` | str | Backtest end date |

#### Class: `BacktestEngine`
Event-driven backtesting engine.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(initial_capital=None, commission=None, slippage=None)` | Initialize with simulation parameters |
| `run` | `(strategy, df, ticker)` | Run a single backtest: generate signals, simulate trades, compute metrics |
| `run_multiple` | `(strategy, tickers, start_date, end_date)` | Run strategy across multiple tickers |
| `_simulate_trades` | `(signals, prices, dates, initial_capital, commission, slippage)` | Event-driven trade simulation with position tracking |

---

### `backtesting/metrics.py`

Performance metric functions.

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_sharpe_ratio` | `(returns, risk_free_rate, periods)` | Annualized Sharpe ratio |
| `compute_sortino_ratio` | `(returns, risk_free_rate, periods)` | Annualized Sortino ratio (downside deviation only) |
| `compute_max_drawdown` | `(equity_curve)` | Maximum peak-to-trough percentage decline |
| `compute_cagr` | `(initial_value, final_value, years)` | Compound Annual Growth Rate |
| `compute_win_rate` | `(trades)` | Percentage of profitable trades |
| `compute_profit_factor` | `(trades)` | Gross profit divided by gross loss |
| `compute_calmar_ratio` | `(cagr, max_drawdown)` | CAGR / max drawdown |
| `compute_all_metrics` | `(equity_curve, trades, risk_free_rate, periods)` | Compute all of the above at once |

---

### `backtesting/report.py`

#### Class: `BacktestReport`
Generates and persists backtest reports.

| Method | Signature | Description |
|--------|-----------|-------------|
| `generate` | `(result)` | Return formatted text summary of a backtest |
| `save_to_db` | `(result, strategy_id, model_id)` | Persist backtest result to the database |
| `load_from_db` | `(backtest_id)` | Load a previously stored backtest result |
| `compare` | `(results)` | Compare multiple backtest results as a DataFrame |

---

### `backtesting/vectorized.py`

#### Class: `VectorizedBacktest`
Fast numpy-based backtesting for rapid iteration.

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `(signals, prices, initial_capital, commission)` | Fast vectorized backtest using numpy |
| `run_with_vectorbt` | `(signals, prices, initial_capital, fees)` | Optional vectorbt integration for advanced analytics |

---

### `research/full_workflow.py`

End-to-end example workflow.

| Function | Signature | Description |
|----------|-----------|-------------|
| `main` | `()` | Demonstrates the full pipeline: ingestion, loading, feature engineering, model training, registration, strategy backtesting, and report storage |

---

### `tests/`

Unit tests using pytest.

| File | Coverage |
|------|----------|
| `tests/unit/test_backtest_engine.py` | `BacktestEngine`, `BacktestResult`, trade simulation |
| `tests/unit/test_metrics.py` | All metric functions (Sharpe, Sortino, drawdown, etc.) |
| `tests/unit/test_strategies.py` | `EMA_Crossover`, `Momentum`, `MeanReversion` signal generation |
| `tests/unit/test_technical.py` | All technical indicator functions |

---

## Design Patterns

| Pattern | Usage |
|---------|-------|
| **Registry** | Models, strategies, and features all use central registries for lookup and management |
| **Strategy** | All trading strategies inherit from `BaseStrategy` with a consistent `generate_signals` interface |
| **Abstract Factory** | `ModelRegistry.load_model()` dynamically loads the correct model class |
| **Pipeline** | Data flows through fetch, clean, load, and feature stages sequentially |
| **Decorator** | `@register_feature` for convenient feature registration |
| **Singleton** | `Settings` uses a singleton accessor |

---

## Dependencies

**Core:** pandas, numpy, sqlalchemy, psycopg2, yfinance, scikit-learn, xgboost, joblib, matplotlib, seaborn

**Additional:** ta, pandas-ta, lightgbm, plotly, python-dotenv, loguru, tqdm

**Optional (dev):** pytest, ruff, jupyterlab, vectorbt, backtrader
