"""
Full Quant Lab Workflow Example
================================

This script demonstrates the complete workflow:
1. Fetch data from yfinance
2. Insert into TimescaleDB
3. Generate features
4. Train ML model
5. Run backtest
6. Store results

Usage:
    python notebooks/full_workflow.py
"""

import sys
import logging

sys.path.insert(0, ".")

from financials.data_pipeline.ingestion import DataFetcher
from financials.data_pipeline.loader import DataLoader
from financials.features.engine import FeatureEngine
from models.xgboost_model import XGBoostModel
from models.trainer import ModelTrainer
from models.registry import ModelRegistry
from strategies.ema_crossover import EMA_Crossover as EMACrossover
from strategies.momentum import Momentum
from financials.backtesting.engine import BacktestEngine
from financials.backtesting.report import BacktestReport
from strategies.registry import StrategyRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # ─── Step 1: Fetch and store market data ───────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Data Ingestion")
    logger.info("=" * 60)

    fetcher = DataFetcher()
    tickers = ["AAPL", "MSFT", "SPY"]
    fetcher.run_full_ingestion(tickers, start_date="2020-01-01")
    logger.info("Data ingestion complete.")

    # ─── Step 2: Load data and verify ──────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Verify Data")
    logger.info("=" * 60)

    loader = DataLoader()
    available = loader.get_available_tickers()
    logger.info(f"Available tickers: {available}")

    df_aapl = loader.load_ohlcv("AAPL", start_date="2020-01-01")
    logger.info(f"AAPL data: {len(df_aapl)} rows, {df_aapl.index.min()} to {df_aapl.index.max()}")

    # ─── Step 3: Feature engineering ───────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Feature Engineering")
    logger.info("=" * 60)

    engine = FeatureEngine()
    engine.compute_and_store("AAPL", start_date="2020-01-01")
    logger.info("Features computed and stored for AAPL.")

    # ─── Step 4: Train ML model ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Train ML Model")
    logger.info("=" * 60)

    trainer = ModelTrainer()
    feature_names = [
        "returns", "log_returns", "sma_20", "ema_12", "ema_26",
        "rsi_14", "volatility_20", "macd", "macd_signal",
    ]

    X_train, X_test, y_train, y_test = trainer.prepare_data(
        ticker="AAPL",
        feature_names=feature_names,
        target_column="direction",
        start_date="2020-01-01",
        test_size=0.2,
    )
    logger.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

    xgb = XGBoostModel(
        name="xgb_aapl_workflow_v1",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
    )
    trainer.train_model(xgb, X_train, y_train)
    metrics = trainer.evaluate_model(xgb, X_test, y_test)
    logger.info(f"XGBoost accuracy: {metrics.get('accuracy', 'N/A'):.4f}")

    # Register the model
    model_registry = ModelRegistry()
    model_registry.register(
        xgb,
        training_info={"ticker": "AAPL", "features": feature_names, "start_date": "2020-01-01"},
        metrics=metrics,
    )
    logger.info("Model registered.")

    # ─── Step 5: Run backtests ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Backtest Strategies")
    logger.info("=" * 60)

    bt_engine = BacktestEngine()
    bt_report = BacktestReport()

    strategies = [
        EMACrossover(fast_period=12, slow_period=26),
        Momentum(lookback=20),
    ]

    results = []
    for strategy in strategies:
        result = bt_engine.run(strategy, df_aapl, ticker="AAPL")
        results.append(result)
        logger.info(f"\n{bt_report.generate(result)}")

    # ─── Step 6: Compare and store results ─────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Compare & Store Results")
    logger.info("=" * 60)

    comparison = bt_report.compare(results)
    logger.info(f"\nStrategy Comparison:\n{comparison.to_string()}")

    # Register strategies and save results
    strat_registry = StrategyRegistry()
    for strategy, result in zip(strategies, results):
        strat_registry.register(strategy, performance_metrics=result.metrics)
        bt_report.save_to_db(result)

    logger.info("All results saved to database.")
    logger.info("=" * 60)
    logger.info("WORKFLOW COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
