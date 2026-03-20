"""Model training pipeline.

Provides :class:`ModelTrainer` which orchestrates data preparation,
training, evaluation, cross-validation, and registry integration.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config.settings import settings
from models.base import BaseModel
from models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelTrainer:
    """End-to-end training pipeline for classification models.

    The trainer loads features from the database (via the feature engine
    and data loader), creates a binary target (next-day direction),
    splits data using a time-aware methodology, and handles training,
    evaluation, and registry persistence.
    """

    def __init__(self) -> None:
        self._registry = ModelRegistry()

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        ticker: str,
        feature_names: list[str],
        target_column: str = "direction",
        start_date: str | None = None,
        end_date: str | None = None,
        test_size: float | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load features and build train / test splits.

        The target is a binary label indicating the next-day price
        direction: ``1`` if the close price increases, ``0`` otherwise.

        Args:
            ticker: Instrument symbol.
            feature_names: List of feature column names to load.
            target_column: Name for the generated target column
                (default ``"direction"``).
            start_date: ISO start date (inclusive).
            end_date: ISO end date (inclusive).
            test_size: Fraction of data to reserve for testing.  Uses
                ``settings.model.default_test_size`` when *None*.

        Returns:
            ``(X_train, X_test, y_train, y_test)`` tuple of DataFrames
            and Series.

        Raises:
            ValueError: If insufficient data is available after feature
                computation.
        """
        from data_pipeline.loader import DataLoader
        from features.engine import FeatureEngine

        test_size = test_size or settings.model.default_test_size
        start_date = start_date or settings.pipeline.default_start_date

        # Load OHLCV data.
        loader = DataLoader()
        df = loader.load_ohlcv(ticker, start_date=start_date, end_date=end_date)

        if df.empty:
            raise ValueError(f"No OHLCV data found for {ticker}.")

        # Compute features.
        engine = FeatureEngine()
        feature_df = engine.compute_features(df, feature_names=feature_names)

        # Create target: 1 if next-day return > 0, else 0.
        feature_df[target_column] = (
            df["close"].shift(-1) > df["close"]
        ).astype(int)

        # Drop rows with NaN (rolling window warm-up + last row).
        feature_df = feature_df.dropna()

        if len(feature_df) < 30:
            raise ValueError(
                f"Only {len(feature_df)} rows after cleaning -- need at least 30."
            )

        # Time-aware split: last `test_size` fraction of rows.
        split_idx = int(len(feature_df) * (1 - test_size))
        X = feature_df[feature_names]
        y = feature_df[target_column]

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(
            "Data prepared for %s: %d train, %d test samples, %d features.",
            ticker,
            len(X_train),
            len(X_test),
            len(feature_names),
        )

        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(
        self,
        model: BaseModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """Train a model on the provided data.

        Args:
            model: An untrained :class:`BaseModel` instance.
            X_train: Training feature matrix.
            y_train: Training target series.
        """
        logger.info("Training model %r ...", model.name)
        model.train(X_train, y_train)
        logger.info("Model %r training complete.", model.name)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model: BaseModel,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        """Evaluate a trained model and return metrics.

        Args:
            model: A trained :class:`BaseModel` instance.
            X_test: Test feature matrix.
            y_test: Test target series.

        Returns:
            Dictionary of evaluation metrics.
        """
        metrics = model.evaluate(X_test, y_test)
        logger.info(
            "Model %r evaluated: accuracy=%.4f, f1=%.4f",
            model.name,
            metrics.get("accuracy", 0),
            metrics.get("f1", 0),
        )
        return metrics

    # ------------------------------------------------------------------
    # End-to-end
    # ------------------------------------------------------------------

    def train_and_register(
        self,
        model: BaseModel,
        ticker: str,
        feature_names: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        test_size: float | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Full pipeline: prepare data -> train -> evaluate -> register.

        Args:
            model: An untrained model instance.
            ticker: Instrument symbol.
            feature_names: Feature columns to use.
            start_date: ISO start date.
            end_date: ISO end date.
            test_size: Fraction of data for testing.
            description: Free-text description for the registry.

        Returns:
            Dictionary with ``metrics``, ``training_info``, and
            ``registry_row`` keys.
        """
        X_train, X_test, y_train, y_test = self.prepare_data(
            ticker=ticker,
            feature_names=feature_names,
            start_date=start_date,
            end_date=end_date,
            test_size=test_size,
        )

        self.train_model(model, X_train, y_train)
        metrics = self.evaluate_model(model, X_test, y_test)

        training_info = {
            "ticker": ticker,
            "features": feature_names,
            "start_date": start_date or settings.pipeline.default_start_date,
            "end_date": end_date or "latest",
            "test_size": test_size or settings.model.default_test_size,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        registry_row = self._registry.register(
            model=model,
            training_info=training_info,
            metrics=metrics,
            description=description,
        )

        logger.info(
            "Model %r trained, evaluated, and registered (id=%d).",
            model.name,
            registry_row.id,
        )

        return {
            "metrics": metrics,
            "training_info": training_info,
            "registry_row": registry_row,
        }

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> dict[str, list[float]]:
        """Perform time-series cross-validation.

        Uses :class:`sklearn.model_selection.TimeSeriesSplit` to respect
        temporal ordering (no look-ahead).

        Args:
            model: A :class:`BaseModel` instance (a fresh copy is
                created for each fold internally by re-instantiating).
            X: Full feature matrix.
            y: Full target series.
            n_splits: Number of folds.

        Returns:
            Dictionary mapping metric names to lists of per-fold values.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics: dict[str, list[float]] = {}

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Create a fresh model instance with the same hyperparameters.
            fresh_model = model.__class__(
                name=f"{model.name}_fold{fold}",
                **{
                    k: v
                    for k, v in model.get_hyperparameters().items()
                    if k not in ("use_label_encoder", "eval_metric", "n_jobs")
                },
            )
            fresh_model.train(X_train, y_train)
            metrics = fresh_model.evaluate(X_test, y_test)

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    fold_metrics.setdefault(key, []).append(float(value))

            logger.info(
                "CV fold %d/%d: accuracy=%.4f, f1=%.4f",
                fold + 1,
                n_splits,
                metrics.get("accuracy", 0),
                metrics.get("f1", 0),
            )

        return fold_metrics
