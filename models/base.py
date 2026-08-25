"""Abstract base class for all ML models in the quant lab."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

#: Prediction tasks a model can declare. Drives which metrics
#: :meth:`BaseModel.evaluate` computes.
TASKS = ("binary", "multiclass", "regression")

#: Returned in place of financial metrics when the caller supplies no prices.
NO_FINANCIAL_METRICS = (
    "NOT COMPUTED - no price bars supplied. Classification metrics alone cannot "
    "tell you whether this model makes money: a 53%-accurate classifier trading "
    "22x/day loses at any realistic cost, and its accuracy looks identical to a "
    "profitable one's. Pass bars=<OHLCV frame> and interval=<bar interval> to "
    "evaluate(), or call evaluate_strategy()."
)


class BaseModel(ABC):
    """Abstract base class that every model implementation must extend.

    Subclasses provide a concrete ML algorithm (XGBoost, Random Forest, LSTM,
    ...) while exposing a uniform train / predict / evaluate / save / load
    interface so that the rest of the pipeline (trainer, registry, backtester)
    can work with any model interchangeably.

    Attributes:
        name: Unique identifier used by the registry.
        model_type: Registry key for the concrete class (``"xgboost"``, ...).
        task: One of :data:`TASKS`. Defaults to ``"binary"``, which is what
            every model in the lab predicted before regression and
            triple-barrier (three-class) targets arrived.
    """

    name: str
    model_type: str
    task: str = "binary"

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model on training data.

        Args:
            X: Feature matrix (rows = samples, columns = features).
            y: Target series. Its meaning follows :attr:`task` -- class labels
                for ``"binary"``/``"multiclass"``, a continuous value (usually
                a forward return) for ``"regression"``.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the given feature matrix.

        Args:
            X: Feature matrix with the same columns used during training.

        Returns:
            1-D numpy array: predicted labels for classifiers, predicted
            values for regressors.
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class-probability estimates.

        Args:
            X: Feature matrix.

        Returns:
            2-D numpy array of shape ``(n_samples, n_classes)``.

        Raises:
            NotImplementedError: For regression models, which have no classes.
        """

    @abstractmethod
    def get_hyperparameters(self) -> dict[str, Any]:
        """Return the current hyperparameter dictionary."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model to *path*."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        """Load a previously saved model from *path* and return an instance."""

    # ------------------------------------------------------------------
    # Concrete helpers (shared by all subclasses)
    # ------------------------------------------------------------------

    def _statistical_metrics(self, y_true, y_pred, y_proba=None) -> dict[str, Any]:
        """Task-appropriate statistical metrics."""
        if self.task == "regression":
            mse = float(mean_squared_error(y_true, y_pred))
            true_arr = np.asarray(y_true, dtype=np.float64)
            pred_arr = np.asarray(y_pred, dtype=np.float64)
            return {
                "task": "regression",
                "rmse": float(np.sqrt(mse)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
                # The metric that actually maps to a trade: a regressor can
                # have a dismal R2 and still be tradeable if it gets the sign
                # right, and vice versa.
                "directional_accuracy": float(
                    np.mean(np.sign(pred_arr) == np.sign(true_arr))
                ),
            }

        out: dict[str, Any] = {
            "task": self.task,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(y_true, y_pred, zero_division=0),
        }
        if self.task == "binary" and y_proba is not None:
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except ValueError:
                # Single-class slice; AUC is undefined rather than zero.
                out["roc_auc"] = float("nan")
        return out

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        bars: pd.DataFrame | None = None,
        interval: str | None = None,
        threshold: float = 0.5,
        commission: float = 0.001,
        slippage: float = 0.0005,
        **backtest_kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate the model statistically and, when possible, financially.

        Statistical metrics depend on :attr:`task`:

        * ``binary`` / ``multiclass`` -- accuracy, weighted precision/recall/F1,
          confusion matrix, classification report, plus ROC-AUC for binary.
        * ``regression`` -- RMSE, MAE, R-squared and directional accuracy.

        The returned dict **always** carries a ``financial_metrics`` key. When
        *bars* is omitted it holds :data:`NO_FINANCIAL_METRICS` rather than
        being silently absent, so a report that cannot say whether the model
        makes money has to admit it. That omission is how a model shipped at
        0.533 accuracy with nobody noticing it had never been backtested.

        Args:
            X: Feature matrix.
            y: True targets.
            bars: OHLCV frame aligned to *X* -- enables the net-of-cost
                backtest.
            interval: Bar interval driving annualisation. Inferred from the
                *bars* index when omitted.
            threshold: Probability threshold for the position rule.
            commission: Proportional commission per leg.
            slippage: Proportional slippage per leg.
            **backtest_kwargs: Forwarded to
                :func:`~financials.backtesting.ml_adapter.backtest_predictions`.

        Returns:
            Dictionary of metric name -> value, including
            ``financial_metrics``.
        """
        preds = self.predict(X)

        proba_pos = None
        if self.task == "binary":
            try:
                proba = self.predict_proba(X)
                proba_pos = proba[:, 1] if proba.ndim == 2 else np.asarray(proba)
            except (NotImplementedError, AttributeError, IndexError):
                proba_pos = None

        metrics = self._statistical_metrics(y, preds, proba_pos)

        if bars is None:
            metrics["financial_metrics"] = NO_FINANCIAL_METRICS
            return metrics

        metrics["financial_metrics"] = self.evaluate_strategy(
            bars,
            proba=proba_pos if proba_pos is not None else preds,
            interval=interval,
            threshold=threshold,
            commission=commission,
            slippage=slippage,
            **backtest_kwargs,
        )
        return metrics

    def evaluate_strategy(
        self,
        bars: pd.DataFrame,
        proba: np.ndarray | pd.Series,
        *,
        interval: str | None = None,
        threshold: float = 0.5,
        commission: float = 0.001,
        slippage: float = 0.0005,
        **backtest_kwargs: Any,
    ) -> dict[str, Any]:
        """Score this model's predictions as a trading strategy, net of costs.

        Args:
            bars: OHLCV frame with a tz-aware ``DatetimeIndex``.
            proba: Positive-class probabilities (or signed predictions for a
                regressor) aligned to *bars*.
            interval: Bar interval driving annualisation.
            threshold: Probability above which to hold a long position.
            commission: Proportional commission per leg.
            slippage: Proportional slippage per leg.
            **backtest_kwargs: Forwarded to
                :func:`~financials.backtesting.ml_adapter.backtest_predictions`.

        Returns:
            Flat dict of strategy and benchmark metrics, including
            ``sharpe_edge`` and ``beats_benchmark``.
        """
        # Imported here so the model layer stays importable without the
        # backtesting package present.
        from financials.backtesting.ml_adapter import backtest_predictions

        result = backtest_predictions(
            bars,
            proba,
            interval=interval,
            threshold=threshold,
            commission=commission,
            slippage=slippage,
            label=self.name,
            **backtest_kwargs,
        )
        return result.row()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name!r}, task={self.task!r})>"
