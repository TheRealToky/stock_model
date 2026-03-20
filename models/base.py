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
    precision_score,
    recall_score,
)


class BaseModel(ABC):
    """Abstract base class that every model implementation must extend.

    Subclasses provide a concrete ML algorithm (XGBoost, Random Forest, etc.)
    while exposing a uniform train / predict / evaluate / save / load interface
    so that the rest of the pipeline (trainer, registry, backtester) can work
    with any model interchangeably.
    """

    name: str
    model_type: str

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model on training data.

        Args:
            X: Feature matrix (rows = samples, columns = features).
            y: Target series (binary: 1 = up, 0 = down).
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the given feature matrix.

        Args:
            X: Feature matrix with the same columns used during training.

        Returns:
            1-D numpy array of predicted labels.
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class-probability estimates.

        Args:
            X: Feature matrix.

        Returns:
            2-D numpy array of shape ``(n_samples, n_classes)``.
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

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Evaluate the model and return a dictionary of classification metrics.

        Metrics returned:
            - accuracy
            - precision  (weighted)
            - recall     (weighted)
            - f1         (weighted)
            - confusion_matrix (as nested list for JSON serialisation)
            - classification_report (as a formatted string)

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            Dictionary of metric name -> value.
        """
        preds = self.predict(X)

        return {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, average="weighted", zero_division=0)),
            "recall": float(recall_score(y, preds, average="weighted", zero_division=0)),
            "f1": float(f1_score(y, preds, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, preds).tolist(),
            "classification_report": classification_report(y, preds, zero_division=0),
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name!r})>"
