"""XGBoost classifier model for direction prediction (up / down)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from models.base import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """Wrapper around :class:`xgboost.XGBClassifier` for binary direction
    prediction.

    Parameters
    ----------
    name:
        A unique identifier for this model instance (used by the registry).
    n_estimators:
        Number of boosting rounds.
    max_depth:
        Maximum tree depth.
    learning_rate:
        Step-size shrinkage.
    subsample:
        Row sub-sampling ratio per tree.
    colsample_bytree:
        Column sub-sampling ratio per tree.
    random_state:
        Seed for reproducibility.
    **kwargs:
        Extra keyword arguments forwarded to ``XGBClassifier``.
    """

    model_type: str = "xgboost"

    def __init__(
        self,
        name: str,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            **kwargs,
        }
        self._model = XGBClassifier(**self._params)
        logger.info("Initialised XGBoostModel %r with params %s", name, self._params)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the XGBoost classifier."""
        logger.info(
            "Training XGBoostModel %r on %d samples, %d features",
            self.name,
            len(X),
            X.shape[1],
        )
        self._model.fit(X, y)
        logger.info("Training complete for %r.", self.name)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels (0 or 1)."""
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class-probability estimates ``(n_samples, 2)``."""
        return self._model.predict_proba(X)

    def get_hyperparameters(self) -> dict[str, Any]:
        """Return the hyperparameter dict used at initialisation."""
        return dict(self._params)

    def save(self, path: str) -> None:
        """Persist the fitted model to disk via :mod:`joblib`."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        logger.info("Saved XGBoostModel %r to %s", self.name, path)

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        """Load a previously saved model and wrap it in a new instance."""
        model_obj = joblib.load(path)
        instance = cls.__new__(cls)
        instance._model = model_obj
        instance._params = model_obj.get_params()
        instance.name = Path(path).stem
        logger.info("Loaded XGBoostModel from %s", path)
        return instance

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def feature_importances(self) -> np.ndarray | None:
        """Return feature importances if the model has been trained."""
        try:
            return self._model.feature_importances_
        except AttributeError:
            return None
