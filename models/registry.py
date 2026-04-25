"""Model registry -- persists trained-model metadata in the database and
manages the corresponding serialised model files on disk."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from financials.config.settings import settings
from financials.database.connection import get_engine, get_session
from models.base import BaseModel

logger = logging.getLogger(__name__)


# ======================================================================
# ORM model for the ``model_registry`` table
# ======================================================================


class _Base(DeclarativeBase):
    pass


class ModelRegistryRow(_Base):
    """SQLAlchemy ORM mapping for the ``model_registry`` table."""

    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    hyperparameters: Mapped[dict] = mapped_column(JSONB, nullable=True)
    training_dataset_info: Mapped[dict] = mapped_column(JSONB, nullable=True)
    performance_metrics: Mapped[dict] = mapped_column(JSONB, nullable=True)
    model_path: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelRegistryRow(id={self.id}, name={self.model_name!r}, "
            f"type={self.model_type!r})>"
        )


# ======================================================================
# Registry service
# ======================================================================

# Map model_type strings to concrete classes so ``load_model`` can
# deserialise any registered model without the caller having to know
# which class to use.
_MODEL_CLASS_MAP: dict[str, type[BaseModel]] = {}


def _ensure_class_map() -> None:
    """Lazily populate ``_MODEL_CLASS_MAP`` so we avoid circular imports."""
    if _MODEL_CLASS_MAP:
        return
    from models.xgboost_model import XGBoostModel
    from models.random_forest_model import RandomForestModel

    _MODEL_CLASS_MAP["xgboost"] = XGBoostModel
    _MODEL_CLASS_MAP["random_forest"] = RandomForestModel


class ModelRegistry:
    """High-level interface for registering, loading, comparing, and deleting
    trained models.

    Model artefacts (joblib files) are stored under
    ``settings.model.models_dir``; metadata lives in the
    ``model_registry`` database table.
    """

    def __init__(self) -> None:
        self._models_dir = Path(settings.model.models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    # ------------------------------------------------------------------
    # Table bootstrapping
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        """Create the ``model_registry`` table if it doesn't exist yet."""
        engine = get_engine()
        _Base.metadata.create_all(engine, checkfirst=True)
        logger.debug("model_registry table ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        model: BaseModel,
        training_info: dict[str, Any],
        metrics: dict[str, Any],
        description: str | None = None,
    ) -> ModelRegistryRow:
        """Save the model artefact to disk and record metadata in the DB.

        Args:
            model: A trained ``BaseModel`` instance.
            training_info: Arbitrary dict describing the training data
                (ticker, date range, feature list, etc.).
            metrics: Evaluation metrics produced by ``model.evaluate()``.
            description: Optional free-text description.

        Returns:
            The created / updated ``ModelRegistryRow``.
        """
        # Serialise model to disk ------------------------------------------------
        model_filename = f"{model.name}.joblib"
        model_path = str(self._models_dir / model_filename)
        model.save(model_path)

        # Persist metadata -------------------------------------------------------
        session = get_session()
        try:
            existing: ModelRegistryRow | None = (
                session.query(ModelRegistryRow)
                .filter_by(model_name=model.name)
                .first()
            )

            # Strip non-JSON-serialisable values (e.g. numpy arrays) from metrics.
            safe_metrics = _make_json_safe(metrics)
            safe_hyperparams = _make_json_safe(model.get_hyperparameters())
            safe_training = _make_json_safe(training_info)

            if existing is not None:
                existing.model_type = model.model_type
                existing.description = description
                existing.hyperparameters = safe_hyperparams
                existing.training_dataset_info = safe_training
                existing.performance_metrics = safe_metrics
                existing.model_path = model_path
                existing.updated_at = datetime.now(timezone.utc)
                row = existing
                logger.info("Updated registry entry for model %r.", model.name)
            else:
                row = ModelRegistryRow(
                    model_name=model.name,
                    model_type=model.model_type,
                    description=description,
                    hyperparameters=safe_hyperparams,
                    training_dataset_info=safe_training,
                    performance_metrics=safe_metrics,
                    model_path=model_path,
                )
                session.add(row)
                logger.info("Created registry entry for model %r.", model.name)

            session.commit()
            session.refresh(row)
            return row
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get(self, model_name: str) -> ModelRegistryRow | None:
        """Fetch model metadata from the database.

        Args:
            model_name: Unique model identifier.

        Returns:
            ``ModelRegistryRow`` if found, otherwise ``None``.
        """
        session = get_session()
        try:
            row = (
                session.query(ModelRegistryRow)
                .filter_by(model_name=model_name)
                .first()
            )
            return row
        finally:
            session.close()

    def load_model(self, model_name: str) -> BaseModel:
        """Load a trained model from disk using the registry metadata.

        Args:
            model_name: Unique model identifier.

        Returns:
            A fully initialised ``BaseModel`` subclass instance.

        Raises:
            ValueError: If the model is not found in the registry.
            FileNotFoundError: If the serialised artefact is missing.
        """
        row = self.get(model_name)
        if row is None:
            raise ValueError(f"Model {model_name!r} not found in registry.")

        if not os.path.isfile(row.model_path):
            raise FileNotFoundError(
                f"Model artefact not found at {row.model_path!r}."
            )

        _ensure_class_map()
        model_cls = _MODEL_CLASS_MAP.get(row.model_type)
        if model_cls is None:
            raise ValueError(
                f"Unknown model_type {row.model_type!r} for model {model_name!r}. "
                f"Known types: {list(_MODEL_CLASS_MAP)}."
            )

        model = model_cls.load(row.model_path)
        model.name = model_name
        logger.info("Loaded model %r (type=%s) from registry.", model_name, row.model_type)
        return model

    def list_all(self) -> list[ModelRegistryRow]:
        """Return a list of all registered models (metadata only)."""
        session = get_session()
        try:
            rows = session.query(ModelRegistryRow).order_by(ModelRegistryRow.created_at.desc()).all()
            return rows
        finally:
            session.close()

    def compare(self, model_names: list[str]) -> pd.DataFrame:
        """Compare performance metrics of several models side by side.

        Args:
            model_names: List of model names to compare.

        Returns:
            A ``DataFrame`` indexed by metric name with one column per model.
        """
        records: list[dict[str, Any]] = []
        session = get_session()
        try:
            for name in model_names:
                row = (
                    session.query(ModelRegistryRow)
                    .filter_by(model_name=name)
                    .first()
                )
                if row is None:
                    logger.warning("Model %r not found -- skipping.", name)
                    continue
                metrics = dict(row.performance_metrics or {})
                # Exclude non-scalar metrics from the comparison table.
                scalar_metrics = {
                    k: v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                }
                scalar_metrics["model_type"] = row.model_type
                scalar_metrics["model_name"] = name
                records.append(scalar_metrics)
        finally:
            session.close()

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("model_name")
        return df

    def delete(self, model_name: str) -> bool:
        """Remove a model's artefact and database entry.

        Args:
            model_name: Unique model identifier.

        Returns:
            ``True`` if the model was found and deleted, ``False`` otherwise.
        """
        session = get_session()
        try:
            row = (
                session.query(ModelRegistryRow)
                .filter_by(model_name=model_name)
                .first()
            )
            if row is None:
                logger.warning("Model %r not found -- nothing to delete.", model_name)
                return False

            # Delete file on disk (best-effort)
            try:
                model_file = Path(row.model_path)
                if model_file.is_file():
                    model_file.unlink()
                    logger.info("Deleted model file %s", row.model_path)
            except OSError as exc:
                logger.error("Could not delete model file %s: %s", row.model_path, exc)

            session.delete(row)
            session.commit()
            logger.info("Deleted registry entry for model %r.", model_name)
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# ======================================================================
# Helpers
# ======================================================================


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert a dict / list tree so that all values are
    JSON-serialisable (needed for JSONB columns).

    Numpy scalars, arrays, and other non-primitive types are converted to
    their Python equivalents.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
