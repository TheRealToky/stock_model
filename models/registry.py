"""Model registry -- persists trained-model metadata in the database and
manages the corresponding serialised model files on disk."""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sqlalchemy import (
    Boolean,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
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
    __table_args__ = (
        UniqueConstraint("model_name", "version", name="ix_model_registry_name_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    hyperparameters: Mapped[dict] = mapped_column(JSONB, nullable=True)
    training_dataset_info: Mapped[dict] = mapped_column(JSONB, nullable=True)
    performance_metrics: Mapped[dict] = mapped_column(JSONB, nullable=True)
    model_path: Mapped[str] = mapped_column(String(500), nullable=False)
    # Versioning (migration 002). register() inserts a new row per run
    # instead of overwriting, so experiment history survives a retrain.
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    artifact_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    scaler_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    is_latest: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
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
    from models.random_forest_model import RandomForestModel
    from models.xgboost_model import XGBoostModel

    _MODEL_CLASS_MAP["xgboost"] = XGBoostModel
    _MODEL_CLASS_MAP["random_forest"] = RandomForestModel

    # LSTMModel needs torch, which is optional outside the GPU container.
    # Register it when available rather than making the whole registry
    # unimportable -- but without this, load_model() on any LSTM raised
    # "Unknown model_type 'lstm'", so the neural models could be written to
    # the registry and never read back.
    try:
        from models.lstm_model import LSTMModel

        _MODEL_CLASS_MAP["lstm"] = LSTMModel
    except ImportError:  # pragma: no cover - torch not installed
        logger.debug("torch unavailable; 'lstm' not registered in the class map.")


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
        *,
        scaler: Any | None = None,
        run_id: str | None = None,
    ) -> ModelRegistryRow:
        """Save the model artefact to disk and record a NEW version in the DB.

        This inserts rather than overwrites. The previous behaviour upserted on
        ``model_name``, so every retrain destroyed the prior run's metrics and
        left no way to compare two experiments or roll back -- which for a
        research lab is the one thing a registry exists to prevent.

        Args:
            model: A trained :class:`~models.base.BaseModel` instance.
            training_info: Arbitrary dict describing the training data
                (ticker, date range, feature list, split boundaries, ...).
            metrics: Evaluation metrics, ideally from
                :meth:`~models.base.BaseModel.evaluate` with ``bars`` supplied
                so the row records net-of-cost performance, not just accuracy.
            description: Optional free-text description.
            scaler: Fitted feature scaler used during training. Persisted
                beside the model, because an artefact that cannot reproduce
                its own input normalisation cannot be served.
            run_id: Identifier for the training run. Generated when omitted.

        Returns:
            The newly created :class:`ModelRegistryRow`.
        """
        run_id = run_id or uuid.uuid4().hex[:16]

        session = get_session()
        try:
            next_version = self._next_version(session, model.name)

            # Version-stamped filenames so artefacts never clobber each other.
            suffix = _artifact_suffix(model)
            stem = f"{model.name}_v{next_version}"
            model_path = str(self._models_dir / f"{stem}{suffix}")
            model.save(model_path)

            scaler_path: str | None = None
            if scaler is not None:
                scaler_path = str(self._models_dir / f"{stem}_scaler.joblib")
                joblib.dump(scaler, scaler_path)

            # Demote the incumbent before promoting this run, so the partial
            # unique index on is_latest never sees two current rows.
            session.query(ModelRegistryRow).filter_by(
                model_name=model.name, is_latest=True
            ).update({"is_latest": False}, synchronize_session=False)

            row = ModelRegistryRow(
                model_name=model.name,
                model_type=model.model_type,
                description=description,
                hyperparameters=_make_json_safe(model.get_hyperparameters()),
                training_dataset_info=_make_json_safe(training_info),
                performance_metrics=_make_json_safe(metrics),
                model_path=model_path,
                version=next_version,
                run_id=run_id,
                artifact_hash=_file_sha256(model_path),
                scaler_path=scaler_path,
                is_latest=True,
            )
            session.add(row)
            session.commit()
            session.refresh(row)

            logger.info(
                "Registered model %r version %d (run_id=%s).",
                model.name, next_version, run_id,
            )
            return row
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _next_version(session: Any, model_name: str) -> int:
        """Next version number for *model_name* (1 when it is new)."""
        current = (
            session.query(ModelRegistryRow.version)
            .filter_by(model_name=model_name)
            .order_by(ModelRegistryRow.version.desc())
            .first()
        )
        return 1 if current is None else int(current[0]) + 1

    def list_versions(self, model_name: str) -> list[ModelRegistryRow]:
        """Every registered version of *model_name*, newest first.

        Args:
            model_name: Model identifier.

        Returns:
            Rows ordered by descending version.
        """
        session = get_session()
        try:
            return (
                session.query(ModelRegistryRow)
                .filter_by(model_name=model_name)
                .order_by(ModelRegistryRow.version.desc())
                .all()
            )
        finally:
            session.close()

    def load_scaler(self, model_name: str, version: int | None = None) -> Any:
        """Load the fitted scaler stored beside a model version.

        Args:
            model_name: Model identifier.
            version: Version to load. ``None`` uses the current one.

        Returns:
            The deserialised scaler.

        Raises:
            ValueError: If the model or version is not registered, or no
                scaler was saved with it.
            FileNotFoundError: If the scaler file is missing from disk.
        """
        row = self.get(model_name, version=version)
        if row is None:
            raise ValueError(f"Model {model_name!r} version {version!r} not found.")
        if not row.scaler_path:
            raise ValueError(
                f"No scaler recorded for {model_name!r} v{row.version}. Predictions "
                "cannot reproduce training-time normalisation without it; re-register "
                "the model passing scaler=<fitted scaler>."
            )
        if not os.path.isfile(row.scaler_path):
            raise FileNotFoundError(f"Scaler file missing at {row.scaler_path!r}.")
        return joblib.load(row.scaler_path)

    def get(
        self, model_name: str, version: int | None = None
    ) -> ModelRegistryRow | None:
        """Fetch model metadata from the database.

        Args:
            model_name: Model identifier.
            version: Specific version. ``None`` returns the current one
                (``is_latest``), falling back to the highest version number if
                no row is flagged.

        Returns:
            ``ModelRegistryRow`` if found, otherwise ``None``.
        """
        session = get_session()
        try:
            q = session.query(ModelRegistryRow).filter_by(model_name=model_name)
            if version is not None:
                return q.filter_by(version=version).first()
            return (
                q.filter_by(is_latest=True).first()
                or q.order_by(ModelRegistryRow.version.desc()).first()
            )
        finally:
            session.close()

    def load_model(self, model_name: str, version: int | None = None) -> BaseModel:
        """Load a trained model from disk using the registry metadata.

        Args:
            model_name: Model identifier.
            version: Version to load. ``None`` loads the current one.

        Returns:
            A fully initialised ``BaseModel`` subclass instance.

        Raises:
            ValueError: If the model, version, or model type is unknown.
            FileNotFoundError: If the serialised artefact is missing.
        """
        row = self.get(model_name, version=version)
        if row is None:
            raise ValueError(
                f"Model {model_name!r} version {version!r} not found in registry."
            )

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
            rows = (
                session.query(ModelRegistryRow)
                .order_by(ModelRegistryRow.created_at.desc())
                .all()
            )
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


def _artifact_suffix(model: BaseModel) -> str:
    """File extension appropriate to how *model* serialises itself.

    Torch models write a ``.pt`` checkpoint via ``torch.save``; sklearn-family
    models use joblib. The registry previously hard-coded ``.joblib`` for
    everything, which mislabelled every LSTM checkpoint it wrote.
    """
    return ".pt" if getattr(model, "model_type", "") == "lstm" else ".joblib"


def _file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, so a registry row can be tied to bytes on disk."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
