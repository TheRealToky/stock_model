"""Strategy registry -- persists strategy metadata in the database.

Each registered strategy stores its name, description, parameters,
dependencies, performance metrics, and the last evaluation timestamp.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import text

from config.settings import settings
from database.connection import get_engine
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """CRUD interface for the ``strategy_registry`` database table."""

    def __init__(self) -> None:
        self._engine = get_engine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        strategy: BaseStrategy,
        performance_metrics: dict[str, Any] | None = None,
    ) -> int:
        """Register (or update) a strategy in the database.

        Args:
            strategy: A :class:`BaseStrategy` instance.
            performance_metrics: Optional dict of backtest metrics.

        Returns:
            The ``id`` of the inserted / updated row.
        """
        params = json.dumps(strategy.get_parameters())
        deps = json.dumps(strategy.get_dependencies())
        metrics_json = json.dumps(
            self._make_json_safe(performance_metrics or {})
        )
        now = datetime.now(timezone.utc)

        with self._engine.connect() as conn:
            existing = conn.execute(
                text("SELECT id FROM strategy_registry WHERE strategy_name = :name"),
                {"name": strategy.name},
            ).fetchone()

            if existing is not None:
                conn.execute(
                    text("""
                        UPDATE strategy_registry
                        SET description = :desc,
                            parameters = :params,
                            dependencies = :deps,
                            performance_metrics = :metrics,
                            last_evaluated_at = :now,
                            updated_at = :now
                        WHERE strategy_name = :name
                    """),
                    {
                        "name": strategy.name,
                        "desc": strategy.description,
                        "params": params,
                        "deps": deps,
                        "metrics": metrics_json,
                        "now": now,
                    },
                )
                conn.commit()
                strategy_id = existing[0]
                logger.info("Updated strategy '%s' (id=%d).", strategy.name, strategy_id)
            else:
                row = conn.execute(
                    text("""
                        INSERT INTO strategy_registry
                            (strategy_name, description, parameters, dependencies,
                             performance_metrics, last_evaluated_at, created_at, updated_at)
                        VALUES (:name, :desc, :params, :deps, :metrics, :now, :now, :now)
                        RETURNING id
                    """),
                    {
                        "name": strategy.name,
                        "desc": strategy.description,
                        "params": params,
                        "deps": deps,
                        "metrics": metrics_json,
                        "now": now,
                    },
                ).fetchone()
                conn.commit()
                strategy_id = row[0]
                logger.info("Registered strategy '%s' (id=%d).", strategy.name, strategy_id)

            return strategy_id

    def get(self, strategy_name: str) -> dict[str, Any] | None:
        """Load strategy metadata by name.

        Args:
            strategy_name: Unique strategy identifier.

        Returns:
            Dictionary of all columns, or *None* if not found.
        """
        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM strategy_registry WHERE strategy_name = :name"),
                {"name": strategy_name},
            ).mappings().fetchone()

        if row is None:
            return None
        return dict(row)

    def list_all(self) -> pd.DataFrame:
        """List all registered strategies.

        Returns:
            DataFrame with one row per strategy.
        """
        with self._engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM strategy_registry ORDER BY created_at DESC")
            )
            rows = result.mappings().all()

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    def update_metrics(
        self,
        strategy_name: str,
        metrics: dict[str, Any],
    ) -> None:
        """Update performance metrics for a strategy.

        Args:
            strategy_name: Unique strategy identifier.
            metrics: New performance metrics dictionary.
        """
        metrics_json = json.dumps(self._make_json_safe(metrics))
        now = datetime.now(timezone.utc)

        with self._engine.connect() as conn:
            conn.execute(
                text("""
                    UPDATE strategy_registry
                    SET performance_metrics = :metrics,
                        last_evaluated_at = :now,
                        updated_at = :now
                    WHERE strategy_name = :name
                """),
                {"name": strategy_name, "metrics": metrics_json, "now": now},
            )
            conn.commit()
            logger.info("Updated metrics for strategy '%s'.", strategy_name)

    def deregister(self, strategy_name: str) -> bool:
        """Remove a strategy from the registry.

        Args:
            strategy_name: Unique strategy identifier.

        Returns:
            *True* if a row was deleted, *False* if nothing was found.
        """
        with self._engine.connect() as conn:
            result = conn.execute(
                text("DELETE FROM strategy_registry WHERE strategy_name = :name"),
                {"name": strategy_name},
            )
            conn.commit()
            deleted = result.rowcount > 0

        if deleted:
            logger.info("Deregistered strategy '%s'.", strategy_name)
        else:
            logger.warning("Strategy '%s' not found -- nothing to delete.", strategy_name)
        return deleted

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_json_safe(obj: Any) -> Any:
        """Recursively convert numpy / non-JSON types to primitives."""
        import numpy as np

        if isinstance(obj, dict):
            return {str(k): StrategyRegistry._make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [StrategyRegistry._make_json_safe(i) for i in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
