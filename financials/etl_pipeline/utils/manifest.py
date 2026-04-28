"""Persistent state for incremental ETL runs.

The manifest is a small JSON file kept at the root of the Parquet dataset.
For each ticker it stores the timestamp of the latest bar that has been
materialized to disk; subsequent incremental runs use that as the lower
bound (plus a warm-up buffer) for the next extract.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from financials.etl_pipeline.utils.logging import get_logger

logger = get_logger(__name__)

_SCHEMA_VERSION = 1


class Manifest:
    """Read/write the dataset manifest in an atomic, idempotent way."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._state: dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {
                "schema_version": _SCHEMA_VERSION,
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "tickers": {},
            }
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Manifest at {} is unreadable -- starting fresh: {}", self.path, exc)
            return {
                "schema_version": _SCHEMA_VERSION,
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "tickers": {},
            }

        if data.get("schema_version") != _SCHEMA_VERSION:
            logger.warning(
                "Manifest schema mismatch (got {}, expected {}). Continuing best-effort.",
                data.get("schema_version"),
                _SCHEMA_VERSION,
            )
        data.setdefault("tickers", {})
        return data

    def save(self) -> None:
        """Atomically write the manifest to disk (write-then-rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state["updated_at"] = _now_iso()

        # Use NamedTemporaryFile in the same directory so os.replace() is atomic.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=self.path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(self._state, tmp, indent=2, sort_keys=True, default=str)
            tmp_path = Path(tmp.name)

        tmp_path.replace(self.path)
        logger.debug("Manifest saved -> {}", self.path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_high_water(self, ticker: str) -> datetime | None:
        """Return the latest processed timestamp for *ticker* (UTC)."""
        entry = self._state["tickers"].get(ticker)
        if not entry:
            return None
        ts = entry.get("last_timestamp")
        if not ts:
            return None
        return datetime.fromisoformat(ts)

    def update(
        self,
        ticker: str,
        last_timestamp: datetime,
        rows_written: int,
    ) -> None:
        """Record progress for one ticker. Caller decides when to ``save()``."""
        if last_timestamp.tzinfo is None:
            last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)

        existing = self._state["tickers"].get(ticker, {"total_rows": 0})
        self._state["tickers"][ticker] = {
            "last_timestamp": last_timestamp.isoformat(),
            "total_rows": int(existing.get("total_rows", 0)) + int(rows_written),
            "last_run_at": _now_iso(),
        }

    def all_tickers(self) -> list[str]:
        return sorted(self._state["tickers"].keys())

    def summary(self) -> dict[str, Any]:
        return {
            "tickers_tracked": len(self._state["tickers"]),
            "total_rows": sum(
                int(v.get("total_rows", 0)) for v in self._state["tickers"].values()
            ),
            "updated_at": self._state.get("updated_at"),
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
