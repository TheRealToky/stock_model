"""Apply pending SQL migrations to the alt-data database.

Usage::

    python -m alt_data_pipeline.src.storage.migrate
    python -m alt_data_pipeline.src.storage.migrate --status
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import text

from alt_data_pipeline.src.storage.connection import get_alt_engine
from alt_data_pipeline.src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def _ensure_table(conn) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version     VARCHAR(50) PRIMARY KEY,
                description TEXT,
                applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
    )
    conn.commit()


def _applied(conn) -> set[str]:
    rows = conn.execute(
        text("SELECT version FROM schema_migrations ORDER BY version")
    ).fetchall()
    return {r[0] for r in rows}


def _discover() -> list[Path]:
    if not MIGRATIONS_DIR.is_dir():
        logger.warning("Migrations directory missing: {}", MIGRATIONS_DIR)
        return []
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def _version(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def show_status() -> None:
    engine = get_alt_engine()
    with engine.connect() as conn:
        _ensure_table(conn)
        applied = _applied(conn)

    for p in _discover():
        mark = "applied" if _version(p) in applied else "PENDING"
        print(f"  [{mark:>7}]  {p.name}")


def run_migrations() -> None:
    engine = get_alt_engine()
    with engine.connect() as conn:
        _ensure_table(conn)
        applied = _applied(conn)

    pending = [p for p in _discover() if _version(p) not in applied]
    if not pending:
        logger.info("alt-data schema is up to date")
        print("All migrations are up to date.")
        return

    for path in pending:
        version = _version(path)
        logger.info("Applying alt migration {} ({})", version, path.name)
        sql = path.read_text(encoding="utf-8")
        with engine.connect() as conn:
            raw = conn.connection
            cur = raw.cursor()
            try:
                cur.execute(sql)
                raw.commit()
            except Exception:
                raw.rollback()
                logger.exception("Migration {} failed", version)
                sys.exit(1)
            finally:
                cur.close()
        logger.info("Migration {} applied", version)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alt-data DB migrations")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    configure_logging()
    if args.status:
        show_status()
    else:
        run_migrations()


if __name__ == "__main__":
    main()
