#!/usr/bin/env python3
"""Run SQL migrations against the quant-lab database.

Usage:
    python -m database.migrate            # Apply all pending migrations
    python -m database.migrate --status   # Show which migrations have been applied

Migrations live in ``database/migrations/`` and are executed in lexicographic
order.  Each file is expected to be idempotent (safe to re-run) and to insert
its own version marker into the ``schema_migrations`` table.

The script connects using the database URL from ``config.settings``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import sqlalchemy
from sqlalchemy import text

from config.settings import settings

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def _get_engine() -> sqlalchemy.Engine:
    """Build a SQLAlchemy engine from the project settings."""
    return sqlalchemy.create_engine(settings.db.url, echo=False)


def _ensure_migrations_table(conn: sqlalchemy.Connection) -> None:
    """Create the ``schema_migrations`` tracking table if it does not exist."""
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version     VARCHAR(50) PRIMARY KEY,
            description TEXT,
            applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """))
    conn.commit()


def _applied_versions(conn: sqlalchemy.Connection) -> set[str]:
    """Return the set of migration versions already applied."""
    rows = conn.execute(
        text("SELECT version FROM schema_migrations ORDER BY version")
    ).fetchall()
    return {row[0] for row in rows}


def _discover_migrations() -> list[Path]:
    """Return all ``.sql`` files in the migrations directory, sorted."""
    if not MIGRATIONS_DIR.is_dir():
        logger.warning("Migrations directory not found: %s", MIGRATIONS_DIR)
        return []
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def _version_from_path(path: Path) -> str:
    """Extract the version prefix from a migration filename.

    ``001_initial_schema.sql`` -> ``"001"``
    """
    return path.stem.split("_", 1)[0]


def show_status() -> None:
    """Print which migrations have been applied and which are pending."""
    engine = _get_engine()
    with engine.connect() as conn:
        _ensure_migrations_table(conn)
        applied = _applied_versions(conn)

    migrations = _discover_migrations()
    if not migrations:
        print("No migration files found.")
        return

    for path in migrations:
        version = _version_from_path(path)
        status = "applied" if version in applied else "PENDING"
        print(f"  [{status:>7}]  {path.name}")


def run_migrations() -> None:
    """Apply all pending migrations in order."""
    engine = _get_engine()

    with engine.connect() as conn:
        _ensure_migrations_table(conn)
        applied = _applied_versions(conn)

    migrations = _discover_migrations()
    pending = [
        m for m in migrations if _version_from_path(m) not in applied
    ]

    if not pending:
        logger.info("All migrations are up to date.")
        print("All migrations are up to date.")
        return

    for path in pending:
        version = _version_from_path(path)
        logger.info("Applying migration %s (%s) ...", version, path.name)
        print(f"Applying migration {version} ({path.name}) ...")

        sql = path.read_text(encoding="utf-8")

        # Each migration runs in its own connection/transaction.
        with engine.connect() as conn:
            # Use raw-connection execution so multi-statement SQL files work
            # correctly (e.g. DO $$ blocks, CREATE EXTENSION, etc.).
            raw_conn = conn.connection
            cursor = raw_conn.cursor()
            try:
                cursor.execute(sql)
                raw_conn.commit()
            except Exception:
                raw_conn.rollback()
                logger.exception(
                    "Migration %s failed.  Rolled back.", version
                )
                print(f"ERROR: Migration {version} failed. See logs.")
                sys.exit(1)
            finally:
                cursor.close()

        logger.info("Migration %s applied successfully.", version)
        print(f"Migration {version} applied successfully.")

    print("All pending migrations applied.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and execute the requested action."""
    parser = argparse.ArgumentParser(
        description="Run quant-lab database migrations."
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show migration status without applying anything.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.status:
        show_status()
    else:
        run_migrations()


if __name__ == "__main__":
    main()
