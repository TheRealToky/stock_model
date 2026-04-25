"""SQLAlchemy engines + session factories for the alt-data pipeline.

We expose *two* engines:

* :func:`get_alt_engine`        - the alt-data PostgreSQL container
* :func:`get_financial_engine`  - the pre-existing OHLCV TimescaleDB

Keeping them separate means the query layer can join across databases
without conflating their connection pools.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from alt_data_pipeline.src.config.settings import alt_settings


@lru_cache(maxsize=1)
def get_alt_engine(echo: bool = False) -> Engine:
    """Return a cached engine for the alt-data Postgres instance."""
    return create_engine(
        alt_settings.db.url,
        echo=echo,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def get_financial_engine(echo: bool = False) -> Engine:
    """Return a cached engine for the existing financial DB (read-only use)."""
    return create_engine(
        alt_settings.financial_db.url,
        echo=echo,
        pool_size=3,
        max_overflow=5,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_alt_engine(),
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


def get_alt_session() -> Session:
    """Open a new session bound to the alt-data engine."""
    return _session_factory()()
