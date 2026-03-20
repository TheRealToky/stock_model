"""SQLAlchemy engine and session factory for the quant-lab database."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config.settings import settings


def get_engine(echo: bool = False):
    """Create and return a SQLAlchemy engine using the database config.

    Args:
        echo: If True, the engine logs all SQL statements to stdout.

    Returns:
        A SQLAlchemy Engine instance.
    """
    return create_engine(
        settings.db.url,
        echo=echo,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )


# Default engine instance (lazy -- created once on first import)
engine = get_engine()

# Session factory bound to the default engine
SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_session() -> Session:
    """Open a new database session.

    Usage::

        session = get_session()
        try:
            ...
            session.commit()
        finally:
            session.close()

    Or as a context-manager style helper::

        with get_session() as session:
            ...
    """
    return SessionLocal()
