import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy import inspect
from db.base import Base
from db.session import engine
import db.models

load_dotenv()

POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = 5432

def create_database():
    """
    Creates a new database if it doesn't exist yet.
    """
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB  # connect to default DB first
    )

    engine = create_engine(url, isolation_level="AUTOCOMMIT")

    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
            {"dbname": POSTGRES_DB}
        )

        if not result.scalar():
            conn.execute(text(f'CREATE DATABASE "{POSTGRES_DB}"'))
            print(f"Database '{POSTGRES_DB}' created.")
        else:
            print(f"Database '{POSTGRES_DB}' already exists.")


def setup_tables():
    """
    Creates simple tables for stocks and daily OHLCV.
    """

    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Tables created: {tables}")

    if "ohlcv" not in tables:
        raise RuntimeError("Table 'ohlcv' was not created successfully")
    print("Tables created.")

    with engine.connect() as conn:
        conn.execute(text("""
            SELECT create_hypertable(
                'ohlcv',
                'timestamp',
                chunk_time_interval => interval '7 days',
                partitioning_column => 'ticker',
                number_partitions => 2000
            );
        """))
    print("Hypertable assigned.")


def main():
    create_database()
    setup_tables()


if __name__ == "__main__":
    main()