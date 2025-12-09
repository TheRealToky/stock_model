import os
import psycopg2
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
from dotenv import load_dotenv
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

load_dotenv()

# DB_NAME = os.getenv("POSTGRES_DB")
# DB_USER = os.getenv("POSTGRES_USER")
# DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = "timescaledb"
DB_PORT = 5432

print(f"The name is motherfucking: {DB_NAME}")
print(f"The user is motherfucking: {DB_USER}")
print(f"The password is motherfucking: {DB_PASSWORD}")

def create_database():
    """
    Creates a new database if it doesn't exist yet.
    """
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}';")
    exists = cursor.fetchone()

    if not exists:
        cursor.execute(f"CREATE DATABASE {DB_NAME};")
        print(f"Database '{DB_NAME}' created successfully.")
    else:
        print(f"Database '{DB_NAME}' already exists.")


def setup_tables():
    """
    Creates simple tables for stocks and daily OHLCV.
    """
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id SERIAL,
            ticker TEXT PRIMARY KEY,
            company_name TEXT,
            market_cap NUMERIC,
            country TEXT
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            ticker TEXT NOT NULL REFERENCES companies(ticker),
            timestamp TIMESTAMPTZ NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume BIGINT,
            dividends REAL,
            stock_splits REAL,
            PRIMARY KEY (ticker, timestamp)
        );

        SELECT create_hypertable(
            'ohlcv',
            'timestamp',
            chunk_time_interval => interval '7 days',
            partitioning_column => 'ticker',
            number_partitions => 2000
        );
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("Tables created.")


def main():
    create_database()
    setup_tables()


if __name__ == "__main__":
    main()