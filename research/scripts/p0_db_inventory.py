"""Phase 0 - Ground truth: inventory the TimescaleDB OHLCV store.

Run inside the python-app container:
    docker compose exec python-app python research/scripts/p0_db_inventory.py
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from financials.database.connection import get_engine


def q(engine, sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def main() -> None:
    engine = get_engine()

    print("=" * 70)
    print("TABLES")
    print("=" * 70)
    print(q(engine, """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema='public'
        ORDER BY table_name
    """).to_string(index=False))

    print("\n" + "=" * 70)
    print("INSTRUMENTS")
    print("=" * 70)
    instr = q(engine, "SELECT count(*) AS n FROM instruments")
    print(f"instrument rows: {instr['n'].iloc[0]}")

    print("\n" + "=" * 70)
    print("OHLCV: rows per interval")
    print("=" * 70)
    print(q(engine, """
        SELECT interval,
               count(*)                         AS rows,
               count(DISTINCT instrument_id)    AS tickers,
               min(timestamp)                   AS first_ts,
               max(timestamp)                   AS last_ts
        FROM ohlcv_data
        GROUP BY interval
        ORDER BY interval
    """).to_string(index=False))

    print("\n" + "=" * 70)
    print("OHLCV 1min: per-ticker coverage (top 60 by rows)")
    print("=" * 70)
    per_ticker = q(engine, """
        SELECT i.ticker,
               count(*)            AS rows,
               min(o.timestamp)    AS first_ts,
               max(o.timestamp)    AS last_ts
        FROM ohlcv_data o
        JOIN instruments i ON i.id = o.instrument_id
        WHERE o.interval = '1min'
        GROUP BY i.ticker
        ORDER BY rows DESC
    """)
    print(f"distinct 1min tickers: {len(per_ticker)}")
    print(per_ticker.head(60).to_string(index=False))
    per_ticker.to_csv("research/outputs/p0_1min_per_ticker.csv", index=False)

    # Session calendar probe: minute-of-day histogram for a liquid name.
    print("\n" + "=" * 70)
    print("SESSION CALENDAR PROBE (AAPL 1min): hour-of-day in UTC")
    print("=" * 70)
    cal = q(engine, """
        SELECT EXTRACT(hour FROM o.timestamp)::int AS utc_hour,
               count(*) AS rows
        FROM ohlcv_data o
        JOIN instruments i ON i.id = o.instrument_id
        WHERE o.interval='1min' AND i.ticker='AAPL'
        GROUP BY 1 ORDER BY 1
    """)
    print(cal.to_string(index=False))

    # How many distinct calendar dates, and bars-per-day distribution for AAPL.
    print("\n" + "=" * 70)
    print("AAPL 1min: bars-per-day distribution")
    print("=" * 70)
    bpd = q(engine, """
        WITH d AS (
            SELECT date_trunc('day', o.timestamp) AS day, count(*) AS bars
            FROM ohlcv_data o
            JOIN instruments i ON i.id=o.instrument_id
            WHERE o.interval='1min' AND i.ticker='AAPL'
            GROUP BY 1
        )
        SELECT count(*) AS trading_days,
               min(bars) AS min_bars, max(bars) AS max_bars,
               round(avg(bars),1) AS avg_bars,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY bars) AS median_bars
        FROM d
    """)
    print(bpd.to_string(index=False))


if __name__ == "__main__":
    main()
