import pandas as pd
import yfinance as yf
import datetime as dt
import os
import psycopg2
from pathlib import Path
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

# DB_NAME = os.getenv("POSTGRES_DB")
# DB_USER = os.getenv("POSTGRES_USER")
# DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = "timescaledb"
DB_PORT = 5432

"""
Fetch weekly new stock numbers
"""

def download_raw_csv(symbol):
    print("Downloading raw data...")

    raw_path = Path("/data/raw/ohlcv")

    today = dt.datetime.today()  # runs every saturday

    tick = yf.Ticker(symbol)

    raw_ohlcv = tick.history(interval="1d", period="6d", end=today)
    df_ohlcv = pd.DataFrame(raw_ohlcv)
    df_ohlcv.to_csv((raw_path / f"{symbol}.csv"), mode="a", header=False)

    return df_ohlcv


def download_processed_csv(symbol, df_ohlcv):
    print("Downloading processed data...")

    processed_path = Path("/data/processed/ohlcv")

    df_ohlcv = df_ohlcv.reset_index()
    df_ohlcv.insert(0, "ticker", symbol)

    df_ohlcv.to_csv((processed_path / f"{symbol}.csv"), mode="a", index=False, header=False)

    return df_ohlcv


def insert_to_db(df_ohlcv, cursor):
    print("Inserting into db...")

    csv_buffer = StringIO()
    df_ohlcv.to_csv(csv_buffer, index=False, header=False)
    csv_buffer.seek(0)

    cursor.copy_from(csv_buffer, "ohlcv", sep=",", null="")


def main():
    top_companies = pd.read_csv(Path("/data/stock_lists/top_companies.csv"))

    failed_list = []

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

    cursor = conn.cursor()

    today = dt.datetime.today()  # runs every saturday

    weekday_number = today.weekday()

    if weekday_number == 5:
        for symbol in top_companies["ticker"]:
            try:
                print(f"Fetching weekly {symbol} data...")

                raw_ohlcv = download_raw_csv(symbol)
                processed_ohlcv = download_processed_csv(symbol, raw_ohlcv)
                insert_to_db(processed_ohlcv, cursor)

            except Exception as e:
                print(f"[!] Failed to fetch weekly data for ticker {symbol}:")
                print(f"[!] {e}")
    else:
        print("Today is not Saturday, retry another day.")

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
