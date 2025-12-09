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

    Path("/data/ohlcv").mkdir(parents=True, exist_ok=True)
    path = Path("/data/ohlcv")

    for symbol in top_companies["ticker"]:
        today = dt.datetime.today()  # runs every saturday

        weekday_number = today.weekday()

        if weekday_number == 5:
            try:
                print(f"Fetching weekly {symbol} data...")

                tick = yf.Ticker(symbol)

                raw_ohlcv = tick.history(interval="1d", period="5d", end=today)
                df_ohlcv = pd.DataFrame(raw_ohlcv)
                df_ohlcv = df_ohlcv.reset_index()

                df_ohlcv.drop_duplicates(subset="Date", keep='last', inplace=True)
                df_ohlcv.insert(0, "ticker", symbol)

                csv_buffer = StringIO()
                df_ohlcv.to_csv(csv_buffer, index=False, header=False)
                df_ohlcv.to_csv(path, mode="a", index=False, header=False)
                csv_buffer.seek(0)

                cursor.copy_from(csv_buffer, "ohlcv", sep=",", null="")
            except Exception as e:
                print(f"[!] Failed to fetch weekly data for ticker {symbol}:")
                print(f"[!] {e}")
                failed_list.append(symbol)
        else:
            return None

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()