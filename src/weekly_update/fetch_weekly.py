import pandas as pd
import yfinance as yf
import datetime as dt
from pathlib import Path
from io import StringIO
from db.session import engine

"""
Fetch weekly new stock numbers
"""


def download_raw_csv(symbol):
    raw_path = Path("/data/raw/ohlcv")

    try:
        print(f"Fetching raw {symbol} data...")
        today = dt.datetime.today()  # runs every saturday
        tick = yf.Ticker(symbol)
        raw_ohlcv = tick.history(interval="1d", period="6d", end=today)
        df_ohlcv = pd.DataFrame(raw_ohlcv)
        df_ohlcv.to_csv((raw_path / f"{symbol}.csv"), mode="a", header=False)
        return df_ohlcv
    except Exception as error:
        print(f"[!] Failed to download raw {symbol}:")
        print(f"[!] {error}")


def download_processed_csv(symbol, df_ohlcv):
    processed_path = Path("/data/processed/ohlcv")

    try:
        print(f"Processing {symbol} data...")
        df_ohlcv = df_ohlcv.reset_index()
        df_ohlcv.insert(0, "ticker", symbol)
        df_ohlcv.to_csv((processed_path / f"{symbol}.csv"), mode="a", index=False, header=False)

        return df_ohlcv
    except Exception as error:
        print(f"[!] Failed to process {symbol}:")
        print(f"[!] {error}")


def insert_to_db(symbol, df_ohlcv, engine):
    try:
        print(f"Inserting {symbol} to db...")
        csv_buffer = StringIO()
        df_ohlcv.to_csv(csv_buffer, index=False, header=False)
        csv_buffer.seek(0)

        with engine.raw_connection() as conn:
            with conn.cursor() as cursor:
                cursor.copy_from(
                    csv_buffer,
                    "ohlcv",
                    sep=",",
                    null=""
                )
            conn.commit()
    except Exception as error:
        print(f"[!] Failed to inserting {symbol} to db:")
        print(f"[!] {error}")


def main():
    top_companies = pd.read_csv(Path("/data/stock_lists/top_companies.csv"))

    failed_list = []

    today = dt.datetime.today()  # runs every saturday
    weekday_number = today.weekday()

    if weekday_number != 5 and weekday_number != 6:
        for symbol in top_companies["ticker"]:
            try:
                print(f"Fetching weekly {symbol} data...")

                raw_ohlcv = download_raw_csv(symbol)
                processed_ohlcv = download_processed_csv(symbol, raw_ohlcv)
                insert_to_db(symbol, processed_ohlcv, engine)

            except Exception as e:
                print(f"[!] Failed to fetch weekly data for ticker {symbol}:")
                print(f"[!] {e}")
    else:
        print("Today is a weekend day, retry another day.")


if __name__ == "__main__":
    main()
