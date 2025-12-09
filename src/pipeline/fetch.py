import pandas as pd
import yfinance as yf
import os
import psycopg2
from dotenv import load_dotenv
from pathlib import Path
from io import StringIO


load_dotenv()

# DB_NAME = os.getenv("POSTGRES_DB")
# DB_USER = os.getenv("POSTGRES_USER")
# DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = "timescaledb"
DB_PORT = 5432

def get_balancesheet(ticker_symbol):
    tick = yf.Ticker(ticker_symbol)

    tick_bs = tick.balancesheet
    df_bs = pd.DataFrame(tick_bs)
    return df_bs


def get_ohlcv(ticker_symbol):
    tick = yf.Ticker(ticker_symbol)

    history = tick.history(period='max', interval='1d')
    df_history = pd.DataFrame(history)
    return df_history


def get_information(ticker_symbol):
    tick = yf.Ticker(ticker_symbol)

    tick_info = tick.info  # ticker_symbol's general info
    # tick_info = json.dumps(tick_info, indent=4)
    return tick_info


def get_financials(ticker_symbol):
    tick = yf.Ticker(ticker_symbol)

    finances = tick.financials
    df_financials = pd.DataFrame(finances)
    return df_financials


def get_cashflow (ticker_symbol):
    tick = yf.Ticker(ticker_symbol)

    tick_cashflow = tick.cashflow  # shows the cashflow data
    df_tick_cashflow = pd.DataFrame(tick_cashflow)
    return df_tick_cashflow


def get_all(ticker_symbol):
    try:
        historical_data = get_ohlcv(ticker_symbol)
        information =  get_information(ticker_symbol)
        financials = get_financials(ticker_symbol)
        cashflow = get_cashflow(ticker_symbol)
        balancesheet = get_balancesheet(ticker_symbol)

        return historical_data, information, financials, cashflow, balancesheet
    except Exception as e:
        print(f"[!] Failed to fetch all {ticker_symbol}:")
        print(f"[!] {e}")


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

    for symbol in top_companies["ticker"]:
        try:
            print(f"Fetching {symbol} data...")
            df_ohlcv = get_ohlcv(symbol)
            df_ohlcv = df_ohlcv.reset_index()

            df_ohlcv.drop_duplicates(subset="Date", keep='last', inplace=True)
            df_ohlcv.insert(0, "ticker", symbol)

            csv_buffer = StringIO()
            df_ohlcv.to_csv(csv_buffer, index=False, header=False)
            csv_buffer.seek(0)

            cursor.copy_from(csv_buffer, "ohlcv", sep=",", null="")

            # For tracking and verification
            df_ohlcv.to_csv((Path(f"/data/ohlcv/{symbol}.csv")), index=False)

        except Exception as error:
            print(f"[!] Failed to download {symbol}:")
            print(f"[!] {error}")
            failed_list.append(symbol)

    conn.commit()
    cursor.close()
    conn.close()

    print(f for f in failed_list)


if __name__ == "__main__":
    main()