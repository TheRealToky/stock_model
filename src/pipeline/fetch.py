import pandas as pd
import yfinance as yf
from pathlib import Path
from io import StringIO
from db.session import engine


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


def download_raw_ticker(symbol):
    raw_path = Path("/data/raw/ohlcv")

    try:
        df_ohlcv = get_ohlcv(symbol)
        df_ohlcv.to_csv((raw_path / f"{symbol}.csv"))
        print(f"Fetching raw {symbol} data...")
        return df_ohlcv
    except Exception as error:
        print(f"[!] Failed to download raw {symbol}:")
        print(f"[!] {error}")


def process_ticker(symbol, df_ohlcv):
    processed_path = Path("/data/processed/ohlcv")

    try:
        print(f"Processing {symbol} data...")
        df_ohlcv = df_ohlcv.reset_index()
        df_ohlcv.drop_duplicates(subset="Date", keep='last', inplace=True)
        df_ohlcv.insert(0, "ticker", symbol)

        df_ohlcv.to_csv((processed_path / f"{symbol}.csv"), index=False)

        return df_ohlcv
    except Exception as error:
        print(f"[!] Failed to process {symbol}:")
        print(f"[!] {error}")


def insert_db_ticker(symbol, df_ohlcv, engine):
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

    Path("/data/processed/ohlcv").mkdir(parents=True, exist_ok=True)
    Path("/data/raw/ohlcv").mkdir(parents=True, exist_ok=True)

    for symbol in top_companies["ticker"]:
        raw_df_ohlcv = download_raw_ticker(symbol)
        df_ohlcv = process_ticker(symbol, raw_df_ohlcv)
        insert_db_ticker(symbol, df_ohlcv, engine)


if __name__ == "__main__":
    main()