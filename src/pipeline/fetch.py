import pandas as pd
import yfinance as yf
import os
from pathlib import Path


def get_balancesheet(ticker_symbol):
    tick = yf.Ticker(ticker_symbol)

    tick_bs = tick.balancesheet
    df_bs = pd.DataFrame(tick_bs)
    return df_bs


def get_historical_data(ticker_symbol):
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
        historical_data = get_historical_data(ticker_symbol)
        information =  get_information(ticker_symbol)
        financials = get_financials(ticker_symbol)
        cashflow = get_cashflow(ticker_symbol)
        balancesheet = get_balancesheet(ticker_symbol)

        return historical_data, information, financials, cashflow, balancesheet
    except Exception as e:
        print(f"[!] Failed to fetch all {ticker_symbol}:")
        print(f"[!] {e}")


def main():
    # project_root = Path(__file__).parent.parent.parent

    # sp_500_list_df = pd.read_csv((project_root / "data" / "stock_lists" / "sp_500_list.csv"))
    # sp_500_change_df = pd.read_csv((project_root / "data" / "stock_lists" / "sp_500_change_list.csv"))
    # top_companies = pd.read_csv((project_root / "data" / "stock_lists" / "top_companies.csv"))
    top_companies = pd.read_csv(Path("/data/stock_lists/top_companies.csv"))

    failed_list = []

    for symbol in top_companies["ticker"].head(2):
        # file_path = project_root / "data" / "raw" / "ohlcv" / f"{symbol}.csv"
        file_path = Path(f"/data/raw/ohlcv/{symbol}.csv")

        if os.path.isfile(file_path):
            print(f"{symbol} already exists.")
            continue
        else:
            try:
                print(f"Downloading {symbol} data...")
                symbol_ohlcv, symbol_information, symbol_financials, symbol_cashflow, symbol_balancesheet = get_all(symbol)

                # symbol_ohlcv.to_csv((project_root / "data" / "raw" / "ohlcv" / f"{symbol}.csv"))
                # symbol_financials.to_csv((project_root / "data" / "raw" / "stock_financials" / f"{symbol}.csv"))
                # symbol_cashflow.to_csv((project_root / "data" / "raw" / "stock_cashflow" / f"{symbol}.csv"))
                # symbol_balancesheet.to_csv((project_root / "data" / "raw" / "stock_balancesheet" / f"{symbol}.csv"))

                symbol_ohlcv.to_csv(Path(f"/data/raw/ohlcv/{symbol}.csv"))
                symbol_financials.to_csv(Path(f"/data/raw/stock_financials/{symbol}.csv"))
                symbol_cashflow.to_csv(Path(f"/data/raw/stock_cashflow/{symbol}.csv"))
                symbol_balancesheet.to_csv(Path(f"/data/raw/stock_balancesheet/{symbol}.csv"))

                # Turns out, dictionaries can't be converted to csv XD
                # symbol_information.to_csv((project_root / "data" / "raw" / "stock_info" / f"{symbol}.csv"))
                # symbol_information.to_csv(Path(f"/data/raw/stock_info/{symbol}.csv"))

            except Exception as error:
                print(f"[!] Failed to download {symbol}:")
                print(f"[!] {error}")
                failed_list.append(symbol)

    print(f for f in failed_list)


if __name__ == "__main__":
    main()