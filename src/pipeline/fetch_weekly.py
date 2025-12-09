import pandas as pd
import yfinance as yf
import datetime as dt
import time
from pathlib import Path

"""
Fetch weekly new stock numbers
"""

# project_root = Path(__file__).parent.parent.parent

# Fetch weekly function
def fetch_weekly(ticker_symbol):
    today = dt.datetime.today() # runs every saturday

    try:
        tick = yf.Ticker(ticker_symbol)
        # path = project_root / "data" / "raw" / "ohlcv" / f"{ticker_symbol}.csv"
        path = Path(f"/data/raw/ohlcv/{ticker_symbol}.csv")

        raw_historical_data = tick.history(interval="1d", period="5d", end=today)
        weekly_df = pd.DataFrame(raw_historical_data)
        weekly_df.to_csv(path, mode="a", header=False)

        updated_df = pd.read_csv(path)
        updated_df.drop_duplicates(subset="Date", keep='last',inplace=True)
        updated_df.to_csv(path)
        return True  # Success
    except Exception as e:
        print("[!] Failed to fetch weekly data for ticker " + ticker_symbol)
        print("[!] ", e)
        return False  # Failure


def main():
    # top_companies = pd.read_csv((project_root / "data" / "stock_lists" / "top_companies.csv"))
    top_companies = pd.read_csv(Path("/data/stock_lists/top_companies.csv"))

    failed_list = []
    delay = 1

    for symbol in top_companies["ticker"]:
        try:
            print(f"Updating {symbol}...")

            success = fetch_weekly(symbol)

            # Add to failed list if fetch was unsuccessful
            if not success:
                failed_list.append(symbol)

            time.sleep(delay)
        except Exception as e:
            failed_list.append(symbol)
            time.sleep(delay)
            delay *= 2

    print("Failed to fetch:")
    print(f for f in failed_list)


if __name__ == "__main__":
    main()