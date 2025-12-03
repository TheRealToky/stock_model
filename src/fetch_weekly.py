import pandas as pd
import datetime as dt
import yfinance as yf
import time

"""
Fetch weekly new stock numbers
"""

# Fetch weekly function
def fetch_weekly(ticker_symbol):
    today = dt.datetime.today() # runs every saturday

    try:
        tick = yf.Ticker(ticker_symbol)
        raw_historical_data = tick.history(interval="1d", period="5d", end=today)
        weekly_df = pd.DataFrame(raw_historical_data)

        weekly_df.to_csv(f"../data/historical_data/{ticker_symbol}.csv", mode="a", header=False)
        updated_df = pd.read_csv(f"../data/historical_data/{ticker_symbol}.csv")
        updated_df.drop_duplicates()
        return True  # Success
    except Exception as e:
        print("[!] Failed to fetch weekly data for ticker " + ticker_symbol)
        print("[!] ", e)
        return False  # Failure


def main():
    stock_by_market_cap = pd.read_csv("../data/stock_lists/stock_by_market_cap.csv")

    failed_list = []
    for symbol in stock_by_market_cap["Symbol"].head(50):
        print(f"Updating {symbol}...")

        success = fetch_weekly(symbol)

        # Add to failed list if fetch was unsuccessful
        if not success:
            failed_list.append(symbol)

        time.sleep(5)

    print("Failed to fetch:")
    print(f for f in failed_list)

if __name__ == "__main__":
    main()