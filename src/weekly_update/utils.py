import pandas as pd
from pathlib import Path


def clean_csv(path):
    df = pd.read_csv(path)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.to_csv(path, index=False)


def clean_all_csv():
    top_companies = pd.read_csv(Path("/data/stock_lists/top_companies.csv"))

    for symbol in top_companies["ticker"]:
        try:
            print(f"Cleaning /data/raw/ohlcv/{symbol}.csv ...")
            clean_csv(Path(f"/data/raw/ohlcv/{symbol}.csv"))

            print(f"Cleaning /data/processed/ohlcv/{symbol}.csv ...")
            clean_csv(Path(f"/data/processed/ohlcv/{symbol}.csv"))

        except Exception as e:
            print(f"[!] Failed to clean CSV for ticker {symbol}:")
            print(f"[!] {e}")


def main():
    clean_all_csv()

if __name__ == "__main__":
    main()