import pandas as pd
from pathlib import Path


def main():
    # project_root = Path(__file__).parent.parent.parent

    # top_companies = pd.read_csv((project_root / "data" / "stock_lists" / "top_companies.csv"))
    top_companies = pd.read_csv(Path("/data/stock_lists/top_companies.csv"))

    failed_list = []
    for symbol in top_companies["ticker"].head(2):
        try:
            # print(f"Removing duplicates from {symbol}...")
            # file_path = project_root / "data" / "raw" / "ohlcv" / f"{symbol}.csv"
            # final_path = project_root / "data" / "processed" / "ohlcv" / f"{symbol}.csv"
            file_path = Path(f"/data/raw/ohlcv/{symbol}.csv")
            final_path = Path(f"/data/processed/ohlcv/{symbol}.csv")

            df = pd.read_csv(file_path)
            df.drop_duplicates(subset="Date", keep='last', inplace=True)
            df.insert(0, "ticker", symbol)
            df.to_csv(final_path, index=False)
        except Exception as error:
            print(f"[!] Failed to remove duplicates from {symbol}")
            print(f"[!] {error}")
            failed_list.append(symbol)

    print("Done")
    print(f for f in failed_list)


if __name__ == "__main__":
    main()