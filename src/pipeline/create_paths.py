from pathlib import Path


def main():
    Path("/data/stock_lists").mkdir(parents=True, exist_ok=True)

    Path("/data/raw/ohlcv").mkdir(parents=True, exist_ok=True)
    Path("/data/raw/stock_balancesheet").mkdir(parents=True, exist_ok=True)
    Path("/data/raw/stock_cashflow").mkdir(parents=True, exist_ok=True)
    Path("/data/raw/stock_financials").mkdir(parents=True, exist_ok=True)

    Path("/data/processed/ohlcv").mkdir(parents=True, exist_ok=True)
    # Path("/data/processed/stock_balancesheet").mkdir(parents=True, exist_ok=True)
    # Path("/data/processed/stock_cashflow").mkdir(parents=True, exist_ok=True)
    # Path("/data/processed/stock_financials").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()