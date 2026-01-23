import pandas as pd
from pathlib import Path


def main():
    # project_root = Path(__file__).parent.parent.parent

    # companies_df = pd.read_csv((project_root / "data" / "stock_lists" / "companies.csv"))
    companies_df = pd.read_csv(Path("/data/stock_lists/companies.csv"))

    cleaned_companies_df = companies_df.head(50).drop(columns="price")

    # cleaned_companies_df.to_csv((project_root / "data" / "stock_lists" / "top_companies.csv"), index=False)
    cleaned_companies_df.to_csv(Path("/data/stock_lists/top_companies.csv"), index=False)


if __name__ == "__main__":
    main()