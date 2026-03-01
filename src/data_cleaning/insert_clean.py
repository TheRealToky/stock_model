import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))

def main():
    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM ohlcv WHERE ticker = 'SPY'", con=conn)

    # Add cleaning here, no need for now

    df.to_sql(
        "ohlcv_clean",
        con=engine,
        if_exists="replace",  # Options: "fail", "replace", "append"
        index=False  # Set to True if you want to write the DataFrame index as a column
    )
    print("Data successfully loaded into table 'ohlcv_clean'.")


if __name__ == "__main__":
    main()