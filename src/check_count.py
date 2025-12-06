from symtable import Symbol

import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
from psycopg2._psycopg import cursor
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def count_lines_enumerate(filepath):
    with open(filepath, 'r') as f:
        for count, line in enumerate(f):
            pass
    return count + 1


def main():
    load_dotenv()

    DB_NAME = os.getenv("POSTGRES_DB")
    DB_USER = os.getenv("POSTGRES_USER")
    DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    DB_HOST = "localhost"
    DB_PORT = 5432

    top_companies = pd.read_csv("../data/stock_lists/top_companies.csv")
    symbols = top_companies["ticker"].tolist()

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    for symbol in symbols:
        cursor.execute(f"SELECT count(*) FROM ohlcv WHERE ticker='{symbol}'")
        output = cursor.fetchall()

        sql_row_count = output[0][0]
        line_count = count_lines_enumerate(f"../data/processed/ohlcv/{symbol}.csv")

        if (line_count - 1) != sql_row_count:
            print(f"Symbol {symbol} row does not match")

if __name__ == "__main__":
    main()