import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# DB_NAME = os.getenv("POSTGRES_DB")
# DB_USER = os.getenv("POSTGRES_USER")
# DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = "timescaledb"
DB_PORT = 5432

def main():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

    cursor = conn.cursor()

    cursor.execute("""
    COPY companies (ticker, company_name, market_cap, country) 
    FROM '/data/stock_lists/top_companies.csv' 
    WITH (FORMAT CSV, HEADER);""")

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()