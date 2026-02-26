from sqlalchemy import text
from db.session import SessionLocal


def main():
    db = SessionLocal()

    db.execute(text("""
        COPY companies (ticker, company_name, market_cap, country) 
        FROM '/data/stock_lists/top_companies.csv' 
        WITH (FORMAT CSV, HEADER);
        """))
    db.commit()
    db.close()


if __name__ == "__main__":
    main()