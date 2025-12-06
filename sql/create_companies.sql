CREATE TABLE IF NOT EXISTS companies (
    id SERIAL,
    ticker TEXT PRIMARY KEY,
    company_name TEXT,
    market_cap NUMERIC,
    country TEXT
);