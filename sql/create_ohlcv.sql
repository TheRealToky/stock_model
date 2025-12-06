CREATE TABLE IF NOT EXISTS ohlcv (
    ticker TEXT NOT NULL REFERENCES companies(ticker),
    timestamp TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    dividends REAL,
    stock_splits REAL,
    PRIMARY KEY (ticker, timestamp)
);

SELECT create_hypertable(
    'ohlcv',
    'timestamp',
    chunk_time_interval => interval '7 days',
    partitioning_column => 'ticker',
    number_partitions => 2000,
    partitioning_method => 'hash'
);