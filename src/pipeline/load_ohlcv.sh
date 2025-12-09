#!/bin/bash

CONTAINER="test_stock_model_db"
DB="test_db"
USER="postgres"

load_csv() {
    file="$1"
    ticker=$(basename "$file" .csv)
    echo "Loading $ticker ..."

    docker exec -i $CONTAINER psql -U $USER -d $DB <<EOF
COPY ohlcv (ticker, timestamp, open, high, low, close, volume, dividends, stock_splits)
FROM '/data/processed/ohlcv/$ticker.csv'
WITH (FORMAT CSV, HEADER);
EOF
}

export -f load_csv
export CONTAINER DB USER

# parallel -j 2 load_csv ::: ../../../data/processed/ohlcv/*.csv
parallel -j 2 load_csv ::: /data/processed/ohlcv/*.csv