#!/bin/bash

source .env

CONTAINER="stock_model_db"
DB="postgres"
USER="postgres"

for file in ./data/processed/ohlcv/*.csv; do
    ticker=$(basename "$file" .csv)

    echo "Loading $ticker ..."

    docker exec -i $CONTAINER psql -U $USER -d $DB <<EOF
    COPY ohlcv (ticker, timestamp, open, high, low, close, volume, dividends, stock_splits)
    FROM '/data/processed/ohlcv/$ticker.csv'
    CSV HEADER;
EOF
done

echo "Done."