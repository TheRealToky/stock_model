#!/bin/bash

CONTAINER="stock_model_db"
DB="postgres"
USER="postgres"

docker exec -i $CONTAINER psql -U $USER -d $DB <<EOF
COPY companies (ticker, company_name, market_cap, country)
FROM '/data/stock_lists/top_companies.csv'
WITH (FORMAT CSV, HEADER);
EOF

echo "Done."