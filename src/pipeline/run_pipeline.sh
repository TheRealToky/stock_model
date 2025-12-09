#!/usr/bin/env bash

set -e

# Creates the company and ohlcv tables
python setup_db.py

# Fetches and prepares the ohlcv data
python clean_list.py

# Load all CSVs to timescale
python load_company.py
python fetch.py

# Runs every Saturday at 4:00 AM
0 4 * * 5 /usr/local/bin/python /app/fetch_weekly.py >> /var/log/stock_model.log 2>&1
