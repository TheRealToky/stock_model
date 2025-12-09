#!/usr/bin/env bash

set -e

# Creates the company and ohlcv tables
python create_paths.py
python setup_db.py

# Fetches and prepares the ohlcv data
python clean_list.py
python fetch.py
python clean_tables.py

# Set this script with a cron job that runs every Saturday at 4:00 AM
# fetch_weekly.py
# cron job here or IDK

# Load all CSVs to timescale
./load_company_list.sh
./load_ohlcv.sh

python check_count.py