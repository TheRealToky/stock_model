# 📈 Stock Market Data Pipeline

A robust, containerized data pipeline system for fetching, storing, cleaning, and analyzing stock market data (OHLCV - Open, High, Low, Close, Volume) using TimescaleDB, SQLAlchemy ORM, and yfinance API.

## 🎯 Overview

This project provides an automated solution for collecting, cleaning, and storing historical and real-time stock market data. It uses TimescaleDB (PostgreSQL extension optimized for time-series data) to efficiently store OHLCV data for top companies, with automated weekly updates to keep the data current. The project also includes a data cleaning pipeline, feature engineering modules, and Jupyter notebooks for exploratory analysis.

## ✨ Features

- **Automated Data Collection**: Fetches historical OHLCV data for top companies using yfinance
- **Time-Series Optimized Storage**: Uses TimescaleDB with hypertables for efficient time-series data management
- **SQLAlchemy ORM**: Structured database models with relationships for companies, raw OHLCV, and cleaned OHLCV data
- **Database Migrations**: Alembic-powered schema migrations for safe, versioned database changes
- **Containerized Architecture**: Fully dockerized with Docker Compose for easy deployment
- **Weekly Auto-Updates**: Scheduled cron jobs to fetch weekly data updates every Saturday
- **Data Cleaning Pipeline**: Automated deduplication and cleaning of CSV and database records
- **Feature Engineering**: Module for building derived features from cleaned data
- **Exploratory Notebooks**: Jupyter notebooks for data exploration, cleaning, scraping, and migration
- **Data Redundancy**: Stores both raw and processed CSV files locally
- **Scalable Design**: Hash partitioning across 2000 partitions for optimal query performance

## 🏗️ Architecture

```
stock_model/
├── db/                         # SQLAlchemy ORM layer
│   ├── base.py                 # Declarative base class
│   ├── models.py               # ORM models (Company, OHLCV, CleanOHLCV)
│   └── session.py              # Database engine & session factory
├── src/
│   ├── pipeline/               # Initial data pipeline
│   │   ├── fetch.py            # Fetch historical OHLCV data
│   │   ├── setup_db.py         # Initialize database schema
│   │   ├── load_company.py     # Load company metadata
│   │   ├── clean_list.py       # Clean & filter company list
│   │   ├── run_pipeline.sh     # Pipeline orchestration script
│   │   └── Dockerfile          # Pipeline container config
│   ├── weekly_update/          # Weekly data updates
│   │   ├── fetch_weekly.py     # Fetch recent week data
│   │   ├── utils.py            # CSV cleaning utilities
│   │   ├── pipeline_cron       # Cron schedule configuration
│   │   └── Dockerfile          # Weekly update container
│   ├── data_cleaning/          # Data cleaning module
│   │   └── insert_clean.py     # Insert cleaned data into DB
│   └── features_engineering/   # Feature engineering module
├── alembic/                    # Database migrations
│   ├── env.py                  # Migration environment config
│   ├── versions/               # Migration scripts
│   └── script.py.mako          # Migration template
├── notebooks/                  # Jupyter notebooks
│   ├── clean_data.ipynb        # Data cleaning exploration
│   ├── features_engineering.ipynb # Feature engineering exploration
│   ├── fetch_data.ipynb        # Data fetching experiments
│   ├── migrate_to_pg.ipynb     # PostgreSQL migration notebook
│   └── scrape_wikipedia.ipynb  # Wikipedia scraping for company data
├── data/                       # Data storage directory
│   ├── raw/                    # Raw OHLCV CSVs
│   ├── processed/              # Cleaned OHLCV CSVs
│   ├── adjusted/               # Adjusted data
│   └── stock_lists/            # Company ticker lists
├── alembic.ini                 # Alembic configuration
├── docker-compose.yml          # Multi-container orchestration
├── main.py                     # Project entry point
└── requirements.txt            # Python dependencies
```

## 🔧 Tech Stack

- **Database**: TimescaleDB 2.24.0 (PostgreSQL 18)
- **ORM**: SQLAlchemy 2.0
- **Migrations**: Alembic
- **Languages**: Python 3.x
- **Key Libraries**:
  - `yfinance` - Stock market data API
  - `pandas` / `numpy` - Data manipulation & analysis
  - `sqlalchemy` - ORM and database toolkit
  - `psycopg2` - PostgreSQL database adapter
  - `beautifulsoup4` / `lxml` - Web scraping & XML parsing
  - `python-dotenv` - Environment variable management

## 📋 Prerequisites

- Docker & Docker Compose
- Python 3.x (for local development)
- At least 2GB RAM for TimescaleDB container

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/TheRealToky/stock_model.git
cd stock_model
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=stock_data
DATABASE_URL=postgresql+psycopg2://your_db_user:your_secure_password@timescaledb:5432/stock_data
```

### 3. Prepare Stock List

Place your company ticker list in `data/stock_lists/top_companies.csv`:

```csv
ticker,company_name,market_cap,country
AAPL,Apple Inc.,2800000000000,USA
MSFT,Microsoft Corporation,2500000000000,USA
GOOGL,Alphabet Inc.,1600000000000,USA
```

### 4. Start the Services

```bash
docker-compose up -d
```

This will:
- Start TimescaleDB container
- Run the initial data pipeline to fetch historical data
- Set up the weekly update service with cron scheduling

### 5. Verify Data Collection

Check the logs:

```bash
docker logs ohlcv_pipeline
docker logs ohlcv_weekly
```

## 📊 Database Schema

### Companies Table

| Column       | Type    | Description                    |
|--------------|---------|--------------------------------|
| id           | SERIAL  | Auto-incrementing ID           |
| ticker       | TEXT    | Stock ticker symbol (UNIQUE)   |
| company_name | TEXT    | Company name                   |
| market_cap   | NUMERIC | Market capitalization          |
| country      | TEXT    | Company's country              |

### OHLCV Hypertable

| Column       | Type             | Description                    |
|--------------|------------------|--------------------------------|
| ticker       | TEXT             | Stock ticker (FOREIGN KEY)     |
| timestamp    | TIMESTAMPTZ      | Data timestamp                 |
| open         | DOUBLE PRECISION | Opening price                  |
| high         | DOUBLE PRECISION | Highest price                  |
| low          | DOUBLE PRECISION | Lowest price                   |
| close        | DOUBLE PRECISION | Closing price                  |
| volume       | BIGINT           | Trading volume                 |
| dividends    | REAL             | Dividend amount                |
| stock_splits | REAL             | Stock split ratio              |

**Primary Key**: (ticker, timestamp)  
**Partitioning**: Hash partitioning on `ticker` (2000 partitions)  
**Chunk Interval**: 7 days

### Clean OHLCV Table

Mirrors the OHLCV schema and stores cleaned/deduplicated data.  
**Primary Key**: (ticker, timestamp)

## 🔄 Data Pipeline Process

### Initial Pipeline

1. **Database Setup**: Creates companies and OHLCV tables with hypertable configuration
2. **Company Loading**: Loads company metadata from CSV into the companies table
3. **Historical Data Fetch**: Downloads maximum available historical data for each ticker
4. **Data Processing**: Cleans data, removes duplicates, standardizes format
5. **Storage**: Saves to both local CSV files and TimescaleDB

### Weekly Updates

- **Schedule**: Runs every Saturday via cron
- **Process**: Fetches last 6 days of data for all tickers
- **CSV Cleaning**: Deduplicates raw and processed CSV files after each update
- **Incremental**: Appends new data to existing CSV files and database
- **Error Handling**: Logs failed tickers for review

### Data Cleaning

- Reads raw data from the OHLCV table
- Applies cleaning transformations (deduplication, formatting)
- Writes cleaned data into the `clean_ohlcv` table

## 🧪 Notebooks

The `notebooks/` directory contains Jupyter notebooks for exploration and prototyping:

| Notebook                     | Description                                      |
|------------------------------|--------------------------------------------------|
| `fetch_data.ipynb`           | Experiment with yfinance data fetching            |
| `clean_data.ipynb`           | Explore and prototype data cleaning steps         |
| `features_engineering.ipynb` | Develop and test feature engineering pipelines     |
| `migrate_to_pg.ipynb`        | Test data migration workflows to PostgreSQL       |
| `scrape_wikipedia.ipynb`     | Scrape Wikipedia for company metadata             |

## 🗃️ Database Migrations

This project uses **Alembic** for database schema migrations:

```bash
# Generate a new migration after modifying models
alembic revision --autogenerate -m "description of change"

# Apply all pending migrations
alembic upgrade head

# Downgrade one revision
alembic downgrade -1
```

## 📁 Data Storage

Data is stored in multiple locations:

1. **TimescaleDB**: Primary database storage with optimized querying
2. **Clean OHLCV Table**: Cleaned data stored separately for analysis
3. **Raw CSV**: `data/raw/ohlcv/{ticker}.csv` - Complete historical data
4. **Processed CSV**: `data/processed/ohlcv/{ticker}.csv` - Cleaned and formatted data
5. **Adjusted**: `data/adjusted/` - Adjusted price data

## 🛠️ Development

### Local Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Individual Components

```bash
# Setup database
cd src/pipeline
python setup_db.py

# Load company data
python load_company.py

# Fetch initial data
python fetch.py

# Fetch weekly updates
cd ../weekly_update
python fetch_weekly.py

# Clean CSV files
python utils.py

# Insert cleaned data
cd ../data_cleaning
python insert_clean.py
```

## 🐳 Docker Services

### TimescaleDB Service

- **Image**: `timescale/timescaledb:2.24.0-pg18`
- **Port**: 5432
- **Shared Memory**: 2GB
- **Health Check**: Automated readiness verification
- **Volumes**: Persistent `stock_data` volume + `data/` directory mount

### Pipeline Service

- **Dependencies**: Waits for TimescaleDB to be healthy
- **Function**: Initial data fetch and database setup
- **Volumes**: Mounts `src/pipeline`, `data`, and `db` directories

### Weekly Update Service

- **Dependencies**: Waits for TimescaleDB to be healthy
- **Function**: Scheduled weekly data updates
- **Schedule**: Configured via cron (Saturdays)
- **Volumes**: Mounts `src/weekly_update`, `data`, and `db` directories

## 🔍 Querying Data

Connect to TimescaleDB:

```bash
docker exec -it stock_model-db psql -U your_db_user -d stock_data
```

Example queries:

```sql
-- Get latest prices for a ticker
SELECT * FROM ohlcv 
WHERE ticker = 'AAPL' 
ORDER BY timestamp DESC 
LIMIT 10;

-- Get average volume by ticker
SELECT ticker, AVG(volume) as avg_volume 
FROM ohlcv 
GROUP BY ticker;

-- Query cleaned data
SELECT * FROM clean_ohlcv
WHERE ticker = 'MSFT'
ORDER BY timestamp DESC
LIMIT 10;

-- Time-based aggregation (monthly averages)
SELECT 
    time_bucket('1 month', timestamp) AS month,
    ticker,
    AVG(close) as avg_close,
    MAX(high) as max_high,
    MIN(low) as min_low
FROM ohlcv
GROUP BY month, ticker
ORDER BY month DESC, ticker;
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance API wrapper
- [TimescaleDB](https://www.timescale.com/) - Time-series database
- [SQLAlchemy](https://www.sqlalchemy.org/) - Python SQL toolkit and ORM
- [Alembic](https://alembic.sqlalchemy.org/) - Database migration tool
- [pandas](https://pandas.pydata.org/) - Data analysis library

## 📧 Contact

**TheRealToky** - [@TheRealToky](https://github.com/TheRealToky)

Project Link: [https://github.com/TheRealToky/stock_model](https://github.com/TheRealToky/stock_model)

---

⭐ If you find this project useful, please consider giving it a star!