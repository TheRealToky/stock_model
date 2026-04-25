# Alt Data Pipeline (Flight Data)

Production-grade flight-tracking data pipeline for the quant-lab
research platform.  Ingests OpenSky Network data, cleans it, persists
it into a dedicated PostgreSQL instance, and exposes a **DataHub**
query layer that joins flight data with the existing OHLCV database
for cross-domain systematic research.

```
alt_data/
├── run_pipeline.py            # CLI entry point
├── requirements.txt
├── cleaning/                  # FlightDataCleaner
├── config/                    # Settings + YAML (features.yaml, mappings.yaml)
├── features/                  # Registry, engine, 10 technical features
├── ingestion/                 # FlightData dataclass, OpenSkyClient, DataFetcher
├── query/                     # DataHub, AssetMapper, time alignment
├── database/                  # SQLAlchemy schema, connection, repository, loader
│   └── migrations/            # SQL schema migrations
└── utils/                     # Logging, time helpers
```

The Dockerfile lives at the repo root under `docker/Dockerfile.alt_data`,
and the alt-data services are part of the root `docker-compose.yml`.
Tests live under `tests/unit/alt_data/`.

## Quick start

```bash
# 1. Start the full stack (financial DB + alt DB + pipeline containers)
docker compose up -d

# 2. Apply the alt-data schema
docker compose exec alt-pipeline \
    python -m alt_data.database.migrate

# 3. Run the pipeline for one airport
docker compose exec alt-pipeline python -m alt_data.run_pipeline \
    --airport KJFK --start 2024-01-01 --end 2024-01-10

# 4. Status
docker compose exec alt-pipeline python -m alt_data.run_pipeline --status
```

## Cross-dataset query example

```python
from alt_data.query.datahub import DataHub

hub = DataHub()

# Flights at KJFK, joined with DAL OHLCV, on trading days only
df = hub.get_joined_data(
    airport="KJFK",
    ticker="DAL",
    start="2024-01-01",
    end="2024-02-01",
    features=["net_flow", "hub_connectivity"],
    shift_days=1,          # flights at t -> returns at t+1
)

df["ret"] = df["close"].pct_change()
df[["net_flow", "hub_connectivity", "ret"]].head()
```

## Environment

The pipeline reads env vars from `.env` at the repo root.  The alt-data
block is:

```ini
ALT_DB_HOST=alt-postgres
ALT_DB_PORT=5432
ALT_DB_NAME=alt_data
ALT_DB_USER=alt_user
ALT_DB_PASSWORD=alt_secret

OPENSKY_USERNAME=
OPENSKY_PASSWORD=
OPENSKY_RPM=30
```

The financial-DB block (`TIMESCALE_*`) is shared with the root pipeline.

## Tests

```bash
pytest tests/unit/alt_data -v
```

Integration tests (which talk to real Postgres and the OpenSky API)
live under `tests/integration/` and are gated behind
`ALT_INTEGRATION_TESTS=1`.
