"""Central configuration for the alternative-data (flight) pipeline.

Extends the quant-lab ``config.settings`` pattern with dataclass-based
settings that read from environment variables.  The alt-data DB is
separate from the financial DB so the two can live in different
Postgres containers (or the same one via distinct schemas).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AltDatabaseConfig:
    """Connection parameters for the alternative-data PostgreSQL instance."""

    host: str = os.getenv("ALT_DB_HOST", "alt-postgres")
    port: int = int(os.getenv("ALT_DB_PORT", "5432"))
    database: str = os.getenv("ALT_DB_NAME", "alt_data")
    user: str = os.getenv("ALT_DB_USER", "alt_user")
    password: str = os.getenv("ALT_DB_PASSWORD", "alt_secret")
    schema: str = os.getenv("ALT_DB_SCHEMA", "public")

    @property
    def url(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @property
    def async_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass
class FinancialDatabaseConfig:
    """Connection parameters for the existing financial (OHLCV) DB.

    Reads the same env vars used by the root ``config.settings`` so the
    alt pipeline can connect to the already-running TimescaleDB.
    """

    host: str = os.getenv("TIMESCALE_HOST", "timescaledb")
    port: int = int(os.getenv("TIMESCALE_PORT", "5432"))
    database: str = os.getenv("TIMESCALE_DB", "quantlab")
    user: str = os.getenv("TIMESCALE_USER", "quantlab")
    password: str = os.getenv("TIMESCALE_PASSWORD", "quantlab_secret")

    @property
    def url(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass
class OpenSkyConfig:
    """OpenSky Network API configuration.

    Authentication uses the OAuth2 client-credentials flow against
    OpenSky's Keycloak realm.  Issue a client at
    https://opensky-network.org/my-opensky/account and set
    ``OPENSKY_CLIENT_ID`` / ``OPENSKY_CLIENT_SECRET``.
    """

    base_url: str = os.getenv("OPENSKY_BASE_URL", "https://opensky-network.org/api")
    token_url: str = os.getenv(
        "OPENSKY_TOKEN_URL",
        "https://auth.opensky-network.org/auth/realms/opensky-network"
        "/protocol/openid-connect/token",
    )
    client_id: str | None = os.getenv("OPENSKY_CLIENT_ID")
    client_secret: str | None = os.getenv("OPENSKY_CLIENT_SECRET")
    # Refresh the cached token this many seconds before its advertised
    # expiry to avoid edge-case 401s from clock skew.
    token_refresh_margin_seconds: int = int(
        os.getenv("OPENSKY_TOKEN_REFRESH_MARGIN", "30")
    )
    # Anonymous ~100 req/day; authenticated ~4000 req/day. Be conservative
    # per-minute so we never spike and trigger a global block.
    max_requests_per_minute: int = int(os.getenv("OPENSKY_RPM", "30"))
    request_timeout_seconds: int = int(os.getenv("OPENSKY_TIMEOUT", "30"))


@dataclass
class AltPipelineConfig:
    """Generic pipeline knobs (batch size, retries, default dates)."""

    default_start_date: str = "2021-01-01"
    batch_size: int = 500
    max_retries: int = 5
    retry_delay_seconds: int = 5
    default_airports: list[str] = field(
        default_factory=lambda: ["KJFK", "KLAX", "KORD", "EGLL", "EDDF"]
    )
    # Drop records where the closest stand/gate distance is greater than
    # this threshold (meters). Records without a distance field are kept.
    max_stand_distance_m: int = 10_000
    feature_config_path: str = str(
        _PROJECT_ROOT / "alt_data" / "config" / "features.yaml"
    )
    mapping_config_path: str = str(
        _PROJECT_ROOT / "alt_data" / "config" / "mappings.yaml"
    )


@dataclass
class AltSettings:
    db: AltDatabaseConfig = field(default_factory=AltDatabaseConfig)
    financial_db: FinancialDatabaseConfig = field(default_factory=FinancialDatabaseConfig)
    opensky: OpenSkyConfig = field(default_factory=OpenSkyConfig)
    pipeline: AltPipelineConfig = field(default_factory=AltPipelineConfig)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    project_root: Path = _PROJECT_ROOT

    def feature_toggles(self) -> dict[str, bool]:
        """Load the feature on/off toggles from YAML (empty dict if missing)."""
        path = Path(self.pipeline.feature_config_path)
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return dict(data.get("features", {}))

    def airport_to_tickers(self) -> dict[str, list[str]]:
        """Load the airport-ICAO -> [tickers] mapping from YAML."""
        path = Path(self.pipeline.mapping_config_path)
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return dict(data.get("airport_to_tickers", {}))


alt_settings = AltSettings()
