"""Central configuration for the quant lab."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatabaseConfig:
    host: str = os.getenv("TIMESCALE_HOST", "localhost")
    port: int = int(os.getenv("TIMESCALE_PORT", "5432"))
    database: str = os.getenv("TIMESCALE_DB", "quantlab")
    user: str = os.getenv("TIMESCALE_USER", "quantlab")
    password: str = os.getenv("TIMESCALE_PASSWORD", "quantlab_secret")

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class DataPipelineConfig:
    default_start_date: str = "2010-01-01"
    default_interval: str = "1d"
    default_data_source: str = "yahoo"
    batch_size: int = 50
    max_retries: int = 7
    retry_delay_seconds: int = 5
    default_tickers: list[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "TSLA", "NVDA", "JPM", "V", "JNJ",
        "SPY", "QQQ", "IWM", "DIA", "VTI",
    ])


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    risk_free_rate: float = 0.04  # 4% annual


@dataclass
class ModelConfig:
    models_dir: str = str(Path(__file__).parent.parent / "models" / "saved")
    default_test_size: float = 0.2
    random_state: int = 42


@dataclass
class Settings:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    pipeline: DataPipelineConfig = field(default_factory=DataPipelineConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    project_root: Path = Path(__file__).parent.parent


settings = Settings()
