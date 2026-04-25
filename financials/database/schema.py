"""SQLAlchemy ORM models for the quant-lab database.

All time-series tables (ohlcv_data, features) are designed to be converted to
TimescaleDB hypertables by the raw-SQL migration.  SQLAlchemy itself is
unaware of hypertables, so the conversion happens in
``database/migrations/001_initial_schema.sql``.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Shared declarative base for every ORM model."""


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------

class Instrument(Base):
    __tablename__ = "instruments"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    ticker = Column(String(20), nullable=False, unique=True, index=True)
    name = Column(String(255), nullable=True)
    exchange = Column(String(50), nullable=True)
    asset_type = Column(String(50), nullable=False, default="equity")
    sector = Column(String(100), nullable=True)
    currency = Column(String(10), nullable=False, default="USD")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    # relationships
    ohlcv_records = relationship(
        "OHLCVData", back_populates="instrument", cascade="all, delete-orphan"
    )
    feature_records = relationship(
        "Feature", back_populates="instrument", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Instrument(ticker={self.ticker!r}, exchange={self.exchange!r})>"


# ---------------------------------------------------------------------------
# OHLCV data  (TimescaleDB hypertable on `timestamp`)
# ---------------------------------------------------------------------------

class OHLCVData(Base):
    __tablename__ = "ohlcv_data"

    # TimescaleDB hypertables require the partitioning column (timestamp) in
    # the primary key.  We use a composite PK so that a standard SERIAL id is
    # not needed.
    instrument_id = Column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    timestamp = Column(
        DateTime(timezone=True),
        primary_key=True,
        nullable=False,
        index=True,
    )
    interval = Column(
        String(10),
        primary_key=True,
        nullable=False,
        default="1d",
    )
    open = Column(Numeric(18, 6), nullable=False)
    high = Column(Numeric(18, 6), nullable=False)
    low = Column(Numeric(18, 6), nullable=False)
    close = Column(Numeric(18, 6), nullable=False)
    volume = Column(Numeric(20, 2), nullable=False, default=0)
    adjusted_close = Column(Numeric(18, 6), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "timestamp", "interval",
            name="uq_ohlcv_instrument_ts_interval",
        ),
        Index(
            "ix_ohlcv_instrument_interval_ts",
            "instrument_id", "interval", "timestamp",
        ),
    )

    # relationships
    instrument = relationship("Instrument", back_populates="ohlcv_records")

    def __repr__(self) -> str:
        return (
            f"<OHLCVData(instrument_id={self.instrument_id!r}, "
            f"ts={self.timestamp!r}, interval={self.interval!r})>"
        )


# ---------------------------------------------------------------------------
# Features  (TimescaleDB hypertable on `timestamp`)
# ---------------------------------------------------------------------------

class Feature(Base):
    __tablename__ = "features"

    instrument_id = Column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    timestamp = Column(
        DateTime(timezone=True),
        primary_key=True,
        nullable=False,
        index=True,
    )
    feature_name = Column(
        String(100),
        primary_key=True,
        nullable=False,
    )
    feature_value = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "timestamp", "feature_name",
            name="uq_features_instrument_ts_name",
        ),
        Index(
            "ix_features_instrument_name_ts",
            "instrument_id", "feature_name", "timestamp",
        ),
    )

    # relationships
    instrument = relationship("Instrument", back_populates="feature_records")

    def __repr__(self) -> str:
        return (
            f"<Feature(instrument_id={self.instrument_id!r}, "
            f"name={self.feature_name!r}, ts={self.timestamp!r})>"
        )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    model_name = Column(String(255), nullable=False, unique=True, index=True)
    model_type = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    hyperparameters = Column(JSONB, nullable=True, default=dict)
    training_dataset_info = Column(JSONB, nullable=True, default=dict)
    performance_metrics = Column(JSONB, nullable=True, default=dict)
    model_path = Column(String(512), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # relationships
    backtest_results = relationship(
        "BacktestResult", back_populates="model", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ModelRegistry(name={self.model_name!r}, type={self.model_type!r})>"


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

class StrategyRegistry(Base):
    __tablename__ = "strategy_registry"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    strategy_name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    parameters = Column(JSONB, nullable=True, default=dict)
    dependencies = Column(JSONB, nullable=True, default=dict)
    performance_metrics = Column(JSONB, nullable=True, default=dict)
    last_evaluated_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # relationships
    backtest_results = relationship(
        "BacktestResult", back_populates="strategy", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<StrategyRegistry(name={self.strategy_name!r})>"


# ---------------------------------------------------------------------------
# Backtest results
# ---------------------------------------------------------------------------

class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    strategy_id = Column(
        UUID(as_uuid=True),
        ForeignKey("strategy_registry.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_id = Column(
        UUID(as_uuid=True),
        ForeignKey("model_registry.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    instrument_ids = Column(JSONB, nullable=False, default=list)
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    initial_capital = Column(Numeric(18, 2), nullable=False)
    final_capital = Column(Numeric(18, 2), nullable=False)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    cagr = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True, default=0)
    parameters = Column(JSONB, nullable=True, default=dict)
    equity_curve = Column(JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    __table_args__ = (
        Index("ix_backtest_strategy_created", "strategy_id", "created_at"),
    )

    # relationships
    strategy = relationship("StrategyRegistry", back_populates="backtest_results")
    model = relationship("ModelRegistry", back_populates="backtest_results")

    def __repr__(self) -> str:
        return (
            f"<BacktestResult(strategy_id={self.strategy_id!r}, "
            f"sharpe={self.sharpe_ratio!r})>"
        )
