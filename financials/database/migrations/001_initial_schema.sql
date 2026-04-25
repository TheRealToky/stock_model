-- 001_initial_schema.sql
-- Idempotent migration: creates all quant-lab tables and TimescaleDB hypertables.
-- Safe to run multiple times (uses IF NOT EXISTS / DO blocks).

-- ============================================================================
-- 0. Extensions
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";      -- uuid_generate_v4()
CREATE EXTENSION IF NOT EXISTS "timescaledb";     -- hypertable support

-- ============================================================================
-- 1. Migration tracking
-- ============================================================================
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     VARCHAR(50)  PRIMARY KEY,
    description TEXT,
    applied_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- 2. instruments
-- ============================================================================
CREATE TABLE IF NOT EXISTS instruments (
    id          UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker      VARCHAR(20)     NOT NULL,
    name        VARCHAR(255),
    exchange    VARCHAR(50),
    asset_type  VARCHAR(50)     NOT NULL DEFAULT 'equity',
    sector      VARCHAR(100),
    currency    VARCHAR(10)     NOT NULL DEFAULT 'USD',
    is_active   BOOLEAN         NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE UNIQUE INDEX IF NOT EXISTS ix_instruments_ticker
    ON instruments (ticker);

CREATE INDEX IF NOT EXISTS ix_instruments_asset_type
    ON instruments (asset_type);

CREATE INDEX IF NOT EXISTS ix_instruments_is_active
    ON instruments (is_active);

-- ============================================================================
-- 3. ohlcv_data  (will be converted to a hypertable)
-- ============================================================================
CREATE TABLE IF NOT EXISTS ohlcv_data (
    instrument_id   UUID            NOT NULL
        REFERENCES instruments (id) ON DELETE CASCADE,
    "timestamp"     TIMESTAMPTZ     NOT NULL,
    "interval"      VARCHAR(10)     NOT NULL DEFAULT '1d',
    "open"          NUMERIC(18,6)   NOT NULL,
    high            NUMERIC(18,6)   NOT NULL,
    low             NUMERIC(18,6)   NOT NULL,
    "close"         NUMERIC(18,6)   NOT NULL,
    volume          NUMERIC(20,2)   NOT NULL DEFAULT 0,
    adjusted_close  NUMERIC(18,6),

    PRIMARY KEY (instrument_id, "timestamp", "interval")
);

-- Unique constraint (same columns as PK, but named explicitly for clarity)
-- The PK already enforces uniqueness, so this is a no-op semantically but
-- gives us a readable constraint name.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_ohlcv_instrument_ts_interval'
    ) THEN
        ALTER TABLE ohlcv_data
            ADD CONSTRAINT uq_ohlcv_instrument_ts_interval
            UNIQUE (instrument_id, "timestamp", "interval");
    END IF;
END $$;

-- Composite index for fast lookups by instrument + interval, ordered by time
CREATE INDEX IF NOT EXISTS ix_ohlcv_instrument_interval_ts
    ON ohlcv_data (instrument_id, "interval", "timestamp" DESC);

-- Convert to hypertable (idempotent via the DO block)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables
        WHERE hypertable_name = 'ohlcv_data'
    ) THEN
        PERFORM create_hypertable(
            'ohlcv_data',
            'timestamp',
            migrate_data       => TRUE,
            if_not_exists      => TRUE
        );
    END IF;
END $$;

-- ============================================================================
-- 4. features  (will be converted to a hypertable)
-- ============================================================================
CREATE TABLE IF NOT EXISTS features (
    instrument_id   UUID            NOT NULL
        REFERENCES instruments (id) ON DELETE CASCADE,
    "timestamp"     TIMESTAMPTZ     NOT NULL,
    feature_name    VARCHAR(100)    NOT NULL,
    feature_value   DOUBLE PRECISION,

    PRIMARY KEY (instrument_id, "timestamp", feature_name)
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_features_instrument_ts_name'
    ) THEN
        ALTER TABLE features
            ADD CONSTRAINT uq_features_instrument_ts_name
            UNIQUE (instrument_id, "timestamp", feature_name);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_features_instrument_name_ts
    ON features (instrument_id, feature_name, "timestamp" DESC);

-- Convert to hypertable
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables
        WHERE hypertable_name = 'features'
    ) THEN
        PERFORM create_hypertable(
            'features',
            'timestamp',
            migrate_data       => TRUE,
            if_not_exists      => TRUE
        );
    END IF;
END $$;

-- ============================================================================
-- 5. model_registry
-- ============================================================================
CREATE TABLE IF NOT EXISTS model_registry (
    id                      UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name              VARCHAR(255)    NOT NULL,
    model_type              VARCHAR(100)    NOT NULL,
    description             TEXT,
    hyperparameters         JSONB           DEFAULT '{}'::jsonb,
    training_dataset_info   JSONB           DEFAULT '{}'::jsonb,
    performance_metrics     JSONB           DEFAULT '{}'::jsonb,
    model_path              VARCHAR(512),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_model_registry_model_name
    ON model_registry (model_name);

CREATE INDEX IF NOT EXISTS ix_model_registry_model_type
    ON model_registry (model_type);

-- ============================================================================
-- 6. strategy_registry
-- ============================================================================
CREATE TABLE IF NOT EXISTS strategy_registry (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_name       VARCHAR(255)    NOT NULL,
    description         TEXT,
    parameters          JSONB           DEFAULT '{}'::jsonb,
    dependencies        JSONB           DEFAULT '{}'::jsonb,
    performance_metrics JSONB           DEFAULT '{}'::jsonb,
    last_evaluated_at   TIMESTAMPTZ,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_strategy_registry_strategy_name
    ON strategy_registry (strategy_name);

-- ============================================================================
-- 7. backtest_results
-- ============================================================================
CREATE TABLE IF NOT EXISTS backtest_results (
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id     UUID            NOT NULL
        REFERENCES strategy_registry (id) ON DELETE CASCADE,
    model_id        UUID
        REFERENCES model_registry (id) ON DELETE SET NULL,
    instrument_ids  JSONB           NOT NULL DEFAULT '[]'::jsonb,
    start_date      TIMESTAMPTZ     NOT NULL,
    end_date        TIMESTAMPTZ     NOT NULL,
    initial_capital NUMERIC(18,2)   NOT NULL,
    final_capital   NUMERIC(18,2)   NOT NULL,
    sharpe_ratio    DOUBLE PRECISION,
    sortino_ratio   DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    cagr            DOUBLE PRECISION,
    win_rate        DOUBLE PRECISION,
    total_trades    INTEGER         DEFAULT 0,
    parameters      JSONB           DEFAULT '{}'::jsonb,
    equity_curve    JSONB,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_backtest_strategy_id
    ON backtest_results (strategy_id);

CREATE INDEX IF NOT EXISTS ix_backtest_model_id
    ON backtest_results (model_id);

CREATE INDEX IF NOT EXISTS ix_backtest_strategy_created
    ON backtest_results (strategy_id, created_at DESC);

-- ============================================================================
-- 8. Record this migration
-- ============================================================================
INSERT INTO schema_migrations (version, description)
VALUES ('001', 'Initial schema with instruments, ohlcv_data, features, model_registry, strategy_registry, backtest_results')
ON CONFLICT (version) DO NOTHING;
