-- 001_initial_alt_schema.sql
-- Idempotent migration: creates all alt-data tables.
-- Safe to run multiple times (uses IF NOT EXISTS / DO blocks).

-- ============================================================================
-- 0. Migration tracking
-- ============================================================================
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     VARCHAR(50)  PRIMARY KEY,
    description TEXT,
    applied_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- 1. raw_flights -- immutable source-of-truth of every API response row
-- ============================================================================
CREATE TABLE IF NOT EXISTS raw_flights (
    icao24                                  VARCHAR(12)   NOT NULL,
    first_seen                              BIGINT        NOT NULL,
    direction                               VARCHAR(10)   NOT NULL,
    airport_icao                            VARCHAR(8)    NOT NULL,
    callsign                                VARCHAR(16),
    last_seen                               BIGINT        NOT NULL,
    est_departure_airport                   VARCHAR(8),
    est_arrival_airport                     VARCHAR(8),
    est_departure_airport_horiz_distance    INTEGER,
    est_departure_airport_vert_distance     INTEGER,
    est_arrival_airport_horiz_distance      INTEGER,
    est_arrival_airport_vert_distance       INTEGER,
    departure_airport_candidates_count      INTEGER,
    arrival_airport_candidates_count        INTEGER,
    ingested_at                             TIMESTAMPTZ   NOT NULL DEFAULT NOW(),

    PRIMARY KEY (icao24, first_seen, direction, airport_icao)
);

CREATE INDEX IF NOT EXISTS ix_raw_flights_airport_first_seen
    ON raw_flights (airport_icao, first_seen);
CREATE INDEX IF NOT EXISTS ix_raw_flights_direction
    ON raw_flights (direction);

-- ============================================================================
-- 2. cleaned_flights
-- ============================================================================
CREATE TABLE IF NOT EXISTS cleaned_flights (
    icao24                              VARCHAR(12)   NOT NULL,
    first_seen                          BIGINT        NOT NULL,
    direction                           VARCHAR(10)   NOT NULL,
    airport_icao                        VARCHAR(8)    NOT NULL,
    callsign                            VARCHAR(16),
    last_seen                           BIGINT        NOT NULL,
    first_seen_dt                       TIMESTAMPTZ   NOT NULL,
    last_seen_dt                        TIMESTAMPTZ   NOT NULL,
    est_departure_airport               VARCHAR(8),
    est_arrival_airport                 VARCHAR(8),
    est_departure_airport_horiz_distance INTEGER,
    est_arrival_airport_horiz_distance  INTEGER,
    departure_airport_candidates_count  INTEGER,
    arrival_airport_candidates_count    INTEGER,
    cleaned_at                          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),

    PRIMARY KEY (icao24, first_seen, direction, airport_icao)
);

CREATE INDEX IF NOT EXISTS ix_cleaned_flights_airport_time
    ON cleaned_flights (airport_icao, first_seen_dt);
CREATE INDEX IF NOT EXISTS ix_cleaned_flights_callsign
    ON cleaned_flights (callsign);
CREATE INDEX IF NOT EXISTS ix_cleaned_flights_route
    ON cleaned_flights (est_departure_airport, est_arrival_airport);
CREATE INDEX IF NOT EXISTS ix_cleaned_flights_first_seen_dt
    ON cleaned_flights (first_seen_dt);

-- ============================================================================
-- 3. flight_features -- long-format feature values
-- ============================================================================
CREATE TABLE IF NOT EXISTS flight_features (
    airport_icao    VARCHAR(8)       NOT NULL,
    "timestamp"     TIMESTAMPTZ      NOT NULL,
    feature_name    VARCHAR(100)     NOT NULL,
    feature_value   DOUBLE PRECISION,
    feature_meta    JSONB,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW(),

    PRIMARY KEY (airport_icao, "timestamp", feature_name)
);

CREATE INDEX IF NOT EXISTS ix_flight_features_airport_name_ts
    ON flight_features (airport_icao, feature_name, "timestamp");

-- ============================================================================
-- 4. Record migration
-- ============================================================================
INSERT INTO schema_migrations (version, description)
VALUES ('001', 'Initial alt-data schema: raw_flights, cleaned_flights, flight_features')
ON CONFLICT (version) DO NOTHING;
