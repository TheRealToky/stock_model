-- 002_model_registry_versioning.sql
-- Idempotent migration: give model_registry real version history.
--
-- Why: `ModelRegistry.register()` upserted by model_name against a UNIQUE
-- index on that column, so every retrain OVERWROTE the previous entry. There
-- was no way to answer "what did v1 score?", to compare two runs, or to roll
-- back to a checkpoint -- the row simply became the newest run, and the older
-- metrics were gone. For a research lab whose whole purpose is comparing
-- experiments, that is the wrong storage model.
--
-- After this migration a register() call INSERTS a new version. The unique
-- constraint moves to (model_name, version), and `is_latest` marks the current
-- one so lookups by bare name still resolve to a single row.
--
-- Also adds scaler_path: a saved .pt is unusable without the fitted scaler
-- that normalised its inputs, and until now that file was written by the
-- training notebook and referenced by nothing.

-- ============================================================================
-- 1. New columns
-- ============================================================================
ALTER TABLE model_registry
    ADD COLUMN IF NOT EXISTS version       INTEGER      NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS run_id        VARCHAR(64),
    ADD COLUMN IF NOT EXISTS artifact_hash VARCHAR(64),
    ADD COLUMN IF NOT EXISTS scaler_path   VARCHAR(512),
    ADD COLUMN IF NOT EXISTS is_latest     BOOLEAN      NOT NULL DEFAULT TRUE;

COMMENT ON COLUMN model_registry.version IS
    'Monotonic per model_name; a new training run inserts version = max + 1.';
COMMENT ON COLUMN model_registry.run_id IS
    'Opaque identifier for the training run that produced this artefact.';
COMMENT ON COLUMN model_registry.artifact_hash IS
    'SHA-256 of the serialised model file, so a row can be tied to bytes on disk.';
COMMENT ON COLUMN model_registry.scaler_path IS
    'Fitted feature scaler saved beside the model; without it the artefact cannot reproduce its own normalisation.';
COMMENT ON COLUMN model_registry.is_latest IS
    'Exactly one TRUE row per model_name -- the version a bare-name lookup resolves to.';

-- ============================================================================
-- 2. Swap the uniqueness constraint from (model_name) to (model_name, version)
-- ============================================================================
DROP INDEX IF EXISTS ix_model_registry_model_name;

CREATE UNIQUE INDEX IF NOT EXISTS ix_model_registry_name_version
    ON model_registry (model_name, version);

-- Enforce "at most one latest per name" at the database level, so a crashed
-- register() cannot leave two rows both claiming to be current.
CREATE UNIQUE INDEX IF NOT EXISTS ix_model_registry_one_latest
    ON model_registry (model_name)
    WHERE is_latest;

CREATE INDEX IF NOT EXISTS ix_model_registry_run_id
    ON model_registry (run_id);

-- ============================================================================
-- 3. Backfill: existing rows become version 1 and remain current
-- ============================================================================
UPDATE model_registry
   SET version = 1, is_latest = TRUE
 WHERE version IS NULL OR version < 1;

-- ============================================================================
-- 4. Record the migration
-- ============================================================================
INSERT INTO schema_migrations (version, description)
VALUES ('002', 'model_registry versioning: version/run_id/artifact_hash/scaler_path/is_latest')
ON CONFLICT (version) DO NOTHING;
