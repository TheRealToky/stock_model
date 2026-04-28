"""Typed configuration for the ETL pipeline.

Configuration is loaded from a YAML file (default: ``pipeline.yaml`` next to
this module) and overlaid with environment variables for any field exposed
through :class:`~financials.config.settings.Settings`.

A small dataclass tree is used instead of pydantic to keep the dependency
footprint minimal -- the validation we need (paths, enums, ranges) is easy
to express directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_YAML = _THIS_DIR / "pipeline.yaml"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class ExtractConfig:
    """Settings governing how OHLCV rows are pulled out of TimescaleDB."""

    # Bar interval to extract from the ``ohlcv_data`` table.
    interval: str = "1min"

    # Date window for full builds. ``end_date=None`` means "up to latest".
    start_date: str = "2021-01-01"
    end_date: str | None = None

    # Tickers to process. Empty list ⇒ all instruments present in the DB.
    tickers: list[str] = field(default_factory=list)

    # Time-window chunk size used to break a long history into manageable
    # pieces. 60 days × 1-min ≈ 86k rows per chunk per ticker -- fits easily
    # in memory and keeps the feature engine fast.
    chunk_days: int = 60

    # Look-back rows to PREPEND to each chunk so rolling-window features
    # have warmed up by the time we hit the chunk start. Must be >= the
    # longest window used by any enabled feature (default covers MACD-26
    # and Bollinger/ATR/SMA-20 with margin).
    warmup_rows: int = 200

    # Number of rows pandas will fetch from the server cursor at a time.
    # Larger = fewer round-trips, more peak memory.
    server_fetch_size: int = 50_000


@dataclass
class FeatureConfig:
    """Which features to compute and their parameter overrides.

    ``enabled_features`` controls the *set* of features. Each entry refers
    to a name registered in :mod:`financials.features.registry`. ``None``
    means "all registered features".

    ``feature_params`` overlays the registry defaults with task-specific
    values, e.g. ``{"sma": {"window": 50}}``.
    """

    enabled_features: list[str] | None = None
    feature_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Drop rows where any ENABLED feature is NaN. Set False if your
    # downstream model handles missing values.
    drop_warmup_nans: bool = True


@dataclass
class LoadConfig:
    """Where and how to write the Parquet dataset."""

    # Root directory of the Hive-partitioned dataset. Created if missing.
    output_path: str = "data/feature_store"

    # Compression: snappy for speed, zstd for ~30% smaller files.
    compression: str = "zstd"
    compression_level: int = 3

    # Row-group size ≈ 256 MB compressed is a good default for analytics.
    # Expressed in rows: 200_000 × ~80 cols × 8 bytes ≈ 128 MB raw → ~30 MB zstd.
    row_group_size: int = 200_000

    # Whether to write a per-symbol second-level partition under each date.
    # date-only = best for cross-sectional XGBoost.
    # date+symbol = best for sequence-per-stock LSTM (single-stock reads
    # touch one tiny file per day).
    partition_by_symbol: bool = True

    # Floats stored as float32 to halve disk footprint with negligible
    # precision loss for ML features. Set False to keep float64.
    cast_float32: bool = True


@dataclass
class RuntimeConfig:
    """Process / parallelism / observability knobs."""

    log_level: str = "INFO"
    log_json: bool = False

    # Workers for full rebuilds. ``None`` ⇒ os.cpu_count().
    # Use 1 to disable multiprocessing (helpful for debugging).
    num_workers: int | None = None

    # Path to the manifest JSON (high-water marks). Relative paths are
    # resolved against ``LoadConfig.output_path``.
    manifest_filename: str = "_manifest.json"

    # Skip schema/range validation on each chunk (slightly faster but
    # lets bad data through).
    skip_validation: bool = False


@dataclass
class ETLConfig:
    """Root configuration object."""

    extract: ExtractConfig = field(default_factory=ExtractConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    load: LoadConfig = field(default_factory=LoadConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def manifest_path(self) -> Path:
        return Path(self.load.output_path) / self.runtime.manifest_filename


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``overlay`` into ``base``. Overlay wins on conflicts."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _build_section(cls, raw: dict[str, Any] | None):
    """Instantiate a dataclass section from a raw dict, ignoring unknown keys."""
    raw = raw or {}
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    known = {k: v for k, v in raw.items() if k in field_names}
    return cls(**known)


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ETLConfig:
    """Build an :class:`ETLConfig`, merging YAML + runtime overrides.

    Resolution order (later wins):
    1. Dataclass defaults
    2. YAML file at ``path`` (or the bundled ``pipeline.yaml``)
    3. ``overrides`` dict (typically built from CLI args)
    """
    yaml_path = Path(path) if path else _DEFAULT_YAML
    raw: dict[str, Any] = {}
    if yaml_path.is_file():
        with yaml_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    if overrides:
        raw = _merge(raw, overrides)

    return ETLConfig(
        extract=_build_section(ExtractConfig, raw.get("extract")),
        features=_build_section(FeatureConfig, raw.get("features")),
        load=_build_section(LoadConfig, raw.get("load")),
        runtime=_build_section(RuntimeConfig, raw.get("runtime")),
    )
