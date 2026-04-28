"""Resolve ``FeatureConfig`` against the global feature registry.

The existing :class:`~financials.features.engine.FeatureEngine` already
knows how to compute features given a list of names and the registry.
This module wraps that with two extras the ETL layer needs:

  * Apply :class:`FeatureConfig` to register custom parameter overrides
    (e.g. ``sma`` with window=50 instead of the registry default of 20).
  * Pre-compute the *flat* list of output column names so the transform
    layer can validate the resulting frame and the parquet schema stays
    stable across runs.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from financials.etl_pipeline.config.etl_config import FeatureConfig
from financials.etl_pipeline.utils.logging import get_logger
from financials.features.registry import (
    FEATURE_REGISTRY,
    get_feature_func,
    list_features,
)

logger = get_logger(__name__)


def resolve_features(cfg: FeatureConfig) -> list[str]:
    """Return the ordered list of feature names to compute.

    Side-effect: applies ``cfg.feature_params`` overrides to the registry
    so that subsequent calls to ``FeatureEngine.compute_features`` use the
    requested parameters. The registry is global module state, but each
    worker process has its own copy (it's re-imported on fork/spawn), so
    this is safe under multiprocessing.
    """
    names = cfg.enabled_features if cfg.enabled_features else list_features()

    # Sanity-check requested names exist in the registry
    unknown = [n for n in names if n not in FEATURE_REGISTRY]
    if unknown:
        raise KeyError(
            f"Unknown feature(s): {unknown}. Available: {list_features()}"
        )

    # Apply parameter overrides
    for name, overrides in cfg.feature_params.items():
        if name not in FEATURE_REGISTRY:
            logger.warning("Skipping override for unknown feature '{}'", name)
            continue
        func, defaults = FEATURE_REGISTRY[name]
        merged = {**defaults, **overrides}
        FEATURE_REGISTRY[name] = (func, merged)
        logger.debug("Override {} -> {}", name, merged)

    return list(names)


def expected_feature_columns(feature_names: list[str]) -> list[str]:
    """Return the flat output columns the FeatureEngine will produce.

    Mirrors the flattening logic in ``FeatureEngine.compute_features``:
    composite features (DataFrame outputs) become ``<name>_<sub>`` columns,
    scalar features keep their name as-is.

    We probe each feature with a tiny synthetic OHLCV frame so we don't
    have to maintain a parallel manifest of sub-columns; this is cheap
    (microseconds per feature) and stays in sync if the indicator
    implementation changes.
    """
    sample = _synthetic_ohlcv()
    cols: list[str] = []

    for name in feature_names:
        try:
            func, default_params = get_feature_func(name)
            output = func(sample, **default_params)
        except Exception as exc:
            logger.warning("Could not probe feature '{}' for column names: {}", name, exc)
            cols.append(name)
            continue

        if isinstance(output, pd.DataFrame):
            cols.extend(f"{name}_{c}" for c in output.columns)
        else:
            cols.append(name)

    return cols


def _synthetic_ohlcv(n: int = 60) -> pd.DataFrame:
    """A 60-row dummy frame sufficient to exercise every built-in feature."""
    idx = pd.date_range("2020-01-01", periods=n, freq="1min", tz="UTC")
    base = 100 + pd.Series(range(n), dtype="float64") * 0.1
    return pd.DataFrame(
        {
            "open": base.values,
            "high": (base + 0.5).values,
            "low": (base - 0.5).values,
            "close": (base + 0.1).values,
            "volume": [1000.0 + i for i in range(n)],
        },
        index=idx,
    )
