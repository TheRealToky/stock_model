"""Feature engineering module for the quant-lab pipeline.

Quick-start::

    from financials.features import FeatureEngine, FEATURE_REGISTRY, list_features

    engine = FeatureEngine()
    print(list_features())
    features_df = engine.compute_features(ohlcv_df)

The registry and the pure indicator functions are imported eagerly -- they
depend on nothing but pandas/numpy.  :class:`FeatureEngine` is resolved
lazily (PEP 562) because it reaches into :mod:`financials.data_pipeline`
for its DB-backed loader, which pulls in yfinance and a Postgres driver.
Computing an RSI should not require a market-data client.
"""

from typing import TYPE_CHECKING

from financials.features.registry import (
    FEATURE_REGISTRY,
    get_feature_func,
    list_features,
    register_feature,
)

if TYPE_CHECKING:  # pragma: no cover - static analysers only
    from financials.features.engine import FeatureEngine

__all__ = [
    "FeatureEngine",
    "FEATURE_REGISTRY",
    "get_feature_func",
    "list_features",
    "register_feature",
]


def __getattr__(name: str):
    """Import :class:`FeatureEngine` on first access (PEP 562)."""
    if name == "FeatureEngine":
        from financials.features.engine import FeatureEngine

        globals()[name] = FeatureEngine
        return FeatureEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
