"""Feature engineering module for the quant-lab pipeline.

Quick-start::

    from features import FeatureEngine, FEATURE_REGISTRY, list_features

    engine = FeatureEngine()
    print(list_features())
    features_df = engine.compute_features(ohlcv_df)
"""

from features.engine import FeatureEngine
from features.registry import (
    FEATURE_REGISTRY,
    get_feature_func,
    list_features,
    register_feature,
)

__all__ = [
    "FeatureEngine",
    "FEATURE_REGISTRY",
    "get_feature_func",
    "list_features",
    "register_feature",
]
