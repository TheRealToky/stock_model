"""Feature-engineering sub-package for flight data.

Importing the sub-package populates the registry with all built-in
technical features (registered via ``@register_feature`` decorators
inside :mod:`technical`).
"""

from alt_data_pipeline.src.features import technical  # noqa: F401  (side-effect)
from alt_data_pipeline.src.features.engine import FlightFeatureEngine
from alt_data_pipeline.src.features.registry import (
    FLIGHT_FEATURE_REGISTRY,
    get_feature_func,
    list_features,
    register_feature,
)

__all__ = [
    "FlightFeatureEngine",
    "FLIGHT_FEATURE_REGISTRY",
    "register_feature",
    "get_feature_func",
    "list_features",
]
