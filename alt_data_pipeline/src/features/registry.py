"""Registry of flight-data feature functions.

Each registered function takes a cleaned-flight DataFrame (as produced
by :class:`FlightDataCleaner`) plus optional kwargs, and returns an
enriched DataFrame indexed by ``(airport_icao, timestamp)`` with one
or more feature columns.

Mirrors the quant-lab :mod:`features.registry` pattern.
"""

from __future__ import annotations

from typing import Any, Callable

from alt_data_pipeline.src.utils.logging import get_logger

logger = get_logger(__name__)

FeatureFunc = Callable[..., Any]

FLIGHT_FEATURE_REGISTRY: dict[str, tuple[FeatureFunc, dict[str, Any]]] = {}


def register_feature(
    name: str,
    default_params: dict[str, Any] | None = None,
) -> Callable[[FeatureFunc], FeatureFunc]:
    """Decorator: register *func* in the flight-data feature registry."""

    def _decorator(func: FeatureFunc) -> FeatureFunc:
        if name in FLIGHT_FEATURE_REGISTRY:
            logger.warning(
                "Feature '{}' already registered -- overwriting with {}",
                name,
                func.__qualname__,
            )
        FLIGHT_FEATURE_REGISTRY[name] = (func, default_params or {})
        logger.debug("Registered flight feature '{}'", name)
        return func

    return _decorator


def get_feature_func(name: str) -> tuple[FeatureFunc, dict[str, Any]]:
    if name not in FLIGHT_FEATURE_REGISTRY:
        raise KeyError(
            f"Flight feature '{name}' is not registered. "
            f"Available: {list(FLIGHT_FEATURE_REGISTRY.keys())}"
        )
    return FLIGHT_FEATURE_REGISTRY[name]


def list_features() -> list[str]:
    return sorted(FLIGHT_FEATURE_REGISTRY.keys())
