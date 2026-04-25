"""Feature registry for the quant-lab feature pipeline.

Provides a central catalogue of available feature functions so that the
:class:`~features.engine.FeatureEngine` (or any other consumer) can look
features up by name, iterate over them, and dynamically register new ones.

Usage
-----
Register a feature at import time with the ``@register_feature`` decorator::

    from financials.features.registry import register_feature

    @register_feature("my_custom_indicator")
    def compute_my_indicator(df, window=10):
        ...

Or register programmatically::

    FEATURE_REGISTRY["my_custom_indicator"] = (compute_my_indicator, {"window": 10})
"""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger

from financials.features.technical import (
    compute_atr,
    compute_bollinger_bands,
    compute_ema,
    compute_log_returns,
    compute_macd,
    compute_price_features,
    compute_returns,
    compute_rolling_stats,
    compute_rsi,
    compute_sma,
    compute_volatility,
    compute_volume_features,
)

# ---------------------------------------------------------------------------
# Type alias for readability
# ---------------------------------------------------------------------------
FeatureFunc = Callable[..., Any]

# ---------------------------------------------------------------------------
# The global registry: feature_name -> (function, default_params_dict)
# ---------------------------------------------------------------------------
FEATURE_REGISTRY: dict[str, tuple[FeatureFunc, dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# Decorator for convenient registration
# ---------------------------------------------------------------------------

def register_feature(
    name: str,
    default_params: dict[str, Any] | None = None,
) -> Callable[[FeatureFunc], FeatureFunc]:
    """Decorator that registers a feature function under *name*.

    Parameters
    ----------
    name:
        Unique string identifier for the feature.
    default_params:
        Optional mapping of default keyword arguments that will be passed to the
        function when the caller does not override them.

    Example
    -------
    ::

        @register_feature("my_feature", default_params={"window": 30})
        def compute_my_feature(df, window=30):
            return df["close"].rolling(window).mean()
    """

    def _decorator(func: FeatureFunc) -> FeatureFunc:
        if name in FEATURE_REGISTRY:
            logger.warning(
                "Feature '{}' is already registered -- overwriting with {}",
                name,
                func.__qualname__,
            )
        FEATURE_REGISTRY[name] = (func, default_params or {})
        logger.debug("Registered feature '{}'", name)
        return func

    return _decorator


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_feature_func(name: str) -> tuple[FeatureFunc, dict[str, Any]]:
    """Return ``(function, default_params)`` for the given feature name.

    Raises
    ------
    KeyError
        If *name* is not found in the registry.
    """
    if name not in FEATURE_REGISTRY:
        raise KeyError(
            f"Feature '{name}' is not registered. "
            f"Available features: {list(FEATURE_REGISTRY.keys())}"
        )
    return FEATURE_REGISTRY[name]


def list_features() -> list[str]:
    """Return a sorted list of all registered feature names."""
    return sorted(FEATURE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in registrations (executed on import)
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    """Populate the registry with the standard technical indicators."""

    builtins: list[tuple[str, FeatureFunc, dict[str, Any]]] = [
        ("returns", compute_returns, {"periods": 1}),
        ("log_returns", compute_log_returns, {"periods": 1}),
        ("sma", compute_sma, {"window": 20}),
        ("ema", compute_ema, {"span": 20}),
        ("rsi", compute_rsi, {"window": 14}),
        ("macd", compute_macd, {"fast": 12, "slow": 26, "signal": 9}),
        ("bollinger_bands", compute_bollinger_bands, {"window": 20, "num_std": 2}),
        ("atr", compute_atr, {"window": 14}),
        ("volatility", compute_volatility, {"window": 20}),
        ("rolling_stats", compute_rolling_stats, {"window": 20}),
        ("volume_features", compute_volume_features, {"window": 20}),
        ("price_features", compute_price_features, {}),
    ]

    for name, func, params in builtins:
        FEATURE_REGISTRY[name] = (func, params)

    logger.debug("Registered {} built-in features", len(builtins))


_register_builtins()
