"""Trading strategy subsystem for the quant lab."""

from strategies.base import BaseStrategy
from strategies.ema_crossover import EMACrossover
from strategies.mean_reversion import MeanReversion
from strategies.momentum import Momentum
from strategies.volatility_target import VolatilityTargetOverlay, BuyAndHold
from strategies.registry import StrategyRegistry

__all__ = [
    "BaseStrategy",
    "EMACrossover",
    "MeanReversion",
    "Momentum",
    "VolatilityTargetOverlay",
    "BuyAndHold",
    "StrategyRegistry",
]
