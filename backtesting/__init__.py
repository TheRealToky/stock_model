"""Backtesting subsystem for the quant lab."""

from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.metrics import (
    compute_all_metrics,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_cagr,
    compute_win_rate,
    compute_profit_factor,
    compute_calmar_ratio,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "compute_all_metrics",
    "compute_sharpe_ratio",
    "compute_sortino_ratio",
    "compute_max_drawdown",
    "compute_cagr",
    "compute_win_rate",
    "compute_profit_factor",
    "compute_calmar_ratio",
]
