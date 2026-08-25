"""Backtesting subsystem for the quant lab."""

from financials.backtesting.constants import (
    PERIODS_PER_YEAR,
    infer_interval,
    periods_per_year,
    resolve_periods,
)
from financials.backtesting.engine import BacktestEngine, BacktestResult
from financials.backtesting.metrics import (
    compute_all_metrics,
    compute_alpha_beta,
    compute_cagr,
    compute_calmar_ratio,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_win_rate,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "PERIODS_PER_YEAR",
    "infer_interval",
    "periods_per_year",
    "resolve_periods",
    "compute_alpha_beta",
    "compute_all_metrics",
    "compute_sharpe_ratio",
    "compute_sortino_ratio",
    "compute_max_drawdown",
    "compute_cagr",
    "compute_win_rate",
    "compute_profit_factor",
    "compute_calmar_ratio",
]
