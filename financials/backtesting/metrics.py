"""Pure functions for computing backtest performance metrics.

All functions operate on numpy arrays or plain Python sequences and are
stateless -- they carry no side effects and return deterministic results
for the same inputs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from financials.backtesting.constants import resolve_periods

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def compute_sharpe_ratio(
    returns: np.ndarray | pd.Series,
    risk_free_rate: float = 0.04,
    periods: int = 252,
) -> float:
    """Compute the annualised Sharpe ratio.

    Args:
        returns: Array of periodic (e.g. daily) simple returns.
        risk_free_rate: Annualised risk-free rate used to compute excess
            returns.  Defaults to 4 %.
        periods: Number of periods per year (252 for daily, 52 for weekly,
            12 for monthly).

    Returns:
        Annualised Sharpe ratio.  Returns 0.0 when the standard deviation
        of excess returns is zero (i.e. constant returns).
    """
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2:
        logger.warning("Sharpe ratio requires at least 2 return observations; returning 0.0.")
        return 0.0

    periodic_rf = risk_free_rate / periods
    excess = returns - periodic_rf

    std = np.std(excess, ddof=1)
    if std == 0.0:
        return 0.0

    sharpe = (np.mean(excess) / std) * np.sqrt(periods)
    return float(sharpe)


def compute_sortino_ratio(
    returns: np.ndarray | pd.Series,
    risk_free_rate: float = 0.04,
    periods: int = 252,
) -> float:
    """Compute the annualised Sortino ratio.

    Like the Sharpe ratio but only penalises downside volatility
    (returns below the risk-free rate).

    Args:
        returns: Array of periodic simple returns.
        risk_free_rate: Annualised risk-free rate.
        periods: Number of periods per year.

    Returns:
        Annualised Sortino ratio.  Returns 0.0 when downside deviation
        is zero.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2:
        logger.warning("Sortino ratio requires at least 2 return observations; returning 0.0.")
        return 0.0

    periodic_rf = risk_free_rate / periods
    excess = returns - periodic_rf

    downside = excess[excess < 0]
    if len(downside) == 0:
        # No downside -- Sortino is theoretically infinite; clamp to 0.
        return 0.0

    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std == 0.0:
        return 0.0

    sortino = (np.mean(excess) / downside_std) * np.sqrt(periods)
    return float(sortino)


def compute_alpha_beta(
    strategy_returns: np.ndarray | pd.Series,
    benchmark_returns: np.ndarray | pd.Series,
    periods: int | None = None,
    interval: str | None = None,
) -> tuple[float, float]:
    """Compute annualised alpha and beta of a strategy against a benchmark.

    Beta is the OLS slope of *strategy_returns* regressed on
    *benchmark_returns*; alpha is the intercept, annualised by *periods*.

    Both inputs must be **realised return series of the same length** --
    typically the strategy's periodic returns and the underlying asset's
    buy-and-hold returns over the same bars.  Passing positions/signals as
    the benchmark (as this function used to do internally) produces
    meaningless numbers.

    Args:
        strategy_returns: Periodic simple returns of the strategy.
        benchmark_returns: Periodic simple returns of the benchmark,
            aligned one-for-one with *strategy_returns*.
        periods: Bars per year used to annualise alpha.
        interval: Bar interval (e.g. ``"1min"``) as an alternative to
            *periods*.

    Returns:
        Tuple of ``(annualised_alpha, beta)``.  Returns ``(0.0, 0.0)`` when
        fewer than 2 overlapping observations are available or the
        benchmark has zero variance.

    Raises:
        ValueError: If the two series differ in length, or if neither
            *periods* nor *interval* is supplied.
    """
    strat = np.asarray(strategy_returns, dtype=np.float64)
    bench = np.asarray(benchmark_returns, dtype=np.float64)

    if strat.shape != bench.shape:
        raise ValueError(
            f"strategy_returns and benchmark_returns must be the same length; "
            f"got {strat.shape} and {bench.shape}."
        )

    ann = resolve_periods(interval=interval, periods=periods)

    mask = ~np.isnan(strat) & ~np.isnan(bench)
    strat = strat[mask]
    bench = bench[mask]

    if len(strat) < 2:
        logger.warning("Alpha and beta require at least 2 paired observations; returning 0.0.")
        return 0.0, 0.0

    var_benchmark = np.var(bench, ddof=1)
    if var_benchmark == 0.0:
        logger.warning("Benchmark has zero variance; alpha/beta undefined, returning 0.0.")
        return 0.0, 0.0

    beta = float(np.cov(strat, bench, ddof=1)[0, 1] / var_benchmark)
    alpha = float((strat.mean() - beta * bench.mean()) * ann)

    return alpha, beta


def compute_max_drawdown(equity_curve: np.ndarray | pd.Series | list) -> float:
    """Compute the maximum drawdown as a positive percentage.

    Args:
        equity_curve: Sequence of portfolio equity values over time.

    Returns:
        Maximum drawdown expressed as a positive fraction (e.g. 0.25 means
        a 25 % peak-to-trough decline).  Returns 0.0 for monotonically
        increasing equity curves.
    """
    equity = np.asarray(equity_curve, dtype=np.float64)
    if len(equity) < 2:
        return 0.0

    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / np.where(running_max == 0, 1.0, running_max)
    max_dd = float(np.max(drawdowns))
    return max_dd


def compute_cagr(
    initial_value: float,
    final_value: float,
    years: float,
) -> float:
    """Compute the Compound Annual Growth Rate (CAGR).

    Args:
        initial_value: Starting portfolio value.
        final_value: Ending portfolio value.
        years: Duration of the period in years.

    Returns:
        CAGR as a decimal (e.g. 0.12 means 12 % per year).  Returns 0.0
        when *years* is non-positive or *initial_value* is not positive.
    """
    if years <= 0 or initial_value <= 0:
        logger.warning(
            "CAGR requires positive initial_value and years; returning 0.0 "
            "(initial_value=%.2f, years=%.4f).",
            initial_value,
            years,
        )
        return 0.0

    cagr = (final_value / initial_value) ** (1.0 / years) - 1.0
    return float(cagr)


def compute_win_rate(trades: list[dict[str, Any]]) -> float:
    """Compute the percentage of winning trades.

    A *winning* trade has ``pnl > 0``.

    Args:
        trades: List of trade dictionaries, each containing at least a
            ``"pnl"`` key.

    Returns:
        Win rate as a decimal in [0, 1].  Returns 0.0 when *trades* is
        empty.
    """
    if not trades:
        return 0.0

    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return wins / len(trades)


def compute_profit_factor(trades: list[dict[str, Any]]) -> float:
    """Compute the profit factor (gross profit / gross loss).

    Args:
        trades: List of trade dictionaries, each containing at least a
            ``"pnl"`` key.

    Returns:
        Profit factor.  Returns ``float('inf')`` when there are no losing
        trades but at least one winning trade.  Returns 0.0 when there
        are no trades or no winning trades.
    """
    if not trades:
        return 0.0

    gross_profit = sum(t["pnl"] for t in trades if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t.get("pnl", 0) < 0))

    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def compute_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """Compute the Calmar ratio (CAGR / max drawdown).

    Args:
        cagr: Compound annual growth rate as a decimal.
        max_drawdown: Maximum drawdown as a positive decimal.

    Returns:
        Calmar ratio.  Returns 0.0 when *max_drawdown* is zero.
    """
    if max_drawdown == 0.0:
        return 0.0

    return cagr / max_drawdown


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------

def compute_all_metrics(
    equity_curve: np.ndarray | pd.Series | list,
    trades: list[dict[str, Any]],
    risk_free_rate: float = 0.04,
    periods: int | None = None,
    interval: str | None = None,
    benchmark_returns: np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """Compute every available performance metric in one call.

    Args:
        equity_curve: Portfolio equity values over time.
        trades: List of trade dictionaries (must contain ``"pnl"``).
        risk_free_rate: Annualised risk-free rate.
        periods: Bars per year, used to annualise Sharpe, Sortino and CAGR.
        interval: Bar interval (e.g. ``"1min"``, ``"1d"``) as an
            alternative to *periods*.  Exactly one of the two is required --
            there is no default, because annualising intraday bars at the
            daily 252 convention overstates Sharpe by up to ~19.7x.
        benchmark_returns: Optional periodic simple returns of a benchmark
            (e.g. the underlying asset's buy-and-hold returns), aligned
            with the equity curve's own returns.  When supplied, ``alpha``
            and ``beta`` are included in the output.

    Returns:
        Dictionary mapping metric names to their computed values.

    Raises:
        ValueError: If neither *periods* nor *interval* is supplied.
    """
    ann = resolve_periods(interval=interval, periods=periods)

    equity = np.asarray(equity_curve, dtype=np.float64)

    # Derive periodic simple returns from the equity curve.
    if len(equity) >= 2:
        returns = np.diff(equity) / np.where(equity[:-1] == 0, 1.0, equity[:-1])
    else:
        returns = np.array([], dtype=np.float64)

    initial_value = float(equity[0]) if len(equity) > 0 else 0.0
    final_value = float(equity[-1]) if len(equity) > 0 else 0.0
    years = len(equity) / ann

    max_dd = compute_max_drawdown(equity)
    cagr = compute_cagr(initial_value, final_value, years)

    metrics: dict[str, float] = {
        "sharpe_ratio": compute_sharpe_ratio(returns, risk_free_rate, ann),
        "sortino_ratio": compute_sortino_ratio(returns, risk_free_rate, ann),
        "max_drawdown": max_dd,
        "cagr": cagr,
        "calmar_ratio": compute_calmar_ratio(cagr, max_dd),
        "win_rate": compute_win_rate(trades),
        "profit_factor": compute_profit_factor(trades),
        "total_trades": len(trades),
        "initial_capital": initial_value,
        "final_capital": final_value,
        "total_return": (final_value / initial_value - 1.0) if initial_value > 0 else 0.0,
        "periods_per_year": ann,
    }

    if benchmark_returns is not None:
        bench = np.asarray(benchmark_returns, dtype=np.float64)
        # The equity curve yields one fewer return than it has points; trim a
        # benchmark supplied at full bar length so the two align.
        if len(bench) == len(returns) + 1:
            bench = bench[1:]
        if len(bench) != len(returns):
            logger.warning(
                "benchmark_returns length %d does not align with %d strategy returns; "
                "skipping alpha/beta.",
                len(bench),
                len(returns),
            )
        else:
            alpha, beta = compute_alpha_beta(returns, bench, periods=ann)
            metrics["alpha"] = alpha
            metrics["beta"] = beta

    logger.info(
        "Metrics computed (periods/yr=%d): Sharpe=%.3f  Sortino=%.3f  MaxDD=%.2f%%  "
        "CAGR=%.2f%%  WinRate=%.1f%%  Trades=%d",
        ann,
        metrics["sharpe_ratio"],
        metrics["sortino_ratio"],
        metrics["max_drawdown"] * 100,
        metrics["cagr"] * 100,
        metrics["win_rate"] * 100,
        metrics["total_trades"],
    )

    return metrics
