"""Fast vectorized backtesting using numpy and optionally vectorbt.

Provides :class:`VectorizedBacktest` as a high-performance alternative
to the event-driven :class:`BacktestEngine` for rapid signal evaluation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from financials.backtesting.constants import resolve_periods
from financials.backtesting.metrics import compute_all_metrics
from financials.config.settings import settings

logger = logging.getLogger(__name__)


class VectorizedBacktest:
    """High-speed vectorized backtest using pure numpy.

    This class sacrifices some realism (no partial fills, simplified
    commission model) in exchange for speed -- suitable for signal
    screening and parameter sweeps.
    """

    def run(
        self,
        signals: pd.Series | np.ndarray,
        prices: pd.Series | np.ndarray,
        initial_capital: float | None = None,
        commission: float | None = None,
        interval: str | None = None,
    ) -> dict[str, Any]:
        """Run a vectorized backtest.

        The simulation assumes full investment on buy and full liquidation
        on sell.  Positions are sized as ``capital / price`` at each entry.

        Args:
            signals: Array of signals (1 = buy, -1 = sell, 0 = hold).
            prices: Array of close prices aligned with *signals*.
            initial_capital: Starting cash (defaults to settings).
            commission: Proportional commission per leg (defaults to settings).
            interval: Bar interval (e.g. ``"1min"``, ``"1d"``) driving
                annualisation.  Inferred from *prices* when it carries a
                ``DatetimeIndex``.

        Returns:
            Dictionary with keys ``equity_curve``, ``trades``, ``metrics``.

        Raises:
            ValueError: If the interval can be neither given nor inferred.
        """
        initial_capital = initial_capital or settings.backtest.initial_capital
        commission = commission or settings.backtest.commission
        ann = resolve_periods(
            interval=interval,
            index=prices.index if isinstance(prices, pd.Series) else None,
        )

        signals = np.asarray(signals, dtype=np.int64)
        prices = np.asarray(prices, dtype=np.float64)
        n = len(prices)

        equity = np.full(n, initial_capital, dtype=np.float64)
        cash = initial_capital
        shares = 0.0
        in_position = False
        trades: list[dict[str, Any]] = []
        entry_price = 0.0
        entry_idx = 0

        for i in range(n):
            sig = signals[i]

            if sig == 1 and not in_position:
                shares = int(cash / (prices[i] * (1 + commission)))
                if shares > 0:
                    cost = shares * prices[i] * (1 + commission)
                    cash -= cost
                    in_position = True
                    entry_price = prices[i]
                    entry_idx = i

            elif sig == -1 and in_position:
                proceeds = shares * prices[i] * (1 - commission)
                cash += proceeds
                pnl = (prices[i] - entry_price) * shares
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": float(entry_price),
                    "exit_price": float(prices[i]),
                    "shares": int(shares),
                    "pnl": float(pnl),
                    "return_pct": float(prices[i] / entry_price - 1),
                    "holding_period": i - entry_idx,
                })
                shares = 0.0
                in_position = False

            equity[i] = cash + shares * prices[i]

        benchmark_returns = np.diff(prices) / np.where(prices[:-1] == 0.0, 1.0, prices[:-1])

        metrics = compute_all_metrics(
            equity_curve=equity,
            trades=trades,
            risk_free_rate=settings.backtest.risk_free_rate,
            periods=ann,
            benchmark_returns=benchmark_returns,
        )

        logger.info(
            "Vectorized backtest complete: %d trades, Sharpe=%.3f, MaxDD=%.2f%%",
            len(trades),
            metrics.get("sharpe_ratio", 0),
            metrics.get("max_drawdown", 0) * 100,
        )

        return {
            "equity_curve": equity.tolist(),
            "trades": trades,
            "metrics": metrics,
        }

    def run_with_vectorbt(
        self,
        signals: pd.Series,
        prices: pd.Series,
        initial_capital: float | None = None,
        fees: float | None = None,
        interval: str | None = None,
    ) -> dict[str, Any]:
        """Run a backtest using the vectorbt library.

        Requires ``vectorbt`` to be installed.

        Args:
            signals: Pandas Series of signals (1 / -1 / 0).
            prices: Pandas Series of close prices (same index as signals).
            initial_capital: Starting cash.
            fees: Commission per trade as a fraction.
            interval: Bar interval driving annualisation.  Inferred from
                *prices*' ``DatetimeIndex`` when omitted.

        Returns:
            Dictionary with ``equity_curve``, ``trades``, ``metrics``,
            and ``vbt_portfolio`` (the raw vectorbt Portfolio object).
        """
        try:
            import vectorbt as vbt
        except ImportError:
            logger.error(
                "vectorbt is not installed. Install it with: "
                "pip install vectorbt"
            )
            raise

        initial_capital = initial_capital or settings.backtest.initial_capital
        fees = fees or settings.backtest.commission
        ann = resolve_periods(
            interval=interval,
            index=prices.index if isinstance(prices, pd.Series) else None,
        )

        # Build boolean entry / exit arrays.
        entries = signals == 1
        exits = signals == -1

        portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=entries,
            exits=exits,
            init_cash=initial_capital,
            fees=fees,
            freq="1D",
        )

        equity_curve = portfolio.value().values
        prices_arr = np.asarray(prices, dtype=np.float64)
        benchmark_returns = np.diff(prices_arr) / np.where(
            prices_arr[:-1] == 0.0, 1.0, prices_arr[:-1]
        )

        metrics = compute_all_metrics(
            equity_curve=equity_curve,
            trades=[],  # vectorbt manages its own trade log
            risk_free_rate=settings.backtest.risk_free_rate,
            periods=ann,
            benchmark_returns=benchmark_returns,
        )

        # Enrich with vectorbt's own metrics.
        metrics["vbt_sharpe"] = float(portfolio.sharpe_ratio())
        metrics["vbt_max_drawdown"] = float(portfolio.max_drawdown())
        metrics["vbt_total_return"] = float(portfolio.total_return())

        logger.info(
            "vectorbt backtest complete: Sharpe=%.3f, MaxDD=%.2f%%",
            metrics["vbt_sharpe"],
            metrics["vbt_max_drawdown"] * 100,
        )

        return {
            "equity_curve": equity_curve.tolist(),
            "trades": [],
            "metrics": metrics,
            "vbt_portfolio": portfolio,
        }
