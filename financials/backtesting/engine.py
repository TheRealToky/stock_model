"""Core backtesting engine for the quant-lab.

Provides :class:`BacktestEngine` which runs a :class:`BaseStrategy` against
historical price data and produces a :class:`BacktestResult` with the full
equity curve, individual trade log, and computed performance metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from financials.config.settings import settings
from financials.backtesting.metrics import compute_all_metrics
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Immutable container for a single backtest run.

    Attributes:
        equity_curve: List of daily portfolio equity values.
        trades: List of trade dictionaries with keys ``entry_date``,
            ``exit_date``, ``entry_price``, ``exit_price``, ``shares``,
            ``pnl``, ``return_pct``, ``holding_period``.
        metrics: Dictionary of performance metrics (Sharpe, Sortino, etc.).
        parameters: Strategy parameters used during the run.
        strategy_name: Name of the strategy that was backtested.
        ticker: Instrument symbol.
        start_date: First date in the backtest window.
        end_date: Last date in the backtest window.
    """

    equity_curve: list[float]
    trades: list[dict[str, Any]]
    metrics: dict[str, float]
    parameters: dict[str, Any]
    strategy_name: str
    ticker: str
    start_date: str
    end_date: str


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Event-driven backtesting engine.

    The engine accepts a :class:`BaseStrategy`, feeds it market data, and
    simulates order execution with configurable commission and slippage.

    Args:
        initial_capital: Starting portfolio cash.  Defaults to the value
            in ``settings.backtest.initial_capital``.
        commission: Proportional commission applied on each trade leg
            (entry and exit).  Defaults to ``settings.backtest.commission``.
        slippage: Proportional slippage applied on each trade leg.
            Defaults to ``settings.backtest.slippage``.
    """

    def __init__(
        self,
        initial_capital: float | None = None,
        commission: float | None = None,
        slippage: float | None = None,
    ) -> None:
        self.initial_capital = (
            initial_capital if initial_capital is not None else settings.backtest.initial_capital
        )
        self.commission = (
            commission if commission is not None else settings.backtest.commission
        )
        self.slippage = (
            slippage if slippage is not None else settings.backtest.slippage
        )

        logger.info(
            "BacktestEngine initialised: capital=%.2f  commission=%.4f  slippage=%.4f",
            self.initial_capital,
            self.commission,
            self.slippage,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        ticker: str = "UNKNOWN",
    ) -> BacktestResult:
        """Run a single backtest.

        Args:
            strategy: An instance of :class:`BaseStrategy`.
            df: DataFrame with at least a ``close`` column and a
                :class:`~pandas.DatetimeIndex`.
            ticker: Instrument symbol for labelling.

        Returns:
            A :class:`BacktestResult` containing the equity curve, trade
            log, and performance metrics.

        Raises:
            ValueError: If *df* is empty or lacks a ``close`` column.
        """
        if df.empty:
            raise ValueError("Cannot run backtest on an empty DataFrame.")
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column.")
        if "open" not in df.columns:
            raise ValueError("DataFrame must contain an 'open' column.")

        # Let the strategy validate its own dependencies.
        strategy.validate_dataframe(df)

        logger.info(
            "Running strategy '%s' on %s (%d bars, %s -> %s)",
            strategy.name,
            ticker,
            len(df),
            df.index[0],
            df.index[-1],
        )

        # Generate signals.
        signals = strategy.generate_signals(df)

        # Shift signals by a day to avoid look ahead bias
        signals = signals.shift(1).fillna(0)

        open_prices = df["open"].values.astype(np.float64)
        close_prices = df["close"].values.astype(np.float64)

        # Build date labels.
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.strftime("%Y-%m-%d").tolist()
        else:
            dates = [str(d) for d in df.index]

        # Simulate.
        equity_curve, trades = self._simulate_trades(
            signals=signals.values if isinstance(signals, pd.Series) else np.asarray(signals),
            open_prices=open_prices,
            close_prices=close_prices,
            dates=dates,
            initial_capital=self.initial_capital,
            commission=self.commission,
            slippage=self.slippage,
        )

        # Metrics.
        metrics = compute_all_metrics(
            equity_curve=equity_curve,
            trades=trades,
            risk_free_rate=settings.backtest.risk_free_rate,
            signals=signals,
        )

        result = BacktestResult(
            equity_curve=equity_curve.tolist() if isinstance(equity_curve, np.ndarray) else equity_curve,
            trades=trades,
            metrics=metrics,
            parameters=strategy.get_parameters(),
            strategy_name=strategy.name,
            ticker=ticker,
            start_date=dates[0],
            end_date=dates[-1],
        )

        logger.info(
            "Backtest complete for '%s' on %s: final_capital=%.2f  sharpe=%.3f  "
            "max_dd=%.2f%%  trades=%d",
            strategy.name,
            ticker,
            metrics.get("final_capital", 0),
            metrics.get("sharpe_ratio", 0),
            metrics.get("max_drawdown", 0) * 100,
            metrics.get("total_trades", 0),
        )

        return result

    def run_multiple(
        self,
        strategy: BaseStrategy,
        tickers: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[BacktestResult]:
        """Run the same strategy across multiple instruments.

        This method loads data for each ticker via :class:`DataLoader`,
        then delegates to :meth:`run`.

        Args:
            strategy: An instance of :class:`BaseStrategy`.
            tickers: List of instrument symbols.
            start_date: ISO-formatted start date (inclusive).  If *None*,
                uses ``settings.pipeline.default_start_date``.
            end_date: ISO-formatted end date (inclusive).  If *None*, uses
                today's date.

        Returns:
            List of :class:`BacktestResult` -- one per ticker that had
            sufficient data.  Tickers that fail are logged and skipped.
        """
        from financials.data_pipeline.loader import DataLoader

        loader = DataLoader()
        results: list[BacktestResult] = []

        effective_start = start_date or settings.pipeline.default_start_date
        effective_end = end_date or datetime.now().strftime("%Y-%m-%d")

        logger.info(
            "Running strategy '%s' on %d tickers (%s -> %s)",
            strategy.name,
            len(tickers),
            effective_start,
            effective_end,
        )

        for ticker in tickers:
            try:
                df = loader.load(
                    ticker=ticker,
                    start_date=effective_start,
                    end_date=effective_end,
                )
                if df.empty:
                    logger.warning("No data returned for %s; skipping.", ticker)
                    continue

                result = self.run(strategy, df, ticker=ticker)
                results.append(result)
            except Exception:
                logger.exception("Backtest failed for %s; skipping.", ticker)

        logger.info(
            "Multi-ticker backtest complete: %d/%d tickers succeeded.",
            len(results),
            len(tickers),
        )

        return results

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_trades(
        signals: np.ndarray,
        open_prices: np.ndarray,
        close_prices: np.ndarray,
        dates: list[str],
        initial_capital: float,
        commission: float,
        slippage: float,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Simulate trades from a signal array.

        Signals are assumed to already be shifted so that the signal at
        index *i* represents a decision made *before* bar *i* opens.
        Trades therefore execute at the **open** price of bar *i*
        (with slippage), while the equity curve is marked to market at
        each bar's **close** price.

        Signal convention:

        * ``1``  -- open a long position (buy) if flat.
        * ``-1`` -- close the current long position (sell) if held.
        * ``0``  -- hold the current position.

        Commission and slippage are applied symmetrically on both entry
        and exit:

        * **Entry price** = ``open * (1 + slippage)`` (buy at a worse
          price).
        * **Exit price** = ``open * (1 - slippage)`` (sell at a worse
          price).
        * **Commission** = ``cost_basis * commission`` on each leg.

        Args:
            signals: 1-D array of trading signals (1, -1, 0).
            open_prices: 1-D array of open prices aligned with *signals*.
            close_prices: 1-D array of close prices aligned with
                *signals* (used only for mark-to-market).
            dates: List of date strings aligned with *signals* / *prices*.
            initial_capital: Starting cash.
            commission: Proportional commission per trade leg.
            slippage: Proportional slippage per trade leg.

        Returns:
            A tuple of ``(equity_curve, trades)`` where *equity_curve* is
            a numpy array of portfolio values and *trades* is a list of
            trade dictionaries.
        """
        n = len(open_prices)
        equity_curve = np.empty(n, dtype=np.float64)

        cash: float = initial_capital
        shares: float = 0.0
        position_open: bool = False
        entry_price: float = 0.0
        entry_date: str = ""
        entry_idx: int = 0

        trades: list[dict[str, Any]] = []

        for i in range(n):
            signal = int(signals[i]) if i < len(signals) else 0
            exec_price = open_prices[i]

            if signal == 1 and not position_open:
                # --- OPEN POSITION ---
                adjusted_price = exec_price * (1.0 + slippage)
                max_shares = cash / (adjusted_price * (1.0 + commission))
                shares = int(max_shares)  # whole shares only

                if shares > 0:
                    cost_basis = shares * adjusted_price
                    commission_cost = cost_basis * commission
                    cash -= cost_basis + commission_cost

                    position_open = True
                    entry_price = adjusted_price
                    entry_date = dates[i]
                    entry_idx = i

                    logger.debug(
                        "BUY  %d shares @ %.4f on %s  (commission=%.2f)",
                        shares,
                        adjusted_price,
                        dates[i],
                        commission_cost,
                    )

            elif signal == -1 and position_open:
                # --- CLOSE POSITION ---
                adjusted_price = exec_price * (1.0 - slippage)
                proceeds = shares * adjusted_price
                commission_cost = proceeds * commission
                cash += proceeds - commission_cost

                pnl = (adjusted_price - entry_price) * shares - (
                    # Entry commission + exit commission
                    (entry_price * shares * commission)
                    + commission_cost
                )
                return_pct = (adjusted_price / entry_price - 1.0) if entry_price > 0 else 0.0
                holding_period = i - entry_idx

                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": dates[i],
                        "entry_price": round(entry_price, 6),
                        "exit_price": round(adjusted_price, 6),
                        "shares": shares,
                        "pnl": round(float(pnl), 2),
                        "return_pct": round(float(return_pct), 6),
                        "holding_period": holding_period,
                    }
                )

                logger.debug(
                    "SELL %d shares @ %.4f on %s  pnl=%.2f  holding=%d days",
                    shares,
                    adjusted_price,
                    dates[i],
                    pnl,
                    holding_period,
                )

                shares = 0.0
                position_open = False

            # --- MARK TO MARKET at close ---
            equity_curve[i] = cash + shares * close_prices[i]

        # If a position is still open at the end, mark it to market (no trade recorded).
        if position_open:
            logger.info(
                "Position still open at end of backtest (%d shares bought @ %.4f on %s).",
                int(shares),
                entry_price,
                entry_date,
            )

        return equity_curve, trades
