"""Backtest report generation and persistence.

Provides :class:`BacktestReport` for producing human-readable summaries,
saving results to the database, and comparing multiple backtests.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import text

from financials.config.settings import settings
from financials.database.connection import get_engine, get_session
from financials.backtesting.engine import BacktestResult

logger = logging.getLogger(__name__)


class BacktestReport:
    """Generate, persist, and compare backtest results."""

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------

    def generate(self, result: BacktestResult) -> str:
        """Return a formatted text summary of a backtest result.

        Args:
            result: A :class:`BacktestResult` from the backtest engine.

        Returns:
            Multi-line string summary suitable for logging or display.
        """
        m = result.metrics
        lines = [
            "=" * 60,
            f"  Backtest Report:  {result.strategy_name}",
            f"  Ticker:           {result.ticker}",
            f"  Period:           {result.start_date} -> {result.end_date}",
            "=" * 60,
            f"  Initial Capital:  ${m.get('initial_capital', 0):>14,.2f}",
            f"  Final Capital:    ${m.get('final_capital', 0):>14,.2f}",
            f"  Total Return:     {m.get('total_return', 0):>14.2%}",
            "-" * 60,
            f"  Sharpe Ratio:     {m.get('sharpe_ratio', 0):>14.3f}",
            f"  Sortino Ratio:    {m.get('sortino_ratio', 0):>14.3f}",
            f"  Calmar Ratio:     {m.get('calmar_ratio', 0):>14.3f}",
            f"  Max Drawdown:     {m.get('max_drawdown', 0):>14.2%}",
            f"  CAGR:             {m.get('cagr', 0):>14.2%}",
            "-" * 60,
            f"  Alpha:            {m.get('alpha', 0):>14.3f}",
            f"  Beta:             {m.get('beta', 0):>14.3f}",
            "-" * 60,
            f"  Total Trades:     {m.get('total_trades', 0):>14d}",
            f"  Win Rate:         {m.get('win_rate', 0):>14.2%}",
            f"  Profit Factor:    {m.get('profit_factor', 0):>14.3f}",
            "=" * 60,
        ]

        if result.trades:
            pnls = [t["pnl"] for t in result.trades]
            lines += [
                f"  Avg PnL / Trade:  ${sum(pnls) / len(pnls):>14,.2f}",
                f"  Best Trade:       ${max(pnls):>14,.2f}",
                f"  Worst Trade:      ${min(pnls):>14,.2f}",
                "=" * 60,
            ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Database persistence
    # ------------------------------------------------------------------

    def save_to_db(
        self,
        result: BacktestResult,
        strategy_id: int | None = None,
        model_id: int | None = None,
    ) -> int:
        """Persist a backtest result to the ``backtest_results`` table.

        Args:
            result: The backtest result to save.
            strategy_id: Foreign key to ``strategy_registry.id``.
                If *None*, the method will attempt to look up the strategy
                by name.
            model_id: Optional foreign key to ``model_registry.id``.

        Returns:
            The auto-generated ``id`` of the inserted row.
        """
        engine = get_engine()

        # Resolve strategy_id by name if not provided.
        if strategy_id is None:
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT id FROM strategy_registry WHERE strategy_name = :name"),
                    {"name": result.strategy_name},
                ).fetchone()
                if row is not None:
                    strategy_id = row[0]

        # Subsample equity curve to avoid storing millions of points.
        equity_json = json.dumps(result.equity_curve[:5000])

        insert_sql = text("""
            INSERT INTO backtest_results (
                strategy_id, model_id, instrument_ids,
                start_date, end_date, initial_capital, final_capital,
                sharpe_ratio, sortino_ratio, max_drawdown, cagr,
                win_rate, total_trades, parameters, equity_curve, created_at
            ) VALUES (
                :strategy_id, :model_id, :instrument_ids,
                :start_date, :end_date, :initial_capital, :final_capital,
                :sharpe_ratio, :sortino_ratio, :max_drawdown, :cagr,
                :win_rate, :total_trades, :parameters, :equity_curve, NOW()
            ) RETURNING id
        """)

        m = result.metrics
        with engine.connect() as conn:
            row = conn.execute(
                insert_sql,
                {
                    "strategy_id": strategy_id,
                    "model_id": model_id,
                    "instrument_ids": json.dumps([result.ticker]),
                    "start_date": result.start_date,
                    "end_date": result.end_date,
                    "initial_capital": m.get("initial_capital", 0),
                    "final_capital": m.get("final_capital", 0),
                    "sharpe_ratio": m.get("sharpe_ratio", 0),
                    "sortino_ratio": m.get("sortino_ratio", 0),
                    "max_drawdown": m.get("max_drawdown", 0),
                    "cagr": m.get("cagr", 0),
                    "win_rate": m.get("win_rate", 0),
                    "total_trades": int(m.get("total_trades", 0)),
                    "parameters": json.dumps(result.parameters),
                    "equity_curve": equity_json,
                },
            ).fetchone()
            conn.commit()
            backtest_id = row[0]
            logger.info(
                "Saved backtest result id=%d for strategy '%s' on %s.",
                backtest_id,
                result.strategy_name,
                result.ticker,
            )
            return backtest_id

    def load_from_db(self, backtest_id: int) -> dict[str, Any]:
        """Load a previously stored backtest result.

        Args:
            backtest_id: Primary key in ``backtest_results``.

        Returns:
            Dictionary with all columns.

        Raises:
            ValueError: If the id is not found.
        """
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM backtest_results WHERE id = :id"),
                {"id": backtest_id},
            ).mappings().fetchone()

        if row is None:
            raise ValueError(f"Backtest result id={backtest_id} not found.")

        return dict(row)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(self, results: list[BacktestResult]) -> pd.DataFrame:
        """Compare metrics across multiple backtest results.

        Args:
            results: List of :class:`BacktestResult` to compare.

        Returns:
            A DataFrame with one row per result and metric columns.
        """
        records = []
        for r in results:
            record = {"strategy": r.strategy_name, "ticker": r.ticker}
            record.update(
                {
                    k: v
                    for k, v in r.metrics.items()
                    if isinstance(v, (int, float))
                }
            )
            records.append(record)

        df = pd.DataFrame(records)
        if "strategy" in df.columns:
            df = df.set_index("strategy")
        return df
