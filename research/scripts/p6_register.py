"""Phase 6 - persist deliverables to the database.

Registers the vol-target overlay and a buy & hold benchmark in
strategy_registry, and writes backtest_results rows for the PRIMARY ticker on
both the in-sample (train+val) and the sealed out-of-sample (test) windows,
using the canonical BacktestReport.save_to_db path. Continuous-exposure metrics
are mapped onto the engine's metric field names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from financials.backtesting.engine import BacktestResult
from financials.backtesting.report import BacktestReport
from strategies.registry import StrategyRegistry
from strategies.volatility_target import VolatilityTargetOverlay, BuyAndHold
from research.lib import reslib, overlays
from research import config

COMM, SLIP = 0.0001, 0.0001
PERIODS = reslib.PERIODS_1MIN
loader_mod = __import__("financials.etl_pipeline.load.reader", fromlist=["MLDataLoader"])
loader = loader_mod.MLDataLoader("data/feature_store")


def eng_metrics(ret: np.ndarray, equity: np.ndarray, exposure: np.ndarray) -> dict:
    s = reslib.summarize_returns(ret, PERIODS, reslib.RF_DEFAULT)
    years = len(ret) / PERIODS
    cagr = (equity[-1] / equity[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    return {  # all native Python floats so psycopg2 can bind them
        "sharpe_ratio": float(s["sharpe"]), "sortino_ratio": float(s["sortino"]),
        "max_drawdown": float(s["max_drawdown"]), "cagr": float(cagr),
        "win_rate": 0.0, "total_trades": 0,
        "initial_capital": float(equity[0]), "final_capital": float(equity[-1]),
        "total_return": float(s["total_return"]), "avg_exposure": float(exposure.mean()),
    }


def make_result(strategy_name, params, ticker, idx, ret, exposure):
    equity = (reslib.INITIAL_CAPITAL * np.cumprod(1.0 + ret))
    m = eng_metrics(ret, equity, exposure)
    return BacktestResult(
        equity_curve=equity.tolist(), trades=[], metrics=m, parameters=params,
        strategy_name=strategy_name, ticker=ticker,
        start_date=str(idx[0].date()), end_date=str(idx[-1].date()),
    )


def in_range(index, start, end):
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return (index >= s) & (index < e)


# ---- compute AAPL overlay + B&H over full history, slice IS / OOS ----
tk = config.PRIMARY
df = reslib.load_ohlcv(loader, tk, config.TRAINVAL_START, config.TEST_END)
e = overlays.voltarget_exposure(df)
ov = reslib.simulate_exposure(df, e, commission=COMM, slippage=SLIP)
bh = reslib.simulate_exposure(df, pd.Series(1.0, index=df.index), commission=COMM, slippage=SLIP)

masks = {
    "IS": in_range(df.index, config.TRAINVAL_START, config.TRAINVAL_END),
    "OOS": in_range(df.index, config.TEST_START, config.TEST_END),
}

# ---- register strategies ----
reg = StrategyRegistry()
ov_strat = VolatilityTargetOverlay(span=overlays.SPAN, target_q=overlays.TARGET_Q, min_days=overlays.MIN_DAYS)
bh_strat = BuyAndHold()

# headline metrics = OOS overlay on AAPL
oos = masks["OOS"]
ov_oos_m = eng_metrics(ov["ret"].to_numpy()[oos],
                       reslib.INITIAL_CAPITAL * np.cumprod(1 + ov["ret"].to_numpy()[oos]),
                       ov["pos"].to_numpy()[oos])
sid_ov = reg.register(ov_strat, performance_metrics={
    "evaluation": "AAPL sealed test 2025-05-01..2026-04-29, 2bp round-trip, "
                  f"Sharpe annualized @ {PERIODS}",
    **ov_oos_m})
sid_bh = reg.register(bh_strat, performance_metrics={"note": "benchmark"})
print(f"registered vol_target_overlay id={sid_ov}, buy_and_hold id={sid_bh}")

# ---- write backtest_results rows ----
report = BacktestReport()
written = []
for split, mask in masks.items():
    idx = df.index[mask]
    r_ov = make_result(ov_strat.name, ov_strat.get_parameters(), tk, idx,
                       ov["ret"].to_numpy()[mask], ov["pos"].to_numpy()[mask])
    r_bh = make_result(bh_strat.name, {}, tk, idx,
                       bh["ret"].to_numpy()[mask], bh["pos"].to_numpy()[mask])
    bid_ov = report.save_to_db(r_ov, strategy_id=sid_ov)
    bid_bh = report.save_to_db(r_bh, strategy_id=sid_bh)
    written += [(split, "overlay", bid_ov, r_ov.metrics["sharpe_ratio"], r_ov.metrics["max_drawdown"]),
                (split, "bh", bid_bh, r_bh.metrics["sharpe_ratio"], r_bh.metrics["max_drawdown"])]

print("\nbacktest_results written (split, kind, id, sharpe, maxDD):")
for w in written:
    print(f"  {w[0]:4s} {w[1]:8s} id={w[2]}  sharpe={w[3]:.3f}  maxDD={w[4]:.3f}")
print("\nDONE p6")
