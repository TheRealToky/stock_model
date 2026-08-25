"""Phase 0.3 - validate the vectorized reslib simulator vs the canonical
event-driven BacktestEngine, and validate metric annualization.

We trust the engine's *accounting* (whole-share dollar simulation) and only
distrust its *annualization*. So:
  1. Run real strategies through BOTH the engine and reslib.simulate.
  2. Confirm terminal equity, trade timing, and the equity-curve path agree to
     a few basis points (the only difference should be whole-share rounding).
  3. Confirm reslib.metrics() on the engine's own equity curve reproduces the
     engine metrics when both use periods=252, then show the corrected
     periods=98280 number.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from financials.etl_pipeline.load.reader import MLDataLoader
from financials.backtesting.engine import BacktestEngine
from financials.backtesting.metrics import compute_all_metrics
from strategies import EMACrossover, MeanReversion, Momentum

from research.lib import reslib

pd.set_option("display.width", 200)

loader = MLDataLoader("data/feature_store")

TICKER = "AAPL"
START, END = "2021-01-04", "2021-06-30"
COMM, SLIP = reslib.DEFAULT_COMMISSION, reslib.DEFAULT_SLIPPAGE

df = reslib.load_ohlcv(loader, TICKER, START, END)
print(f"{TICKER} {START}..{END}: {len(df)} bars, "
      f"{df.index[0]} .. {df.index[-1]}")

strategies = {
    "ema_crossover": EMACrossover(fast_period=12, slow_period=26),
    "mean_reversion": MeanReversion(lookback=20, entry_std=2.0, exit_std=0.5),
    "momentum": Momentum(lookback=20, top_pct=0.2),
}

print("\n" + "=" * 78)
print("ENGINE vs RESLIB  (terminal equity, #trades, equity-curve agreement)")
print("=" * 78)
engine = BacktestEngine(initial_capital=reslib.INITIAL_CAPITAL, commission=COMM, slippage=SLIP)

for name, strat in strategies.items():
    # Canonical engine
    res = engine.run(strat, df, ticker=TICKER)
    eng_equity = np.asarray(res.equity_curve, dtype=np.float64)
    eng_trades = len(res.trades)

    # reslib vectorized
    action = strat.generate_signals(df)
    sim = reslib.simulate(df, action, commission=COMM, slippage=SLIP)
    res_equity = sim["equity"].to_numpy(np.float64)

    rel_term = abs(res_equity[-1] - eng_equity[-1]) / eng_equity[-1]
    # path agreement
    corr = np.corrcoef(res_equity, eng_equity)[0, 1]
    max_rel = np.max(np.abs(res_equity - eng_equity) / eng_equity)

    print(f"\n{name}:")
    print(f"  engine  final={eng_equity[-1]:,.2f}  trades={eng_trades}")
    print(f"  reslib  final={res_equity[-1]:,.2f}  entries={sim['n_entries']}")
    print(f"  terminal rel.diff={rel_term:.2e}  path corr={corr:.6f}  max rel.diff={max_rel:.2e}")

print("\n" + "=" * 78)
print("METRIC ANNUALIZATION CHECK (mean_reversion equity curve)")
print("=" * 78)
strat = strategies["mean_reversion"]
res = engine.run(strat, df, ticker=TICKER)
eng_equity = np.asarray(res.equity_curve, dtype=np.float64)

# reslib metrics on the engine's own equity, periods=252 vs 98280
sim_like = {"equity": pd.Series(eng_equity, index=df.index),
            "ret": pd.Series(np.r_[0.0, np.diff(eng_equity) / eng_equity[:-1]], index=df.index),
            "pos": reslib.action_to_position(strat.generate_signals(df).shift(1).fillna(0))}
m252 = reslib.metrics(sim_like, periods=252)
m98280 = reslib.metrics(sim_like, periods=reslib.PERIODS_1MIN)
eng_metrics = res.metrics  # engine's own (periods=252 default)

print(f"  engine.metrics sharpe (periods=252 default) = {eng_metrics['sharpe_ratio']:.4f}")
print(f"  reslib  sharpe @252                         = {m252['sharpe']:.4f}")
print(f"  reslib  sharpe @98280 (CORRECT for 1-min)   = {m98280['sharpe']:.4f}")
print(f"  ratio 252/98280 sharpe = {m252['sharpe']/m98280['sharpe']:.3f} "
      f"(expect sqrt(98280/252)={np.sqrt(98280/252):.3f})")

print("\n" + "=" * 78)
print("SPEED BENCHMARK (full-history AAPL: engine loop vs vectorized)")
print("=" * 78)
df_full = reslib.load_ohlcv(loader, TICKER, "2021-01-04", "2026-04-29")
print(f"  full history bars: {len(df_full):,}")
strat = EMACrossover(12, 26)

t0 = time.time()
_ = engine.run(strat, df_full, ticker=TICKER)
t_engine = time.time() - t0

action = strat.generate_signals(df_full)
t0 = time.time()
_ = reslib.simulate(df_full, action, commission=COMM, slippage=SLIP)
t_vec = time.time() - t0
print(f"  engine.run      : {t_engine:.3f}s")
print(f"  reslib.simulate : {t_vec:.3f}s   (speedup {t_engine/max(t_vec,1e-9):.1f}x)")
