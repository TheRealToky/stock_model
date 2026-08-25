"""Research evaluation library for 1-minute long/flat strategy studies.

This module is the *trusted* evaluation layer for the 1-minute research in
this repo. It deliberately re-derives a few things the stock
:mod:`financials.backtesting` stack gets wrong for 1-minute bars:

* **Annualization.** ``BacktestEngine`` calls ``compute_all_metrics`` without
  ``periods`` so every Sharpe/Sortino/CAGR is annualized at 252 (daily). For
  US-regular-hours 1-minute bars the correct factor is ``390 * 252 = 98_280``.
  Using 252 overstates the Sharpe ratio by ``sqrt(98280/252) ~= 19.7x``.

* **Apples-to-apples buy & hold.** Buy & hold is simulated through the *same*
  cost model and the *same* execution convention (trade at next bar's open,
  mark to market at close) as every strategy, so the comparison is fair.

* **Alpha / beta vs buy & hold.** Computed here against the asset's own
  close-to-close returns (the real benchmark), not against the strategy's own
  equity returns (which is what the stock metrics module accidentally does).

Execution & accounting convention (matches ``BacktestEngine._simulate_trades``):

* Signals are the BaseStrategy action encoding ``{1: enter-if-flat,
  -1: exit-if-long, 0: hold}``. They are **shifted by one bar** before
  simulating, so the action taken at bar ``i`` uses only information available
  through bar ``i-1`` and executes at bar ``i``'s open.
* Long/flat only, single instrument. Entry buys at ``open*(1+slippage)``,
  exit sells at ``open*(1-slippage)``; proportional ``commission`` is charged
  on each leg's notional.
* The vectorized simulator works in returns space with fractional shares. It
  is validated against the whole-share dollar engine in
  ``research/scripts/p0_validate_engine.py`` (terminal-equity agreement to a
  few bp, identical trade timing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# US regular trading hours: 390 minutes/day * 252 trading days/year.
PERIODS_1MIN: int = 390 * 252  # 98_280
RF_DEFAULT: float = 0.04

# Lab default costs (financials.config.settings.BacktestConfig).
DEFAULT_COMMISSION: float = 0.001   # 10 bps per leg
DEFAULT_SLIPPAGE: float = 0.0005    # 5 bps per leg
INITIAL_CAPITAL: float = 100_000.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ohlcv(
    loader,
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    columns: list[str] | None = None,
    drop_last_partial_day: bool = True,
) -> pd.DataFrame:
    """Load 1-minute OHLCV (+ optional feature columns) for one ticker.

    Returns a DataFrame indexed by tz-aware UTC ``timestamp`` (sorted, unique),
    float64 OHLCV. The known-partial final calendar day in the store
    (2026-04-30, truncated mid-session) is dropped by default so it cannot
    contaminate any window that ends at the dataset's edge.
    """
    base_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    if columns:
        want = list(dict.fromkeys(base_cols + list(columns)))
    else:
        want = base_cols
    df = loader.load_pandas(symbols=[ticker], start=start, end=end, columns=want)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = (
        df.drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = df[c].astype(np.float64)
    if drop_last_partial_day and len(df):
        last_day = df.index[-1].normalize()
        # Only trim if that final day is the known truncated edge.
        if last_day.date().isoformat() >= "2026-04-30":
            df = df[df.index < last_day]
    return df


# ---------------------------------------------------------------------------
# Signal -> position
# ---------------------------------------------------------------------------

def action_to_position(action: pd.Series) -> pd.Series:
    """Convert BaseStrategy action signals to a held-through-close position.

    ``{1 -> long, -1 -> flat, 0/NaN -> carry forward}`` then forward-fill.
    This reproduces the engine's stateful rule exactly: a ``1`` while already
    long is a no-op, a ``-1`` while already flat is a no-op.

    ``action`` must ALREADY be shifted by one bar (decision from the prior bar).
    The returned position at bar ``i`` is what we hold through the close of
    bar ``i``.
    """
    mapped = action.map({1: 1.0, -1: 0.0}).where(lambda s: s.notna(), np.nan)
    pos = mapped.ffill().fillna(0.0)
    return pos.astype(np.float64)


# ---------------------------------------------------------------------------
# Vectorized long/flat simulator (returns space, fractional shares)
# ---------------------------------------------------------------------------

def _sim_core(
    o: np.ndarray,
    c: np.ndarray,
    pos: np.ndarray,
    commission: float,
    slippage: float,
    initial_capital: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Core mark-to-market given a held-through-close position path.

    ``pos[i]`` is the position carried through the close of bar ``i`` (already
    lagged / shifted by the caller so it uses no future information). A
    transition into/out of the position at bar ``i`` executes at bar ``i``'s
    open with slippage and commission. Returns (equity, ret, is_entry, is_exit).
    """
    n = len(o)
    prev = np.empty(n, dtype=np.float64)
    prev[0] = 0.0
    prev[1:] = pos[:-1]

    c_prev = np.empty(n, dtype=np.float64)
    c_prev[0] = c[0]
    c_prev[1:] = c[:-1]

    buy = (1.0 + slippage) * (1.0 + commission)
    sell = (1.0 - slippage) * (1.0 - commission)

    is_entry = (prev == 0) & (pos == 1)
    is_exit = (prev == 1) & (pos == 0)
    is_hold = (prev == 1) & (pos == 1)

    m = np.ones(n, dtype=np.float64)
    m = np.where(is_entry, c / (o * buy), m)        # buy at open, mark at close
    m = np.where(is_hold, c / c_prev, m)            # close-to-close
    m = np.where(is_exit, (o * sell) / c_prev, m)   # sell at open, flat at close

    equity = initial_capital * np.cumprod(m)
    ret = m - 1.0
    return equity, ret, is_entry, is_exit


def _pack(df, equity, ret, pos, is_entry, is_exit, commission, slippage, initial_capital):
    return {
        "equity": pd.Series(equity, index=df.index, name="equity"),
        "ret": pd.Series(ret, index=df.index, name="ret"),
        "pos": pd.Series(pos, index=df.index, name="pos"),
        "n_entries": int(is_entry.sum()),
        "n_exits": int(is_exit.sum()),
        "commission": commission,
        "slippage": slippage,
        "initial_capital": initial_capital,
    }


def simulate_position(
    df: pd.DataFrame,
    desired_pos: pd.Series,
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    initial_capital: float = INITIAL_CAPITAL,
    shift: int = 1,
) -> dict[str, Any]:
    """Simulate from a desired-position series (0/1) decided at each bar.

    ``desired_pos[i]`` is the position the rule WANTS using information through
    bar ``i``; it is shifted by ``shift`` bars (default 1, engine-consistent)
    so the position actually held at bar ``i`` was decided at ``i-shift`` and
    executes at bar ``i``'s open. This enforces the no-look-ahead discipline
    uniformly for clock-based and feature-based overlays alike.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("df must be sorted by timestamp")
    o = df["open"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    pos = (
        desired_pos.reindex(df.index).shift(shift).fillna(0.0)
        .clip(0.0, 1.0).to_numpy(np.float64)
    )
    equity, ret, is_entry, is_exit = _sim_core(o, c, pos, commission, slippage, initial_capital)
    return _pack(df, equity, ret, pos, is_entry, is_exit, commission, slippage, initial_capital)


def simulate(
    df: pd.DataFrame,
    action: pd.Series,
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict[str, Any]:
    """Simulate a long/flat strategy from BaseStrategy action signals.

    Mirrors ``BacktestEngine``: shift the action by one bar, derive the
    stateful position path, then mark to market.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("df must be sorted by timestamp")
    o = df["open"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    action_shifted = action.reindex(df.index).shift(1).fillna(0)
    pos = action_to_position(action_shifted).to_numpy(np.float64)
    equity, ret, is_entry, is_exit = _sim_core(o, c, pos, commission, slippage, initial_capital)
    return _pack(df, equity, ret, pos, is_entry, is_exit, commission, slippage, initial_capital)


def buy_and_hold_action(df: pd.DataFrame) -> pd.Series:
    """Action series that enters long on the first bar and never exits."""
    a = pd.Series(0, index=df.index, dtype=int)
    a.iloc[0] = 1
    return a


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _sharpe(ret: np.ndarray, rf: float, periods: int) -> float:
    if len(ret) < 2:
        return 0.0
    excess = ret - rf / periods
    sd = np.std(excess, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(periods))


def _sortino(ret: np.ndarray, rf: float, periods: int) -> float:
    if len(ret) < 2:
        return 0.0
    excess = ret - rf / periods
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    dd = np.sqrt(np.mean(downside ** 2))
    if dd == 0:
        return 0.0
    return float(np.mean(excess) / dd * np.sqrt(periods))


def summarize_returns(ret: np.ndarray, periods: int = PERIODS_1MIN, rf: float = RF_DEFAULT) -> dict[str, float]:
    """Sharpe / Sortino / maxDD / total return from a per-bar return series.

    Useful for evaluating a sub-window (e.g. one calendar year) of an already
    simulated strategy without re-running the simulator.
    """
    ret = np.asarray(ret, dtype=np.float64)
    ret = ret[~np.isnan(ret)]
    if len(ret) < 2:
        return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
                "total_return": 0.0, "ann_return": 0.0, "n": len(ret)}
    equity = np.cumprod(1.0 + ret)
    return {
        "sharpe": _sharpe(ret, rf, periods),
        "sortino": _sortino(ret, rf, periods),
        "max_drawdown": _max_drawdown(equity),
        "total_return": float(equity[-1] - 1.0),
        "ann_return": float(np.mean(ret) * periods),
        "n": int(len(ret)),
    }


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    run_max = np.maximum.accumulate(equity)
    dd = (run_max - equity) / np.where(run_max == 0, 1.0, run_max)
    return float(np.max(dd))


def trade_stats(pos: pd.Series) -> dict[str, float]:
    """Trade-level diagnostics from a 0/1 position path.

    Counts a *trade* as one entry (0->1). Average holding period is the mean
    run length of in-market segments (in bars == minutes). Exposure is the
    fraction of bars in the market. ``turnover_per_day`` counts position
    changes (entries+exits) per trading day -- a direct cost driver.
    """
    p = pos.to_numpy(np.float64)
    n = len(p)
    prev = np.empty(n)
    prev[0] = 0.0
    prev[1:] = p[:-1]
    entries = (prev == 0) & (p == 1)
    changes = p != prev
    n_entries = int(entries.sum())
    n_days = max(pos.index.normalize().nunique(), 1)

    # in-market run lengths
    holdings: list[int] = []
    run = 0
    for v in p:
        if v == 1:
            run += 1
        elif run > 0:
            holdings.append(run)
            run = 0
    if run > 0:
        holdings.append(run)

    return {
        "n_trades": n_entries,
        "exposure": float(p.mean()),
        "avg_holding_min": float(np.mean(holdings)) if holdings else 0.0,
        "trades_per_day": n_entries / n_days,
        "turnover_per_day": int(changes.sum()) / n_days,
        "n_days": int(n_days),
    }


def alpha_beta(strat_ret: np.ndarray, asset_ret: np.ndarray, periods: int) -> tuple[float, float]:
    """Annualized alpha and beta of strategy net returns vs asset gross returns."""
    mask = ~np.isnan(strat_ret) & ~np.isnan(asset_ret)
    x = asset_ret[mask]
    y = strat_ret[mask]
    if len(x) < 2:
        return 0.0, 0.0
    var = np.var(x, ddof=1)
    if var == 0:
        return 0.0, 0.0
    cov = np.cov(y, x, ddof=1)[0, 1]
    beta = cov / var
    alpha_per_bar = np.mean(y) - beta * np.mean(x)
    return float(alpha_per_bar * periods), float(beta)


def metrics(
    sim: dict[str, Any],
    asset_ret: np.ndarray | None = None,
    periods: int = PERIODS_1MIN,
    rf: float = RF_DEFAULT,
) -> dict[str, float]:
    """Compute the full metric set for a simulate() result."""
    equity = sim["equity"].to_numpy(np.float64)
    ret = sim["ret"].to_numpy(np.float64)
    pos = sim["pos"]

    n = len(equity)
    years = n / periods if periods > 0 else 0.0
    initial = float(equity[0]) if n else 0.0
    final = float(equity[-1]) if n else 0.0
    total_return = (final / initial - 1.0) if initial > 0 else 0.0
    cagr = ((final / initial) ** (1.0 / years) - 1.0) if (years > 0 and initial > 0) else 0.0
    max_dd = _max_drawdown(equity)

    out = {
        "sharpe": _sharpe(ret, rf, periods),
        "sortino": _sortino(ret, rf, periods),
        "max_drawdown": max_dd,
        "cagr": cagr,
        "calmar": (cagr / max_dd) if max_dd > 0 else 0.0,
        "ann_return": float(np.mean(ret) * periods),
        "ann_vol": float(np.std(ret, ddof=1) * np.sqrt(periods)) if n > 1 else 0.0,
        "total_return": total_return,
        "final_capital": final,
    }
    out.update(trade_stats(pos))
    if asset_ret is not None:
        a, b = alpha_beta(ret, asset_ret, periods)
        out["alpha_ann"] = a
        out["beta"] = b
    return out


# ---------------------------------------------------------------------------
# High-level: evaluate a strategy and its buy&hold benchmark together
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    strat: dict[str, float]
    bh: dict[str, float]
    strat_sim: dict[str, Any]
    bh_sim: dict[str, Any]
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        """Flat comparison row for tabulation."""
        r = {"label": self.label}
        r.update(self.meta)
        for k, v in self.strat.items():
            r[f"strat_{k}"] = v
        for k, v in self.bh.items():
            r[f"bh_{k}"] = v
        r["sharpe_edge"] = self.strat["sharpe"] - self.bh["sharpe"]
        r["dd_ok"] = self.strat["max_drawdown"] <= self.bh["max_drawdown"] + 1e-12
        r["beats_bh"] = (r["sharpe_edge"] > 0) and bool(r["dd_ok"])
        return r


def _finish_eval(df, strat_sim, commission, slippage, initial_capital, periods, rf, label, meta):
    asset_ret = df["close"].pct_change().to_numpy(np.float64)
    bh_sim = simulate(df, buy_and_hold_action(df), commission, slippage, initial_capital)
    strat_m = metrics(strat_sim, asset_ret=asset_ret, periods=periods, rf=rf)
    bh_m = metrics(bh_sim, asset_ret=asset_ret, periods=periods, rf=rf)
    return EvalResult(
        strat=strat_m, bh=bh_m, strat_sim=strat_sim, bh_sim=bh_sim,
        label=label, meta=meta or {},
    )


def evaluate(
    df: pd.DataFrame,
    action: pd.Series,
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    initial_capital: float = INITIAL_CAPITAL,
    periods: int = PERIODS_1MIN,
    rf: float = RF_DEFAULT,
    label: str = "",
    meta: dict[str, Any] | None = None,
) -> EvalResult:
    """Run an action-signal strategy + buy&hold through the same pipeline."""
    strat_sim = simulate(df, action, commission, slippage, initial_capital)
    return _finish_eval(df, strat_sim, commission, slippage, initial_capital, periods, rf, label, meta)


def evaluate_position(
    df: pd.DataFrame,
    desired_pos: pd.Series,
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    initial_capital: float = INITIAL_CAPITAL,
    periods: int = PERIODS_1MIN,
    rf: float = RF_DEFAULT,
    label: str = "",
    meta: dict[str, Any] | None = None,
    shift: int = 1,
) -> EvalResult:
    """Run a position-overlay strategy + buy&hold through the same pipeline."""
    strat_sim = simulate_position(df, desired_pos, commission, slippage, initial_capital, shift=shift)
    return _finish_eval(df, strat_sim, commission, slippage, initial_capital, periods, rf, label, meta)


# ---------------------------------------------------------------------------
# Continuous-exposure extension (documented deviation from the long/flat engine)
# ---------------------------------------------------------------------------
#
# The stock BacktestEngine is whole-share, all-in/all-out long/flat. Volatility
# targeting needs *sizing*: a continuous exposure e_t in [0, 1] (cap at 1 => no
# leverage; the overlay only ever DE-risks vs buy & hold). We model it in
# returns space:
#
#   port_ret_t = e_t * r_t + (1 - e_t) * (rf / periods) - (comm+slip)*|e_t - e_{t-1}|
#
# where r_t is the asset close-to-close return, the uninvested fraction earns
# the risk-free rate, and rebalancing |Δe| of notional costs one leg of
# (commission + slippage). With e_t == 1 for all t this reduces to buy & hold
# (plus a one-off entry cost), which is asserted in p3 before any result is
# trusted.

def simulate_exposure(
    df: pd.DataFrame,
    exposure: pd.Series,
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    initial_capital: float = INITIAL_CAPITAL,
    rf: float = RF_DEFAULT,
    periods: int = PERIODS_1MIN,
    shift: int = 1,
) -> dict[str, Any]:
    """Simulate a continuous exposure e_t in [0,1], rebalanced when e changes.

    ``exposure[i]`` is the target decided using info through bar ``i``; it is
    shifted by ``shift`` (default 1) so it is held from the next bar. Cash earns
    ``rf``. Turnover cost = (commission+slippage)*|Δe| per bar.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("df must be sorted by timestamp")
    e = exposure.reindex(df.index).shift(shift).fillna(0.0).clip(0.0, 1.0).to_numpy(np.float64)
    r = df["close"].pct_change().fillna(0.0).to_numpy(np.float64)
    n = len(e)
    e_prev = np.empty(n, dtype=np.float64)
    e_prev[0] = 0.0
    e_prev[1:] = e[:-1]
    turnover = np.abs(e - e_prev)
    rf_pp = rf / periods
    cost = (commission + slippage) * turnover
    port_ret = e * r + (1.0 - e) * rf_pp - cost
    equity = initial_capital * np.cumprod(1.0 + port_ret)
    return {
        "equity": pd.Series(equity, index=df.index, name="equity"),
        "ret": pd.Series(port_ret, index=df.index, name="ret"),
        "pos": pd.Series(e, index=df.index, name="exposure"),
        "turnover_total": float(turnover.sum()),
        "commission": commission,
        "slippage": slippage,
        "initial_capital": initial_capital,
    }


def metrics_exposure(
    sim: dict[str, Any],
    asset_ret: np.ndarray | None = None,
    periods: int = PERIODS_1MIN,
    rf: float = RF_DEFAULT,
) -> dict[str, float]:
    """Metric set for a continuous-exposure sim (turnover-based diagnostics)."""
    equity = sim["equity"].to_numpy(np.float64)
    ret = sim["ret"].to_numpy(np.float64)
    e = sim["pos"]
    n = len(equity)
    years = n / periods if periods > 0 else 0.0
    initial = float(equity[0]) if n else 0.0
    final = float(equity[-1]) if n else 0.0
    total_return = (final / initial - 1.0) if initial > 0 else 0.0
    cagr = ((final / initial) ** (1.0 / years) - 1.0) if (years > 0 and initial > 0) else 0.0
    max_dd = _max_drawdown(equity)
    n_days = max(e.index.normalize().nunique(), 1)
    out = {
        "sharpe": _sharpe(ret, rf, periods),
        "sortino": _sortino(ret, rf, periods),
        "max_drawdown": max_dd,
        "cagr": cagr,
        "calmar": (cagr / max_dd) if max_dd > 0 else 0.0,
        "ann_return": float(np.mean(ret) * periods),
        "ann_vol": float(np.std(ret, ddof=1) * np.sqrt(periods)) if n > 1 else 0.0,
        "total_return": total_return,
        "final_capital": final,
        "exposure": float(e.mean()),
        "turnover_total": sim.get("turnover_total", 0.0),
        "turnover_per_day": sim.get("turnover_total", 0.0) / n_days,
        "n_days": int(n_days),
        # not meaningful for continuous exposure but kept for row() compatibility
        "n_trades": 0, "avg_holding_min": 0.0, "trades_per_day": 0.0,
    }
    if asset_ret is not None:
        a, b = alpha_beta(ret, asset_ret, periods)
        out["alpha_ann"] = a
        out["beta"] = b
    return out


def evaluate_exposure(
    df: pd.DataFrame,
    exposure: pd.Series,
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    initial_capital: float = INITIAL_CAPITAL,
    periods: int = PERIODS_1MIN,
    rf: float = RF_DEFAULT,
    label: str = "",
    meta: dict[str, Any] | None = None,
    shift: int = 1,
) -> EvalResult:
    """Continuous-exposure overlay vs buy & hold (e==1) through one pipeline."""
    asset_ret = df["close"].pct_change().to_numpy(np.float64)
    strat_sim = simulate_exposure(df, exposure, commission, slippage, initial_capital, rf, periods, shift)
    bh_exposure = pd.Series(1.0, index=df.index)
    bh_sim = simulate_exposure(df, bh_exposure, commission, slippage, initial_capital, rf, periods, shift)
    strat_m = metrics_exposure(strat_sim, asset_ret=asset_ret, periods=periods, rf=rf)
    bh_m = metrics_exposure(bh_sim, asset_ret=asset_ret, periods=periods, rf=rf)
    return EvalResult(strat=strat_m, bh=bh_m, strat_sim=strat_sim, bh_sim=bh_sim,
                      label=label, meta=meta or {})
