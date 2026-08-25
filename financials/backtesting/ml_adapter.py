"""Turn model predictions into positions, and positions into net-of-cost P&L.

This is the missing link between the model layer and the backtester. Until now
a model's report card stopped at accuracy and ROC-AUC, which say nothing about
whether it makes money: the shipped LSTM posts 0.533 accuracy / 0.536 AUC and
had **never been run through a backtest**, so nobody could say whether that
edge survives a single basis point of friction. It usually does not. A
classifier at 53% accuracy that trades 22 times a day is a losing strategy at
any realistic cost, and the classification metrics look identical either way.

The pipeline here is deliberately short and auditable::

    proba -> position -> equity (net of costs) -> annualised metrics

Three rules it enforces, all of which are easy to get wrong by hand:

* **Positions are shifted before they earn.** A probability computed from bar
  *t*'s features can only be traded at *t+1*. ``shift=1`` is the default and
  turning it off requires saying so explicitly.
* **Costs are charged on turnover, not on trades.** Every change in exposure
  pays ``(commission + slippage) * |delta position|``, so a strategy that
  flickers in and out is penalised the way a real one would be.
* **Annualisation follows the bar interval.** Metrics route through
  :func:`~financials.backtesting.metrics.compute_all_metrics` with an explicit
  ``interval``/``periods``, so a 1-minute result is never reported at the daily
  252 convention.

The simulator works in *returns space* with fractional position sizing, which
is a deliberate deviation from :class:`~financials.backtesting.engine.
BacktestEngine`'s whole-share, all-in/all-out model. Probability-driven sizing
needs continuous exposure; whole shares would quantise it away.

It reproduces :func:`~research.lib.reslib.simulate_exposure` bit for bit (see
``tests/unit/financials/test_ml_adapter.py``), and that function was in turn
validated against the dollar engine at an equity correlation of 1.000000 -- so
results here chain back to the engine rather than forming a third, unchecked
convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from financials.backtesting.constants import resolve_periods
from financials.backtesting.metrics import compute_all_metrics

__all__ = [
    "SIZING_MODES",
    "probabilities_to_positions",
    "simulate_positions",
    "MLBacktestResult",
    "backtest_predictions",
    "cost_sweep",
    "tune_threshold",
]

#: How a probability is mapped onto an exposure in ``[0, 1]`` (or ``[-1, 1]``).
SIZING_MODES: tuple[str, ...] = ("binary", "proportional", "confidence")

#: Cost grid mirroring ``research.config.COST_GRID``: (commission, slippage,
#: label, round-trip bps). Retail-tier friction is the pessimistic end.
DEFAULT_COST_GRID: tuple[tuple[float, float, str, float], ...] = (
    (0.0,     0.0,     "frictionless",     0.0),
    (0.00005, 0.00005, "ultra_low_1bp",    2.0),
    (0.0001,  0.0001,  "low_2bp",          4.0),
    (0.00025, 0.00025, "mid_5bp",         10.0),
    (0.0005,  0.0005,  "high_10bp",       20.0),
    (0.001,   0.0005,  "lab_default_15bp", 30.0),
)


def probabilities_to_positions(
    proba: np.ndarray | pd.Series,
    *,
    threshold: float = 0.5,
    sizing: str = "binary",
    allow_short: bool = False,
    max_position: float = 1.0,
    index: pd.Index | None = None,
) -> pd.Series:
    """Map predicted probabilities onto desired exposures.

    Args:
        proba: Probability of the positive class, one per bar, in ``[0, 1]``.
        threshold: Probability above which the model wants to be long (and,
            when *allow_short*, below ``1 - threshold`` to be short).
        sizing: How conviction becomes size.

            * ``"binary"`` -- full position once past the threshold. Matches
              what ``research/scripts/p4_ml_route.py`` did.
            * ``"proportional"`` -- exposure scales linearly with how far the
              probability sits past the threshold, reaching *max_position* at
              certainty. Trades less on marginal signals.
            * ``"confidence"`` -- exposure is ``2 * |proba - 0.5|``, ignoring
              *threshold* entirely; size tracks conviction in either direction.
        allow_short: Permit negative exposure. When ``False`` the strategy is
            long/flat, matching the rest of the lab.
        max_position: Cap on absolute exposure. ``1.0`` means no leverage.
        index: Index for the result when *proba* is a bare array.

    Returns:
        Float Series of desired exposures, ``NaN`` where *proba* is ``NaN``.

    Raises:
        ValueError: If *sizing* is unknown, *threshold* is outside ``(0, 1)``,
            or *max_position* is not positive.
    """
    if sizing not in SIZING_MODES:
        raise ValueError(f"sizing must be one of {SIZING_MODES}, got {sizing!r}.")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}.")
    if max_position <= 0:
        raise ValueError(f"max_position must be > 0, got {max_position}.")

    if isinstance(proba, pd.Series):
        idx = proba.index
        p = proba.to_numpy(np.float64)
    else:
        p = np.asarray(proba, dtype=np.float64)
        idx = index if index is not None else pd.RangeIndex(len(p))

    if len(idx) != len(p):
        raise ValueError(f"index length {len(idx)} != probability length {len(p)}.")

    valid = ~np.isnan(p)
    pos = np.zeros(len(p), dtype=np.float64)

    if sizing == "binary":
        pos[valid & (p > threshold)] = max_position
        if allow_short:
            pos[valid & (p < 1.0 - threshold)] = -max_position

    elif sizing == "proportional":
        # Distance past the threshold, normalised by the room left above it.
        room = max(1.0 - threshold, 1e-12)
        long_leg = np.clip((p - threshold) / room, 0.0, 1.0) * max_position
        pos = np.where(valid & (p > threshold), long_leg, 0.0)
        if allow_short:
            short_room = max(threshold, 1e-12)
            short_leg = np.clip((1.0 - threshold - p) / short_room, 0.0, 1.0) * max_position
            pos = np.where(valid & (p < 1.0 - threshold), -short_leg, pos)

    else:  # confidence
        conviction = np.clip(2.0 * np.abs(p - 0.5), 0.0, 1.0) * max_position
        if allow_short:
            pos = np.where(valid, np.sign(p - 0.5) * conviction, 0.0)
        else:
            pos = np.where(valid & (p > 0.5), conviction, 0.0)

    out = pd.Series(pos, index=idx, dtype=np.float64, name="position")
    return out.where(pd.Series(valid, index=idx), np.nan)


def simulate_positions(
    bars: pd.DataFrame,
    positions: pd.Series,
    *,
    commission: float = 0.001,
    slippage: float = 0.0005,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.04,
    periods: int | None = None,
    interval: str | None = None,
    price_col: str = "close",
    shift: int = 1,
) -> dict[str, Any]:
    """Simulate a continuous-exposure strategy in returns space.

    Per bar::

        port_ret = pos * asset_ret + (1 - pos) * rf_per_bar
                   - (commission + slippage) * |pos - pos_prev|

    The idle-cash term matters more than it looks. A thresholded model sits
    flat much of the time, and dropping that term would charge it the full
    opportunity cost of being out of the market while the buy-and-hold
    benchmark is credited none -- quietly biasing every comparison against the
    strategy. This mirrors :func:`~research.lib.reslib.simulate_exposure`.

    Args:
        bars: Price data with a tz-aware ``DatetimeIndex``.
        positions: Desired exposure per bar, aligned to *bars*.
        commission: Proportional commission per leg.
        slippage: Proportional slippage per leg.
        initial_capital: Starting equity.
        risk_free_rate: Annualised rate earned on the uninvested fraction.
        periods: Bars per year, used to de-annualise *risk_free_rate*.
        interval: Bar interval, as an alternative to *periods*. Inferred from
            the index when both are omitted.
        price_col: Column holding the price the strategy marks against.
        shift: Bars to delay a position before it earns. ``1`` (the default)
            means a signal formed on bar *t* is traded into at *t+1*, which is
            the only setting free of look-ahead. ``0`` is available for
            diagnostics and will silently flatter results.

    Returns:
        Dict with ``equity``, ``ret``, ``pos``, ``turnover`` and ``asset_ret``.

    Raises:
        ValueError: If inputs misalign, costs are negative, or *shift* < 0.
    """
    if shift < 0:
        raise ValueError(f"shift must be >= 0, got {shift}.")
    if commission < 0 or slippage < 0:
        raise ValueError("commission and slippage must be non-negative.")
    if price_col not in bars.columns:
        raise KeyError(f"{price_col!r} not in bars columns: {list(bars.columns)}")
    if not positions.index.equals(bars.index):
        raise ValueError("positions must share the bars index.")

    ann = resolve_periods(interval=interval, periods=periods, index=bars.index)

    price = bars[price_col].astype(np.float64)
    asset_ret = price.pct_change().fillna(0.0).to_numpy(np.float64)

    held = positions.shift(shift).fillna(0.0).to_numpy(np.float64)
    prev = np.concatenate([[0.0], held[:-1]])
    turnover = np.abs(held - prev)

    cost_rate = commission + slippage
    rf_per_bar = risk_free_rate / ann
    ret = held * asset_ret + (1.0 - held) * rf_per_bar - cost_rate * turnover
    equity = initial_capital * np.cumprod(1.0 + ret)

    return {
        "equity": pd.Series(equity, index=bars.index, name="equity"),
        "ret": pd.Series(ret, index=bars.index, name="ret"),
        "pos": pd.Series(held, index=bars.index, name="pos"),
        "turnover": pd.Series(turnover, index=bars.index, name="turnover"),
        "asset_ret": pd.Series(asset_ret, index=bars.index, name="asset_ret"),
    }


@dataclass
class MLBacktestResult:
    """A model's strategy performance beside its buy-and-hold benchmark.

    Attributes:
        strategy: Annualised metrics for the model-driven strategy.
        benchmark: The same metrics for buying and holding the asset.
        simulation: Raw series from :func:`simulate_positions`.
        label: Human-readable identifier.
        meta: Free-form context (threshold, cost label, interval, ...).
    """

    strategy: dict[str, float]
    benchmark: dict[str, float]
    simulation: dict[str, Any]
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def sharpe_edge(self) -> float:
        """Strategy Sharpe minus benchmark Sharpe. The number that matters."""
        return self.strategy["sharpe_ratio"] - self.benchmark["sharpe_ratio"]

    @property
    def beats_benchmark(self) -> bool:
        """True only when the strategy wins on Sharpe *and* on drawdown."""
        return bool(
            self.sharpe_edge > 0
            and self.strategy["max_drawdown"] <= self.benchmark["max_drawdown"] + 1e-12
        )

    def row(self) -> dict[str, Any]:
        """Flatten to one tabulation row."""
        out: dict[str, Any] = {"label": self.label, **self.meta}
        for k, v in self.strategy.items():
            if not isinstance(v, (list, str)):
                out[f"strat_{k}"] = v
        for k, v in self.benchmark.items():
            if not isinstance(v, (list, str)):
                out[f"bh_{k}"] = v
        out["sharpe_edge"] = self.sharpe_edge
        out["beats_benchmark"] = self.beats_benchmark
        return out


def _trade_stats(pos: pd.Series, turnover: pd.Series, index: pd.Index) -> dict[str, float]:
    """Exposure and turnover diagnostics, per trading day."""
    p = pos.to_numpy(np.float64)
    n_days = max(len(pd.DatetimeIndex(index).normalize().unique()), 1)
    entries = int(np.sum((p != 0) & (np.concatenate([[0.0], p[:-1]]) == 0)))
    return {
        "exposure": float(np.mean(np.abs(p))),
        "n_entries": entries,
        "trades_per_day": entries / n_days,
        "turnover_per_day": float(turnover.sum()) / n_days,
        "n_days": float(n_days),
    }


def backtest_predictions(
    bars: pd.DataFrame,
    proba: np.ndarray | pd.Series,
    *,
    interval: str | None = None,
    periods: int | None = None,
    threshold: float = 0.5,
    sizing: str = "binary",
    allow_short: bool = False,
    max_position: float = 1.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.04,
    price_col: str = "close",
    shift: int = 1,
    label: str = "",
    meta: dict[str, Any] | None = None,
) -> MLBacktestResult:
    """Score model probabilities as a trading strategy, net of costs.

    The one call that answers "does this model make money?". Everything is
    annualised with the bar interval's own periods-per-year, and the buy-and-
    hold benchmark is simulated through the identical code path so the
    comparison is apples to apples.

    Args:
        bars: Price data with a tz-aware ``DatetimeIndex``.
        proba: Positive-class probability per bar, aligned to *bars*. ``NaN``
            is treated as flat.
        interval: Bar interval (``"1min"``, ``"1d"``, ...) driving
            annualisation. Inferred from the index when omitted.
        periods: Explicit bars-per-year, overriding *interval*.
        threshold: Probability above which to go long.
        sizing: One of :data:`SIZING_MODES`.
        allow_short: Permit negative exposure.
        max_position: Cap on absolute exposure.
        commission: Proportional commission per leg.
        slippage: Proportional slippage per leg.
        initial_capital: Starting equity.
        risk_free_rate: Annualised risk-free rate for Sharpe/Sortino.
        price_col: Price column to mark against.
        shift: Bars between signal and fill; keep at 1.
        label: Identifier carried into the result.
        meta: Extra context carried into :meth:`MLBacktestResult.row`.

    Returns:
        An :class:`MLBacktestResult`.

    Raises:
        ValueError: If the interval cannot be resolved or inputs misalign.
    """
    ann = resolve_periods(interval=interval, periods=periods, index=bars.index)

    if isinstance(proba, pd.Series):
        if not proba.index.equals(bars.index):
            raise ValueError("proba must share the bars index.")
        p = proba
    else:
        p = pd.Series(np.asarray(proba, dtype=np.float64), index=bars.index)

    positions = probabilities_to_positions(
        p,
        threshold=threshold,
        sizing=sizing,
        allow_short=allow_short,
        max_position=max_position,
    ).fillna(0.0)

    sim_kwargs = dict(
        commission=commission, slippage=slippage,
        initial_capital=initial_capital, risk_free_rate=risk_free_rate,
        periods=ann, price_col=price_col, shift=shift,
    )
    sim = simulate_positions(bars, positions, **sim_kwargs)

    # Benchmark: fully invested throughout, charged one entry leg, through the
    # exact same simulator so any modelling choice cancels in the comparison.
    bh_sim = simulate_positions(bars, pd.Series(1.0, index=bars.index), **sim_kwargs)

    bench_ret = sim["asset_ret"].to_numpy(np.float64)
    strat_metrics = compute_all_metrics(
        sim["equity"], trades=[], risk_free_rate=risk_free_rate,
        periods=ann, benchmark_returns=bench_ret,
    )
    bh_metrics = compute_all_metrics(
        bh_sim["equity"], trades=[], risk_free_rate=risk_free_rate,
        periods=ann, benchmark_returns=bench_ret,
    )
    strat_metrics.update(_trade_stats(sim["pos"], sim["turnover"], bars.index))
    bh_metrics.update(_trade_stats(bh_sim["pos"], bh_sim["turnover"], bars.index))

    return MLBacktestResult(
        strategy=strat_metrics,
        benchmark=bh_metrics,
        simulation=sim,
        label=label or f"thr={threshold:.2f}",
        meta={"threshold": threshold, "sizing": sizing, "periods_per_year": ann,
              **(meta or {})},
    )


def cost_sweep(
    bars: pd.DataFrame,
    proba: np.ndarray | pd.Series,
    *,
    interval: str | None = None,
    thresholds: Sequence[float] = (0.5,),
    cost_grid: Iterable[tuple[float, float, str, float]] = DEFAULT_COST_GRID,
    **kwargs: Any,
) -> pd.DataFrame:
    """Score predictions across a grid of thresholds and cost assumptions.

    Friction is usually the deciding variable, not the model. Reporting a
    single cost point invites the reader to assume the flattering one, so this
    reports the whole curve -- including frictionless, which shows how much of
    any edge is theoretical.

    Args:
        bars: Price data.
        proba: Positive-class probabilities.
        interval: Bar interval for annualisation.
        thresholds: Probability thresholds to evaluate.
        cost_grid: Tuples of ``(commission, slippage, label, round_trip_bps)``.
        **kwargs: Forwarded to :func:`backtest_predictions`.

    Returns:
        One row per (threshold, cost) pair, sorted by strategy Sharpe
        descending.
    """
    rows: list[dict[str, Any]] = []
    for thr in thresholds:
        for commission, slippage, cost_label, rt_bps in cost_grid:
            result = backtest_predictions(
                bars, proba,
                interval=interval, threshold=thr,
                commission=commission, slippage=slippage,
                label=f"thr={thr:.2f}/{cost_label}",
                meta={"cost_label": cost_label, "round_trip_bps": rt_bps},
                **kwargs,
            )
            rows.append(result.row())
    return pd.DataFrame(rows).sort_values("strat_sharpe_ratio", ascending=False)


def tune_threshold(
    bars: pd.DataFrame,
    proba: np.ndarray | pd.Series,
    *,
    interval: str | None = None,
    candidates: Sequence[float] = (0.50, 0.52, 0.55, 0.58, 0.60),
    objective: str = "sharpe_edge",
    **kwargs: Any,
) -> tuple[float, pd.DataFrame]:
    """Pick the threshold that maximises *objective* on the data given.

    Run this on a **validation** split only, then freeze the winner before
    touching the test window. Tuning and reporting on the same data is the
    fastest way to manufacture an edge that does not exist -- and note that
    every candidate here counts as a trial for
    :func:`~research.lib.validation.deflated_sharpe_ratio`.

    Args:
        bars: Validation-window price data.
        proba: Validation-window probabilities.
        interval: Bar interval for annualisation.
        candidates: Thresholds to try.
        objective: ``"sharpe_edge"`` (vs buy-and-hold) or ``"sharpe"``.
        **kwargs: Forwarded to :func:`backtest_predictions`.

    Returns:
        ``(best_threshold, table)`` where *table* holds every candidate's row.

    Raises:
        ValueError: If *objective* is unknown or *candidates* is empty.
    """
    if objective not in {"sharpe_edge", "sharpe"}:
        raise ValueError(f"objective must be 'sharpe_edge' or 'sharpe', got {objective!r}.")
    if not len(candidates):
        raise ValueError("candidates must not be empty.")

    rows = []
    for thr in candidates:
        result = backtest_predictions(bars, proba, interval=interval, threshold=thr, **kwargs)
        rows.append(result.row())

    table = pd.DataFrame(rows)
    key = "sharpe_edge" if objective == "sharpe_edge" else "strat_sharpe_ratio"
    best = float(table.loc[table[key].idxmax(), "threshold"])
    return best, table
