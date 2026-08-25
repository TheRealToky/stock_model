"""Target construction for multi-horizon research.

Every column in the feature store is backward-looking, so a label has to be
*built*, never picked off the shelf. This module builds them.

Three families, in increasing order of how much they respect trading reality:

1. **Fixed-horizon** -- ``close[i+h] / close[i] - 1``. Simple, and what the
   existing LSTM uses at ``h=1``. Its weakness is that it ignores the path: a
   label of +1 is recorded identically whether the trade drifted up calmly or
   first drew down 3% and recovered.
2. **Triple-barrier** (Lopez de Prado) -- run each observation forward until it
   hits a profit-take barrier, a stop-loss barrier, or a time limit, and label
   by which came first. Barriers are scaled by local volatility, so a label
   means the same thing in calm and turbulent regimes.
3. **Meta-labels** -- given a primary model's *side*, label whether acting on
   it would have been profitable. Lets a second model learn when to size up and
   when to stand aside, which is usually easier than predicting direction.

Session handling
----------------
Every function here is session-aware by default. US regular trading hours run
09:30-16:00 ET, which never crosses UTC midnight, so the UTC calendar date
identifies the session. A forward return spanning the overnight gap is not a
tradeable one-bar move -- it is unhedgeable gap risk that no intraday model can
act on -- so those labels are masked to NaN rather than silently learned.
``research/scripts/p4_ml_route.py`` already did this inline; this module makes
it the default everywhere.

On **daily or coarser** bars that masking would be catastrophic rather than
careful: every bar is its own session, so "does the next bar share this
session?" is false everywhere and the mask would erase the entire series.
:func:`is_intraday` detects that case and the session logic stands down, which
is what lets the same labelling code serve every horizon in
``config.HORIZONS``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "session_ids",
    "is_intraday",
    "same_session_forward",
    "session_end_positions",
    "forward_return",
    "fixed_horizon_label",
    "ewm_volatility",
    "realized_volatility",
    "TripleBarrierResult",
    "triple_barrier_labels",
    "meta_labels",
]


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def session_ids(index: pd.DatetimeIndex) -> np.ndarray:
    """Integer session id per bar, so bars can be grouped by trading day.

    Args:
        index: Tz-aware (UTC) bar timestamps.

    Returns:
        Integer array, constant within a session and increasing across them.

    Raises:
        TypeError: If *index* is not a ``DatetimeIndex``.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"session_ids requires a DatetimeIndex, got {type(index).__name__}.")
    codes, _ = pd.factorize(index.normalize(), sort=True)
    return codes.astype(np.int64)


def is_intraday(index: pd.DatetimeIndex) -> bool:
    """Whether the bars are finer than one per session.

    Session masking only makes sense for intraday bars. On daily-or-coarser
    data every bar is its own session, so "does the next bar share this
    session?" is always false -- and masking on that would discard the entire
    series rather than one bar per day. This is the guard that keeps the
    labelling functions horizon-agnostic.

    Args:
        index: Tz-aware bar timestamps.

    Returns:
        ``True`` when at least one consecutive pair shares a session.
    """
    sess = session_ids(index)
    if len(sess) < 2:
        return False
    return bool(np.any(sess[1:] == sess[:-1]))


def same_session_forward(index: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    """Mask of bars whose *horizon*-ahead bar lies in the same session.

    On daily-or-coarser bars there is no intraday structure to protect, so
    every bar with a successor is allowed through.

    Args:
        index: Tz-aware bar timestamps.
        horizon: Look-ahead in bars. Must be >= 1.

    Returns:
        Boolean array; ``False`` where the forward bar would cross a session
        boundary or run off the end of the series.

    Raises:
        ValueError: If *horizon* < 1.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    sess = session_ids(index)
    n = len(sess)
    out = np.zeros(n, dtype=bool)
    if n <= horizon:
        return out
    if not is_intraday(index):
        out[: n - horizon] = True
        return out
    out[: n - horizon] = sess[horizon:] == sess[: n - horizon]
    return out


def session_end_positions(index: pd.DatetimeIndex) -> np.ndarray:
    """Integer position of the last bar of each bar's own session.

    Used as the hard cap on a vertical barrier: an intraday label may not run
    past the close, because holding through it is a different trade.

    On daily-or-coarser bars each bar is its own session, which would cap every
    barrier at zero bars. There is no intraday close to respect there, so the
    cap is lifted to the end of the series instead.

    Args:
        index: Tz-aware bar timestamps.

    Returns:
        Integer array ``end[i]`` = position of the final bar in bar *i*'s
        session.
    """
    sess = session_ids(index)
    n = len(sess)
    end = np.empty(n, dtype=np.int64)
    if n == 0:
        return end
    if not is_intraday(index):
        end[:] = n - 1
        return end
    # Walk backwards: the last bar of a run is the session end for that run.
    last = n - 1
    for i in range(n - 1, -1, -1):
        if i < n - 1 and sess[i] != sess[i + 1]:
            last = i
        end[i] = last
    return end


# ---------------------------------------------------------------------------
# Fixed-horizon targets
# ---------------------------------------------------------------------------

def forward_return(
    df: pd.DataFrame,
    horizon: int = 1,
    *,
    price_col: str = "close",
    log: bool = False,
    mask_cross_session: bool = True,
) -> pd.Series:
    """Return realised over the next *horizon* bars.

    Args:
        df: Bar data indexed by a tz-aware ``DatetimeIndex``.
        horizon: Number of bars ahead.
        price_col: Column to measure.
        log: Return log returns instead of simple returns.
        mask_cross_session: NaN out returns that span an overnight gap.

    Returns:
        Series aligned to *df*'s index. ``NaN`` for the final *horizon* bars
        and, when masking, for each session's trailing bars.

    Raises:
        KeyError: If *price_col* is absent.
        ValueError: If *horizon* < 1.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    if price_col not in df.columns:
        raise KeyError(f"{price_col!r} not in DataFrame columns: {list(df.columns)}")

    px = df[price_col].astype(np.float64)
    fwd = px.shift(-horizon)
    out = np.log(fwd / px) if log else (fwd / px - 1.0)

    if mask_cross_session:
        ok = same_session_forward(df.index, horizon)
        out = out.where(ok, np.nan)
    return out.rename(f"fwd_return_{horizon}")


def fixed_horizon_label(
    df: pd.DataFrame,
    horizon: int = 1,
    *,
    threshold: float = 0.0,
    price_col: str = "close",
    binary: bool = True,
    mask_cross_session: bool = True,
) -> pd.Series:
    """Directional label from the *horizon*-bar forward return.

    Args:
        df: Bar data.
        horizon: Bars ahead.
        threshold: Dead zone. With ``binary=False``, moves inside
            +/- *threshold* label 0 (no trade). With ``binary=True`` the label
            is simply ``fwd_return > threshold``.
        price_col: Column to measure.
        binary: ``True`` -> ``{0, 1}``; ``False`` -> ``{-1, 0, 1}``.
        mask_cross_session: NaN out labels spanning an overnight gap.

    Returns:
        Float Series (NaN-preserving, so it survives ``dropna``).

    Raises:
        ValueError: If *threshold* is negative.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}.")

    fwd = forward_return(
        df, horizon, price_col=price_col, mask_cross_session=mask_cross_session
    )
    if binary:
        out = (fwd > threshold).astype(np.float64).where(fwd.notna(), np.nan)
    else:
        out = pd.Series(0.0, index=fwd.index, dtype=np.float64)
        out = out.mask(fwd > threshold, 1.0).mask(fwd < -threshold, -1.0)
        out = out.where(fwd.notna(), np.nan)
    return out.rename(f"label_h{horizon}")


# ---------------------------------------------------------------------------
# Volatility estimates used to scale barriers
# ---------------------------------------------------------------------------

def _intrasession_returns(close: pd.Series, mask_cross_session: bool) -> pd.Series:
    """One-bar returns, optionally with overnight gaps removed.

    Leaving gaps in inflates any volatility estimate by the gap size once per
    session, which would widen every barrier for the wrong reason.
    """
    ret = close.pct_change()
    # On daily-or-coarser bars every return IS an overnight return; masking
    # would leave nothing behind.
    if not mask_cross_session or not is_intraday(close.index):
        return ret
    sess = session_ids(close.index)
    same = np.empty(len(sess), dtype=bool)
    if len(sess):
        same[0] = False
        same[1:] = sess[1:] == sess[:-1]
    return ret.where(same, np.nan)


def ewm_volatility(
    close: pd.Series,
    span: int = 100,
    *,
    mask_cross_session: bool = True,
) -> pd.Series:
    """Exponentially-weighted volatility of one-bar returns.

    This is the natural barrier scale: a 1-sigma move means the same thing in
    a calm and a turbulent regime, so labels stay comparable across time.

    Args:
        close: Close price series with a tz-aware index.
        span: EWM span in bars.
        mask_cross_session: Exclude overnight gap returns from the estimate.

    Returns:
        Series of per-bar return standard deviations.
    """
    ret = _intrasession_returns(close, mask_cross_session)
    return ret.ewm(span=span, min_periods=max(2, span // 4)).std().rename("ewm_vol")


def realized_volatility(
    close: pd.Series,
    window: int = 100,
    *,
    mask_cross_session: bool = True,
) -> pd.Series:
    """Rolling equal-weighted volatility of one-bar returns.

    Args:
        close: Close price series.
        window: Rolling window in bars.
        mask_cross_session: Exclude overnight gap returns.

    Returns:
        Series of per-bar return standard deviations.
    """
    ret = _intrasession_returns(close, mask_cross_session)
    return ret.rolling(window, min_periods=max(2, window // 4)).std().rename("realized_vol")


# ---------------------------------------------------------------------------
# Triple-barrier method
# ---------------------------------------------------------------------------

@dataclass
class TripleBarrierResult:
    """Outcome of running observations forward to their first barrier touch.

    Attributes:
        label: ``{-1, 0, 1}`` for a plain run (which barrier was hit first;
            0 means the time limit came first and ``zero_on_vertical`` was
            set), or ``{0, 1}`` when a *side* was supplied (meta-labelling:
            did acting on that side pay?).
        ret: Realised return from entry to the touch, signed by *side* when
            one was given.
        touch_idx: Integer position of the touching bar.
        touch_time: Timestamp of the touching bar.
        barrier: Which barrier was hit -- ``"pt"``, ``"sl"`` or ``"vertical"``.
        holding_bars: Bars held from entry to touch.
        side: The side that was traded, when meta-labelling.
    """

    label: pd.Series
    ret: pd.Series
    touch_idx: pd.Series
    touch_time: pd.Series
    barrier: pd.Series
    holding_bars: pd.Series
    side: pd.Series | None = None

    def to_frame(self) -> pd.DataFrame:
        """Assemble the fields into a single DataFrame."""
        data = {
            "label": self.label,
            "ret": self.ret,
            "touch_idx": self.touch_idx,
            "touch_time": self.touch_time,
            "barrier": self.barrier,
            "holding_bars": self.holding_bars,
        }
        if self.side is not None:
            data["side"] = self.side
        return pd.DataFrame(data)

    def __len__(self) -> int:
        return len(self.label)


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    horizon_bars: int,
    pt: float = 1.0,
    sl: float = 1.0,
    target: pd.Series | float | None = None,
    side: pd.Series | None = None,
    price_col: str = "close",
    high_col: str | None = None,
    low_col: str | None = None,
    min_target: float = 0.0,
    zero_on_vertical: bool = False,
    stop_at_session_end: bool = True,
) -> TripleBarrierResult:
    """Label each bar by which barrier its forward path touches first.

    For every bar *i*, walk forward at most *horizon_bars* and find the first
    bar where the cumulative return from ``close[i]`` crosses either
    ``+pt * target[i]`` or ``-sl * target[i]``. Whichever comes first (or the
    time limit) determines the label.

    Why this beats a fixed-horizon label: it encodes *path*. A fixed-horizon
    label calls a trade a winner if it happened to end up, even if it would
    have stopped you out on the way. Barriers are how a real position behaves.

    Args:
        df: Bar data with a tz-aware ``DatetimeIndex``.
        horizon_bars: Vertical barrier -- maximum bars to hold.
        pt: Profit-take width as a multiple of *target*. ``0`` disables it.
        sl: Stop-loss width as a multiple of *target*. ``0`` disables it.
        target: Per-bar barrier scale, normally a volatility estimate. A float
            applies one width everywhere. ``None`` uses
            :func:`ewm_volatility` with span ``100``.
        side: Optional primary-model side (``+1`` long / ``-1`` short). When
            given, barriers apply to the *side-adjusted* return and the label
            becomes binary (meta-labelling): 1 if the trade made money.
        price_col: Entry/exit price column.
        high_col: Column used to test the profit-take for longs. Defaults to
            *price_col*. Pass ``"high"`` to test intrabar touches.
        low_col: Column used to test the stop-loss for longs. Defaults to
            *price_col*. Pass ``"low"`` to test intrabar touches.
        min_target: Drop observations whose target is below this, where
            barriers would be too tight to be meaningful.
        zero_on_vertical: Label 0 when the time limit is hit first. When
            ``False`` (default) the label is the sign of the realised return.
        stop_at_session_end: Cap the vertical barrier at the session close, so
            a label never implies holding overnight.

    Returns:
        A :class:`TripleBarrierResult` aligned to *df*'s index, with ``NaN``
        where no valid forward window exists.

    Raises:
        ValueError: If *horizon_bars* < 1, if both barriers are disabled, or
            if *side* does not align with *df*.
        KeyError: If a requested column is absent.
    """
    if horizon_bars < 1:
        raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}.")
    if pt <= 0 and sl <= 0:
        raise ValueError(
            "At least one of pt/sl must be > 0; with both disabled this "
            "reduces to a fixed-horizon label -- use fixed_horizon_label()."
        )
    if price_col not in df.columns:
        raise KeyError(f"{price_col!r} not in DataFrame columns: {list(df.columns)}")

    high_col = high_col or price_col
    low_col = low_col or price_col
    for col in (high_col, low_col):
        if col not in df.columns:
            raise KeyError(f"{col!r} not in DataFrame columns: {list(df.columns)}")

    n = len(df)
    index = df.index

    # --- barrier scale -----------------------------------------------------
    if target is None:
        tgt = ewm_volatility(df[price_col], span=100).to_numpy(np.float64)
    elif isinstance(target, (int, float)):
        tgt = np.full(n, float(target), dtype=np.float64)
    else:
        if not target.index.equals(index):
            raise ValueError("target must share the DataFrame's index.")
        tgt = target.to_numpy(np.float64)

    # --- side (meta-labelling) --------------------------------------------
    if side is not None:
        if not side.index.equals(index):
            raise ValueError("side must share the DataFrame's index.")
        side_arr = side.to_numpy(np.float64)
    else:
        side_arr = None

    entry = df[price_col].to_numpy(np.float64)
    hi = df[high_col].to_numpy(np.float64)
    lo = df[low_col].to_numpy(np.float64)
    close = df[price_col].to_numpy(np.float64)

    sess_end = (
        session_end_positions(index)
        if stop_at_session_end
        else np.full(n, n - 1, dtype=np.int64)
    )

    label = np.full(n, np.nan, dtype=np.float64)
    ret = np.full(n, np.nan, dtype=np.float64)
    touch_idx = np.full(n, -1, dtype=np.int64)
    barrier = np.empty(n, dtype=object)
    holding = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        t = tgt[i]
        if not np.isfinite(t) or t < min_target or t <= 0:
            continue
        if side_arr is not None and not np.isfinite(side_arr[i]):
            continue

        last = min(i + horizon_bars, sess_end[i], n - 1)
        if last <= i:
            continue

        s = side_arr[i] if side_arr is not None else 1.0
        e = entry[i]
        if not np.isfinite(e) or e == 0.0:
            continue

        up = pt * t if pt > 0 else np.inf
        dn = sl * t if sl > 0 else np.inf

        window = slice(i + 1, last + 1)
        if s >= 0:
            # Long: profit on the high, stop on the low.
            fav = hi[window] / e - 1.0
            adv = lo[window] / e - 1.0
        else:
            # Short: profit when price falls (low), stop when it rises (high).
            fav = -(lo[window] / e - 1.0)
            adv = -(hi[window] / e - 1.0)

        hit_pt = np.flatnonzero(fav >= up)
        hit_sl = np.flatnonzero(adv <= -dn)
        first_pt = hit_pt[0] if hit_pt.size else np.iinfo(np.int64).max
        first_sl = hit_sl[0] if hit_sl.size else np.iinfo(np.int64).max

        if first_pt == first_sl == np.iinfo(np.int64).max:
            k = last - (i + 1)
            which = "vertical"
        elif first_pt <= first_sl:
            k = int(first_pt)
            which = "pt"
        else:
            k = int(first_sl)
            which = "sl"

        j = i + 1 + k
        # Realise at the barrier level when one was struck, else at the close.
        if which == "pt":
            r = up
        elif which == "sl":
            r = -dn
        else:
            r = s * (close[j] / e - 1.0)

        ret[i] = r
        touch_idx[i] = j
        barrier[i] = which
        holding[i] = j - i

        if side_arr is not None:
            label[i] = 1.0 if r > 0 else 0.0
        elif which == "vertical" and zero_on_vertical:
            label[i] = 0.0
        else:
            label[i] = float(np.sign(r))

    idx_series = pd.Series(touch_idx, index=index, dtype="int64")
    touch_time = pd.Series(pd.NaT, index=index, dtype=index.dtype)
    valid = touch_idx >= 0
    if valid.any():
        touch_time.iloc[np.flatnonzero(valid)] = index[touch_idx[valid]]

    return TripleBarrierResult(
        label=pd.Series(label, index=index, name="tb_label"),
        ret=pd.Series(ret, index=index, name="tb_ret"),
        touch_idx=idx_series.where(pd.Series(valid, index=index), -1).rename("tb_touch_idx"),
        touch_time=touch_time.rename("tb_touch_time"),
        barrier=pd.Series(barrier, index=index, name="tb_barrier"),
        holding_bars=pd.Series(holding, index=index, name="tb_holding_bars"),
        side=None if side is None else side.rename("tb_side"),
    )


def meta_labels(
    df: pd.DataFrame,
    side: pd.Series,
    *,
    horizon_bars: int,
    pt: float = 1.0,
    sl: float = 1.0,
    target: pd.Series | float | None = None,
    **kwargs,
) -> TripleBarrierResult:
    """Label whether acting on a primary model's *side* would have paid.

    Meta-labelling splits the problem in two: the primary model picks a
    direction, and a second model -- trained on these labels -- decides
    whether to take the trade and how big. Predicting "is this signal any
    good?" is usually an easier problem than predicting direction, and it
    gives you a natural place to put a position size.

    A ``side`` of 0 means the primary model is flat; those bars get no label.

    Args:
        df: Bar data.
        side: Primary model's direction, ``+1`` / ``-1`` (``0`` = no signal).
        horizon_bars: Vertical barrier in bars.
        pt: Profit-take multiple of *target*.
        sl: Stop-loss multiple of *target*.
        target: Barrier scale; defaults to EWM volatility.
        **kwargs: Forwarded to :func:`triple_barrier_labels`.

    Returns:
        A :class:`TripleBarrierResult` whose ``label`` is binary: 1 = the
        trade made money, 0 = it did not.

    Raises:
        ValueError: If *side* does not align with *df*.
    """
    if not side.index.equals(df.index):
        raise ValueError("side must share the DataFrame's index.")
    masked = side.replace(0, np.nan)
    return triple_barrier_labels(
        df,
        horizon_bars=horizon_bars,
        pt=pt,
        sl=sl,
        target=target,
        side=masked,
        **kwargs,
    )
