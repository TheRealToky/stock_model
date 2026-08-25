"""Sample weights for overlapping labels.

Triple-barrier labels overlap: a label opened at bar *i* may still be running
when the label at bar *i+1* opens, so the two are driven by largely the same
price path. Standard ML assumes independent samples. Feeding it 977k
overlapping 1-minute labels and calling that 977k observations overstates the
effective sample size by roughly the average holding period -- which is exactly
the kind of error that makes an out-of-sample AUC of 0.50 look like a
surprise rather than the expected outcome.

Three corrections, all from Lopez de Prado's *Advances in Financial Machine
Learning*, chapter 4:

* **Uniqueness** -- how much of a label's lifespan is *not* shared with other
  labels. A label overlapped by nine others counts for a tenth of a sample.
* **Return attribution** -- weight labels by the magnitude of the return they
  actually explain, so a label spanning a big move counts more than one
  spanning noise.
* **Time decay** -- optionally fade older observations, on the view that
  recent market structure is more relevant.

All three return weights aligned to the label index, ready to hand to
``sample_weight=`` in sklearn/XGBoost or to a torch loss reduction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "concurrency",
    "average_uniqueness",
    "return_attribution_weights",
    "time_decay_weights",
    "effective_sample_size",
    "combined_weights",
]


def _validate(touch_idx: pd.Series, n_bars: int) -> np.ndarray:
    """Coerce touch positions to a clean int array, or raise.

    Args:
        touch_idx: Integer end position per label; ``-1`` marks "no label".
        n_bars: Total bars in the underlying price series.

    Returns:
        Integer numpy array of end positions.

    Raises:
        ValueError: If any end position points outside the series.
    """
    arr = touch_idx.to_numpy(np.int64)
    bad = arr[(arr >= n_bars) & (arr >= 0)]
    if bad.size:
        raise ValueError(
            f"touch_idx contains positions beyond the series end "
            f"(n_bars={n_bars}, offending max={bad.max()})."
        )
    return arr


def concurrency(
    touch_idx: pd.Series,
    *,
    start_idx: pd.Series | None = None,
    n_bars: int | None = None,
) -> pd.Series:
    """Number of labels live at each bar.

    Computed with a difference array, so this is O(n) rather than O(n * span).

    Args:
        touch_idx: Integer position where each label closes; ``-1`` = no label.
        start_idx: Integer position where each label opens. Defaults to the
            label's own position (0, 1, 2, ...).
        n_bars: Length of the underlying series. Defaults to ``len(touch_idx)``.

    Returns:
        Integer Series, indexed like *touch_idx*, counting live labels per bar.
    """
    n = int(n_bars if n_bars is not None else len(touch_idx))
    ends = _validate(touch_idx, n)
    starts = (
        np.arange(len(touch_idx), dtype=np.int64)
        if start_idx is None
        else start_idx.to_numpy(np.int64)
    )

    diff = np.zeros(n + 1, dtype=np.int64)
    live = ends >= 0
    np.add.at(diff, starts[live], 1)
    np.add.at(diff, ends[live] + 1, -1)
    counts = np.cumsum(diff[:-1])
    return pd.Series(counts, index=touch_idx.index[:n], name="concurrency")


def average_uniqueness(
    touch_idx: pd.Series,
    *,
    start_idx: pd.Series | None = None,
) -> pd.Series:
    """Mean uniqueness of each label over its own lifespan.

    A label live alongside *c* others at a given bar contributes ``1/c`` of an
    independent observation at that bar. Averaging across its lifespan gives a
    number in ``(0, 1]``: 1.0 means fully independent, 0.1 means it is
    essentially a tenth of a sample.

    Args:
        touch_idx: Integer close position per label; ``-1`` = no label.
        start_idx: Integer open position per label. Defaults to positional.

    Returns:
        Float Series in ``(0, 1]``, ``NaN`` where there is no label.
    """
    n = len(touch_idx)
    ends = _validate(touch_idx, n)
    starts = (
        np.arange(n, dtype=np.int64)
        if start_idx is None
        else start_idx.to_numpy(np.int64)
    )

    conc = concurrency(touch_idx, start_idx=start_idx, n_bars=n).to_numpy(np.float64)
    # Prefix sums of 1/concurrency let each label's mean be read in O(1).
    inv = np.divide(1.0, conc, out=np.zeros_like(conc), where=conc > 0)
    prefix = np.concatenate([[0.0], np.cumsum(inv)])

    out = np.full(n, np.nan, dtype=np.float64)
    live = ends >= 0
    s = starts[live]
    e = ends[live]
    span = (e - s + 1).astype(np.float64)
    out[live] = (prefix[e + 1] - prefix[s]) / span
    return pd.Series(out, index=touch_idx.index, name="uniqueness")


def return_attribution_weights(
    touch_idx: pd.Series,
    close: pd.Series,
    *,
    start_idx: pd.Series | None = None,
    normalize: bool = True,
) -> pd.Series:
    """Weight labels by the magnitude of return they uniquely explain.

    Each bar's log return is split evenly among the labels live at that bar;
    a label's weight is the absolute sum of its shares. A label that spans a
    sharp, uncrowded move earns more weight than one spanning flat, crowded
    noise.

    Args:
        touch_idx: Integer close position per label; ``-1`` = no label.
        close: Close prices for the same bars.
        start_idx: Integer open position per label. Defaults to positional.
        normalize: Scale weights to average 1.0, which keeps the effective
            learning rate comparable to unweighted training.

    Returns:
        Non-negative float Series, ``NaN`` where there is no label.

    Raises:
        ValueError: If *close* does not align with *touch_idx*.
    """
    if len(close) != len(touch_idx):
        raise ValueError(
            f"close and touch_idx must be the same length; "
            f"got {len(close)} and {len(touch_idx)}."
        )

    n = len(touch_idx)
    ends = _validate(touch_idx, n)
    starts = (
        np.arange(n, dtype=np.int64)
        if start_idx is None
        else start_idx.to_numpy(np.int64)
    )

    conc = concurrency(touch_idx, start_idx=start_idx, n_bars=n).to_numpy(np.float64)
    log_ret = np.log(close.to_numpy(np.float64)).astype(np.float64)
    bar_ret = np.zeros(n, dtype=np.float64)
    bar_ret[1:] = np.diff(log_ret)
    bar_ret = np.nan_to_num(bar_ret, nan=0.0, posinf=0.0, neginf=0.0)

    share = np.divide(bar_ret, conc, out=np.zeros_like(bar_ret), where=conc > 0)
    prefix = np.concatenate([[0.0], np.cumsum(share)])

    out = np.full(n, np.nan, dtype=np.float64)
    live = ends >= 0
    s = starts[live]
    e = ends[live]
    # A label entered at the close of bar s realises returns over bars
    # s+1 .. e, matching triple_barrier_labels' log(close[e]/close[s]).
    # Summing from s instead would credit it with the move INTO its own
    # entry bar -- a return it could not have captured.
    out[live] = np.abs(prefix[e + 1] - prefix[s + 1])

    result = pd.Series(out, index=touch_idx.index, name="return_weight")
    if normalize:
        mean = np.nanmean(out) if live.any() else np.nan
        if np.isfinite(mean) and mean > 0:
            result = result / mean
    return result


def time_decay_weights(
    uniqueness: pd.Series,
    *,
    last_weight: float = 1.0,
) -> pd.Series:
    """Linear time decay applied over cumulative uniqueness.

    Decay runs on *uniqueness-time* rather than clock time, so a stretch of
    heavily-overlapping labels does not age faster than it deserves to.

    Args:
        uniqueness: Output of :func:`average_uniqueness`.
        last_weight: Weight of the newest observation relative to... itself
            (always 1.0 at the newest point); this sets the weight of the
            *oldest*. ``1.0`` = no decay. ``0.0`` = oldest gets zero weight.
            Negative values (down to -1) zero out the oldest fraction
            entirely, dropping that history from training.

    Returns:
        Non-negative float Series aligned to *uniqueness*.

    Raises:
        ValueError: If *last_weight* is outside ``[-1, 1]``.
    """
    if not -1.0 <= last_weight <= 1.0:
        raise ValueError(f"last_weight must be in [-1, 1], got {last_weight}.")

    u = uniqueness.dropna()
    if u.empty:
        return pd.Series(np.nan, index=uniqueness.index, name="time_decay")

    cum = u.cumsum()
    total = float(cum.iloc[-1])
    if total <= 0:
        return pd.Series(1.0, index=uniqueness.index, name="time_decay")

    if last_weight >= 0:
        slope = (1.0 - last_weight) / total
    else:
        # Negative: weights hit zero partway through, erasing the oldest data.
        slope = 1.0 / ((last_weight + 1) * total)
    const = 1.0 - slope * total

    w = const + slope * cum
    w[w < 0] = 0.0
    return w.reindex(uniqueness.index).rename("time_decay")


def effective_sample_size(uniqueness: pd.Series) -> float:
    """Sum of uniqueness -- the number of *independent* observations.

    Compare this against ``len()`` to see how badly overlap is inflating an
    apparent sample size.

    Args:
        uniqueness: Output of :func:`average_uniqueness`.

    Returns:
        Effective count of independent samples.
    """
    return float(uniqueness.sum(skipna=True))


def combined_weights(
    touch_idx: pd.Series,
    close: pd.Series,
    *,
    start_idx: pd.Series | None = None,
    decay: float = 1.0,
    normalize: bool = True,
) -> pd.Series:
    """Return-attribution weights, optionally faded by time decay.

    The usual default: weight by the return a label explains, then decay.

    Args:
        touch_idx: Integer close position per label.
        close: Close prices.
        start_idx: Integer open position per label.
        decay: ``last_weight`` for :func:`time_decay_weights`; 1.0 disables.
        normalize: Rescale the result to average 1.0.

    Returns:
        Float Series of sample weights, ``NaN`` where there is no label.
    """
    w = return_attribution_weights(
        touch_idx, close, start_idx=start_idx, normalize=False
    )
    if decay != 1.0:
        u = average_uniqueness(touch_idx, start_idx=start_idx)
        w = w * time_decay_weights(u, last_weight=decay)
    if normalize:
        mean = np.nanmean(w.to_numpy(np.float64))
        if np.isfinite(mean) and mean > 0:
            w = w / mean
    return w.rename("sample_weight")
