"""Cross-validation and overfitting diagnostics for financial ML.

Standard k-fold cross-validation is invalid on this data, for two reasons that
compound:

1. **Shuffling breaks time.** Training on 2026 to predict 2021 is not a
   forecast. ``TimeSeriesSplit`` fixes only this much, and it is what
   ``models/trainer.py`` currently uses.
2. **Overlapping labels leak across the fold boundary.** A triple-barrier
   label opened just before the test block closes *inside* it, so the training
   set contains a sample whose outcome is determined by test-period prices.
   ``TimeSeriesSplit`` does nothing about this, and at a 390-bar horizon on
   1-minute data it contaminates 390 training samples at every boundary.

:class:`PurgedKFold` removes both. :class:`CombinatorialPurgedCV` goes further
and produces *many* backtest paths from one dataset, which is what makes the
overfitting statistics below meaningful.

The diagnostics answer the question that matters after a search: **given that I
tried N configurations, how surprised should I be by the best one?**

* :func:`deflated_sharpe_ratio` -- the probability the true Sharpe exceeds
  zero, after discounting for the number of trials it took to find it. A
  Sharpe of 2.0 found on the first try and one found on the five-hundredth are
  not the same evidence.
* :func:`probability_of_backtest_overfitting` -- how often the best in-sample
  configuration lands below median out-of-sample. Above ~0.5 the selection
  procedure is worse than choosing at random.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "PurgedKFold",
    "CombinatorialPurgedCV",
    "purged_train_indices",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "minimum_track_record_length",
    "probability_of_backtest_overfitting",
]

EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# Purging and embargo
# ---------------------------------------------------------------------------

def purged_train_indices(
    n_samples: int,
    test_start: int,
    test_end: int,
    *,
    touch_idx: np.ndarray | pd.Series | None = None,
    embargo: int = 0,
) -> np.ndarray:
    """Training positions that neither overlap nor immediately follow a test block.

    Two exclusions:

    * **Purge** -- drop any training sample whose label is still open when the
      test block starts. With ``touch_idx`` supplied, sample *i* is dropped
      when ``i <= test_end and touch_idx[i] >= test_start``.
    * **Embargo** -- drop the *embargo* samples immediately after the test
      block. Serial correlation means a sample just after the test window is
      still nearly the same observation, even if its label does not formally
      overlap.

    Args:
        n_samples: Total samples.
        test_start: First position of the test block (inclusive).
        test_end: Last position of the test block (inclusive).
        touch_idx: Integer position where each sample's label closes. When
            ``None``, only the test block itself and the embargo are removed.
        embargo: Samples to embargo after the test block.

    Returns:
        Sorted integer array of usable training positions.

    Raises:
        ValueError: If the test block is out of range or inverted.
    """
    if not 0 <= test_start <= test_end < n_samples:
        raise ValueError(
            f"Invalid test block [{test_start}, {test_end}] for n_samples={n_samples}."
        )
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}.")

    keep = np.ones(n_samples, dtype=bool)
    keep[test_start : test_end + 1] = False

    if embargo:
        keep[test_end + 1 : test_end + 1 + embargo] = False

    if touch_idx is not None:
        t1 = np.asarray(
            touch_idx.to_numpy() if isinstance(touch_idx, pd.Series) else touch_idx,
            dtype=np.int64,
        )
        if len(t1) != n_samples:
            raise ValueError(
                f"touch_idx length {len(t1)} does not match n_samples={n_samples}."
            )
        starts = np.arange(n_samples, dtype=np.int64)
        # A label spanning [i, t1_i] overlaps [test_start, test_end] iff
        # i <= test_end and t1_i >= test_start. Unlabelled (-1) never overlaps.
        overlaps = (starts <= test_end) & (t1 >= test_start) & (t1 >= 0)
        keep &= ~overlaps

    return np.flatnonzero(keep)


class PurgedKFold:
    """Time-ordered k-fold with label purging and an embargo.

    Folds are contiguous blocks in time (never shuffled). For each fold, the
    training set drops every sample whose label overlaps the test block, plus
    an embargo immediately after it.

    Follows the sklearn splitter protocol, so it can be handed straight to
    ``cross_val_score`` or ``GridSearchCV``.

    Args:
        n_splits: Number of folds. Must be >= 2.
        touch_idx: Integer close position per sample, from
            :func:`research.lib.labeling.triple_barrier_labels`. Without it
            no purging happens and this degrades to blocked CV.
        embargo_pct: Embargo size as a fraction of the whole sample.
        expanding: When ``True``, training uses only data *before* the test
            block (walk-forward). When ``False`` (default) it uses data on
            both sides, which is valid once purged and uses more data.

    Raises:
        ValueError: If *n_splits* < 2 or *embargo_pct* is negative.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        touch_idx: np.ndarray | pd.Series | None = None,
        embargo_pct: float = 0.0,
        expanding: bool = False,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}.")
        if embargo_pct < 0:
            raise ValueError(f"embargo_pct must be >= 0, got {embargo_pct}.")
        self.n_splits = int(n_splits)
        self.touch_idx = touch_idx
        self.embargo_pct = float(embargo_pct)
        self.expanding = bool(expanding)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of folds (sklearn protocol)."""
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_positions, test_positions)`` per fold.

        Args:
            X: Only its length is used.
            y: Ignored; present for the sklearn protocol.
            groups: Ignored; present for the sklearn protocol.

        Yields:
            Tuples of integer position arrays.
        """
        n = len(X)
        embargo = int(n * self.embargo_pct)
        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)

        for k in range(self.n_splits):
            test_start, test_end = bounds[k], bounds[k + 1] - 1
            if test_end < test_start:
                continue
            train = purged_train_indices(
                n, test_start, test_end, touch_idx=self.touch_idx, embargo=embargo
            )
            if self.expanding:
                train = train[train < test_start]
            test = np.arange(test_start, test_end + 1)
            if len(train) and len(test):
                yield train, test


class CombinatorialPurgedCV:
    """Combinatorial purged cross-validation (Lopez de Prado, ch. 12).

    Splits the sample into *n_groups* contiguous blocks and tests on every
    combination of *n_test_groups* of them at once, purging and embargoing as
    usual. With 6 groups tested 2 at a time that is 15 splits -- and because
    each group appears in several of them, they recombine into **5 distinct
    backtest paths** rather than one.

    That matters because a single backtest gives you one number with no
    dispersion. Five paths give you a distribution, which is what
    :func:`probability_of_backtest_overfitting` and
    :func:`deflated_sharpe_ratio` need in order to say anything honest about
    whether a result is luck.

    Args:
        n_groups: Contiguous blocks to divide the sample into.
        n_test_groups: How many blocks form the test set in each split.
        touch_idx: Integer close position per sample, for purging.
        embargo_pct: Embargo as a fraction of the sample.

    Raises:
        ValueError: If the group counts are inconsistent.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        *,
        touch_idx: np.ndarray | pd.Series | None = None,
        embargo_pct: float = 0.0,
    ) -> None:
        if n_groups < 2:
            raise ValueError(f"n_groups must be >= 2, got {n_groups}.")
        if not 1 <= n_test_groups < n_groups:
            raise ValueError(
                f"n_test_groups must be in [1, {n_groups - 1}], got {n_test_groups}."
            )
        self.n_groups = int(n_groups)
        self.n_test_groups = int(n_test_groups)
        self.touch_idx = touch_idx
        self.embargo_pct = float(embargo_pct)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of train/test combinations."""
        return math.comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """Number of distinct backtest paths the splits recombine into."""
        return (
            math.comb(self.n_groups, self.n_test_groups)
            * self.n_test_groups
            // self.n_groups
        )

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_positions, test_positions)`` per combination.

        Args:
            X: Only its length is used.
            y: Ignored; sklearn protocol.
            groups: Ignored; sklearn protocol.

        Yields:
            Tuples of integer position arrays. Test positions may be
            non-contiguous (several blocks at once).
        """
        n = len(X)
        embargo = int(n * self.embargo_pct)
        bounds = np.linspace(0, n, self.n_groups + 1).astype(int)
        blocks = [(bounds[i], bounds[i + 1] - 1) for i in range(self.n_groups)]

        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test_parts = [np.arange(blocks[g][0], blocks[g][1] + 1) for g in combo]
            test = np.concatenate(test_parts)

            keep = np.ones(n, dtype=bool)
            for g in combo:
                s, e = blocks[g]
                # Purge/embargo around each test block independently, then
                # intersect -- a sample must survive all of them.
                allowed = purged_train_indices(
                    n, s, e, touch_idx=self.touch_idx, embargo=embargo
                )
                mask = np.zeros(n, dtype=bool)
                mask[allowed] = True
                keep &= mask

            train = np.flatnonzero(keep)
            if len(train) and len(test):
                yield train, np.sort(test)


# ---------------------------------------------------------------------------
# Overfitting diagnostics
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    n_obs: int,
    *,
    benchmark_sharpe: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability the true Sharpe exceeds *benchmark_sharpe*.

    Corrects for short samples and for non-normal returns: negative skew and
    fat tails both make a given Sharpe less trustworthy than the textbook
    standard error suggests.

    All Sharpe inputs must be in the **same periodicity** as *n_obs* -- pass
    per-bar Sharpe with a per-bar count, or annualised Sharpe with a count of
    years. Mixing them silently produces nonsense.

    Args:
        observed_sharpe: Estimated Sharpe.
        n_obs: Number of observations it was estimated from.
        benchmark_sharpe: Threshold to beat.
        skew: Skewness of the return series.
        kurtosis: Kurtosis (3.0 = normal).

    Returns:
        Probability in ``[0, 1]``.

    Raises:
        ValueError: If *n_obs* < 2.
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}.")

    denom = 1.0 - skew * observed_sharpe + 0.25 * (kurtosis - 1.0) * observed_sharpe**2
    if denom <= 0:
        # Extreme skew/kurtosis relative to the Sharpe; the estimator breaks.
        return float("nan")
    z = (observed_sharpe - benchmark_sharpe) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe from *n_trials* draws under a null of no skill.

    This is the bar a genuine result has to clear. Run enough backtests and
    some configuration will post a great Sharpe purely by chance; this
    quantifies how great, so it can be subtracted off.

    Args:
        n_trials: Number of configurations tried.
        sharpe_variance: Variance of Sharpe *across* those trials.

    Returns:
        Expected maximum Sharpe under the null.

    Raises:
        ValueError: If *n_trials* < 1 or *sharpe_variance* < 0.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}.")
    if sharpe_variance < 0:
        raise ValueError(f"sharpe_variance must be >= 0, got {sharpe_variance}.")
    if n_trials == 1 or sharpe_variance == 0:
        return 0.0

    g = EULER_MASCHERONI
    q1 = norm.ppf(1.0 - 1.0 / n_trials)
    q2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(sharpe_variance) * ((1.0 - g) * q1 + g * q2))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_obs: int,
    *,
    n_trials: int,
    sharpe_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability a Sharpe is real, given how many tries it took to find it.

    The single most useful number to report after a parameter sweep. A Sharpe
    of 1.5 discovered on the first attempt is strong evidence; the same 1.5
    picked as the best of 500 configurations usually is not, and this collapses
    toward 0.5 or below to say so.

    Args:
        observed_sharpe: Sharpe of the selected (best) configuration.
        n_obs: Observations behind that estimate.
        n_trials: How many configurations were tried before selecting it.
        sharpe_variance: Variance of Sharpe across those trials.
        skew: Skewness of the selected strategy's returns.
        kurtosis: Kurtosis of the selected strategy's returns.

    Returns:
        Probability in ``[0, 1]``. Convention: below ~0.95 is not evidence of
        skill.
    """
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)
    return probabilistic_sharpe_ratio(
        observed_sharpe,
        n_obs,
        benchmark_sharpe=sr0,
        skew=skew,
        kurtosis=kurtosis,
    )


def minimum_track_record_length(
    observed_sharpe: float,
    *,
    benchmark_sharpe: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> float:
    """Observations needed before a Sharpe is significant at *confidence*.

    Answers "how much more data do I need?" -- and often reveals that a result
    could not possibly be established from the sample available.

    Args:
        observed_sharpe: Estimated Sharpe.
        benchmark_sharpe: Threshold to beat.
        skew: Skewness of returns.
        kurtosis: Kurtosis of returns.
        confidence: Required confidence level.

    Returns:
        Minimum number of observations. ``inf`` when the observed Sharpe does
        not exceed the benchmark at all.

    Raises:
        ValueError: If *confidence* is not in ``(0, 1)``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")
    diff = observed_sharpe - benchmark_sharpe
    if diff <= 0:
        return float("inf")
    denom = 1.0 - skew * observed_sharpe + 0.25 * (kurtosis - 1.0) * observed_sharpe**2
    if denom <= 0:
        return float("nan")
    return float(1.0 + denom * (norm.ppf(confidence) / diff) ** 2)


def probability_of_backtest_overfitting(
    performance: pd.DataFrame | np.ndarray,
    *,
    n_splits: int = 16,
    higher_is_better: bool = True,
) -> dict[str, float]:
    """Combinatorially-symmetric estimate of overfitting probability (PBO).

    Takes a matrix of per-period performance -- rows are time slices, columns
    are candidate configurations -- and repeatedly splits the rows in half. In
    each split it picks the configuration that looked best in-sample and checks
    where that choice ranks out-of-sample.

    PBO is the fraction of splits where the in-sample winner landed **below
    median** out-of-sample. At 0.5 your selection procedure carries no
    information; above it, you are reliably picking losers.

    Read a single PBO with caution. The statistic is unbiased in expectation
    (pure noise averages to 0.5 over repeated datasets) but its variance on
    *one* dataset is large: in a 600x20 noise simulation, individual matrices
    produced mean OOS ranks anywhere from 5.2 to 18.7 out of 20, i.e. PBO
    values from roughly 0.1 to 0.9 with no edge present at all. Treat a lone
    reading as weak evidence, and prefer comparing PBO across horizons or
    universes over thresholding one number.

    Args:
        performance: ``(n_periods, n_configs)`` of per-period performance,
            e.g. returns per bar for each parameter setting.
        n_splits: Number of row groups. Must be even; ``C(n_splits, n_splits/2)``
            combinations are evaluated, so keep it modest (16 -> 12,870).
        higher_is_better: Whether larger performance values are better.

    Returns:
        Dict with ``pbo`` (the probability), ``n_combinations`` evaluated, and
        ``mean_logit`` -- the average log-odds of the OOS rank, where negative
        values indicate systematic overfitting.

    Raises:
        ValueError: If *n_splits* is odd, or the matrix is too small.
    """
    mat = performance.to_numpy(np.float64) if isinstance(performance, pd.DataFrame) else np.asarray(
        performance, dtype=np.float64
    )
    if mat.ndim != 2:
        raise ValueError(f"performance must be 2-D, got shape {mat.shape}.")
    n_periods, n_configs = mat.shape
    if n_configs < 2:
        raise ValueError(f"Need at least 2 configurations, got {n_configs}.")
    if n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even, got {n_splits}.")
    if n_periods < n_splits:
        raise ValueError(
            f"Need at least n_splits={n_splits} periods, got {n_periods}."
        )

    if not higher_is_better:
        mat = -mat

    bounds = np.linspace(0, n_periods, n_splits + 1).astype(int)
    groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_splits)]

    half = n_splits // 2
    logits: list[float] = []

    for combo in combinations(range(n_splits), half):
        is_rows = np.concatenate([groups[g] for g in combo])
        oos_rows = np.concatenate(
            [groups[g] for g in range(n_splits) if g not in combo]
        )
        if not len(is_rows) or not len(oos_rows):
            continue

        is_perf = mat[is_rows].mean(axis=0)
        oos_perf = mat[oos_rows].mean(axis=0)

        best = int(np.argmax(is_perf))
        # Relative rank of the IS winner among OOS results, in (0, 1).
        rank = (float(np.sum(oos_perf <= oos_perf[best])) ) / (n_configs + 1)
        rank = min(max(rank, 1e-9), 1.0 - 1e-9)
        logits.append(math.log(rank / (1.0 - rank)))

    if not logits:
        return {"pbo": float("nan"), "n_combinations": 0, "mean_logit": float("nan")}

    arr = np.asarray(logits, dtype=np.float64)
    return {
        "pbo": float(np.mean(arr <= 0.0)),
        "n_combinations": len(arr),
        "mean_logit": float(np.mean(arr)),
    }
