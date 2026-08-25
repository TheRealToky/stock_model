"""Tests for research.lib.weights.

The property that matters: overlapping labels must not be counted as
independent observations. A run of 977k one-minute triple-barrier labels with a
390-bar horizon is nowhere near 977k independent samples, and treating it as
such is what makes a 0.50 out-of-sample AUC look surprising.
"""

import numpy as np
import pandas as pd
import pytest

from research.lib import weights as W


def idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")


class TestConcurrency:
    def test_counts_live_labels_per_bar(self):
        i = idx(6)
        touch = pd.Series([2, 3, 4, 5, 5, -1], index=i)
        assert list(W.concurrency(touch)) == [1, 2, 3, 3, 3, 2]

    def test_non_overlapping_labels_are_never_concurrent(self):
        i = idx(4)
        touch = pd.Series([0, 1, 2, 3], index=i)
        assert list(W.concurrency(touch)) == [1, 1, 1, 1]

    def test_unlabelled_rows_contribute_nothing(self):
        i = idx(4)
        assert list(W.concurrency(pd.Series([-1, -1, -1, -1], index=i))) == [0, 0, 0, 0]

    def test_touch_beyond_series_end_raises(self):
        i = idx(3)
        with pytest.raises(ValueError, match="beyond the series end"):
            W.concurrency(pd.Series([0, 1, 99], index=i))


class TestAverageUniqueness:
    def test_disjoint_labels_are_fully_unique(self):
        i = idx(5)
        u = W.average_uniqueness(pd.Series(list(range(5)), index=i))
        assert np.allclose(u.to_numpy(), 1.0)

    def test_effective_sample_size_equals_count_when_disjoint(self):
        i = idx(5)
        u = W.average_uniqueness(pd.Series(list(range(5)), index=i))
        assert W.effective_sample_size(u) == pytest.approx(5.0)

    def test_total_overlap_collapses_the_effective_sample(self):
        """Six labels all spanning the whole series are ~one observation."""
        i = idx(6)
        u = W.average_uniqueness(pd.Series([5] * 6, index=i))
        eff = W.effective_sample_size(u)
        assert eff < 2.0, f"expected near-total collapse, got {eff}"
        assert eff > 0.0

    def test_uniqueness_is_bounded(self):
        i = idx(20)
        touch = pd.Series(np.minimum(np.arange(20) + 5, 19), index=i)
        u = W.average_uniqueness(touch).dropna()
        assert (u > 0).all() and (u <= 1.0 + 1e-12).all()

    def test_unlabelled_rows_are_nan(self):
        i = idx(4)
        u = W.average_uniqueness(pd.Series([1, 2, -1, 3], index=i))
        assert np.isnan(u.iloc[2])
        assert u.notna().sum() == 3

    def test_overlap_reduces_effective_size_below_raw_count(self):
        i = idx(50)
        touch = pd.Series(np.minimum(np.arange(50) + 10, 49), index=i)
        u = W.average_uniqueness(touch)
        assert W.effective_sample_size(u) < 50 * 0.5


class TestReturnAttribution:
    def test_solo_label_weight_equals_its_own_realised_return(self):
        i = idx(6)
        close = pd.Series([100, 100, 100, 100, 110, 110.0], index=i)
        solo = pd.Series([-1, -1, -1, 4, -1, -1], index=i)
        w = W.return_attribution_weights(solo, close, normalize=False)
        assert w.iloc[3] == pytest.approx(abs(np.log(110 / 100)))

    def test_label_spanning_the_move_outweighs_a_flat_one(self):
        i = idx(6)
        close = pd.Series([100, 100, 100, 100, 110, 110.0], index=i)
        touch = pd.Series([1, 2, 3, 4, 5, -1], index=i)
        w = W.return_attribution_weights(touch, close, normalize=False)
        assert int(np.nanargmax(w.to_numpy())) == 3
        assert w.iloc[4] == pytest.approx(0.0)

    def test_weights_are_non_negative(self):
        i = idx(30)
        rng = np.random.default_rng(0)
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 30)), index=i)
        touch = pd.Series(np.minimum(np.arange(30) + 4, 29), index=i)
        w = W.return_attribution_weights(touch, close).dropna()
        assert (w >= 0).all()

    def test_normalisation_gives_mean_one(self):
        i = idx(30)
        rng = np.random.default_rng(1)
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 30)), index=i)
        touch = pd.Series(np.minimum(np.arange(30) + 4, 29), index=i)
        w = W.return_attribution_weights(touch, close, normalize=True)
        assert np.nanmean(w.to_numpy()) == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        i = idx(5)
        with pytest.raises(ValueError, match="same length"):
            W.return_attribution_weights(
                pd.Series([1, 2, 3, 4, 4], index=i), pd.Series([100.0, 101.0])
            )


class TestTimeDecay:
    def _uniqueness(self, n=20):
        i = idx(n)
        return W.average_uniqueness(pd.Series(list(range(n)), index=i))

    def test_no_decay_leaves_weights_flat(self):
        w = W.time_decay_weights(self._uniqueness(), last_weight=1.0)
        assert np.allclose(w.to_numpy(), 1.0)

    def test_decay_makes_older_observations_lighter(self):
        w = W.time_decay_weights(self._uniqueness(), last_weight=0.2)
        arr = w.to_numpy()
        assert arr[0] < arr[-1]
        assert np.all(np.diff(arr) >= -1e-12), "weights must increase with time"

    def test_negative_last_weight_zeroes_the_oldest_history(self):
        w = W.time_decay_weights(self._uniqueness(), last_weight=-0.5)
        assert (w.to_numpy() == 0.0).any()
        assert (w.to_numpy() >= 0).all()

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"last_weight must be in \[-1, 1\]"):
            W.time_decay_weights(self._uniqueness(), last_weight=2.0)


class TestCombinedWeights:
    def test_combines_attribution_and_decay(self):
        i = idx(40)
        rng = np.random.default_rng(2)
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 40)), index=i)
        touch = pd.Series(np.minimum(np.arange(40) + 5, 39), index=i)
        w = W.combined_weights(touch, close, decay=0.3)
        assert w.notna().sum() > 0
        assert (w.dropna() >= 0).all()
        assert np.nanmean(w.to_numpy()) == pytest.approx(1.0)

    def test_decay_of_one_matches_plain_attribution(self):
        i = idx(30)
        rng = np.random.default_rng(3)
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 30)), index=i)
        touch = pd.Series(np.minimum(np.arange(30) + 3, 29), index=i)
        a = W.combined_weights(touch, close, decay=1.0)
        b = W.return_attribution_weights(touch, close, normalize=True)
        assert np.allclose(a.dropna().to_numpy(), b.dropna().to_numpy())
