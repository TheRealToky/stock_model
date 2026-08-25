"""Tests for research.lib.validation.

The headline property: a training sample whose label is still open when the
test block begins must not survive into the training set. Plain blocked CV
leaves those in, and at a 390-bar horizon on 1-minute data that is 390
contaminated samples per fold boundary.
"""

import numpy as np
import pandas as pd
import pytest

from research.lib import validation as V


def overlapping_touches(n: int, span: int) -> np.ndarray:
    """Label ending *span* bars after it opens, clipped at the series end."""
    return np.minimum(np.arange(n) + span, n - 1)


class TestPurgedTrainIndices:
    def test_test_block_itself_is_always_excluded(self):
        tr = V.purged_train_indices(50, 10, 19)
        assert not set(range(10, 20)) & set(tr)

    def test_purge_removes_labels_reaching_into_the_test_block(self):
        n, span = 50, 5
        t1 = overlapping_touches(n, span)
        tr = V.purged_train_indices(n, 20, 29, touch_idx=t1)
        # Bars 15..19 have labels closing at 20..24 -- inside the test block.
        assert not set(range(15, 20)) & set(tr)
        assert 14 in tr

    def test_embargo_removes_samples_just_after_the_test_block(self):
        tr = V.purged_train_indices(50, 10, 19, embargo=5)
        assert not set(range(20, 25)) & set(tr)
        assert 25 in tr

    def test_without_touch_idx_only_block_and_embargo_are_dropped(self):
        tr = V.purged_train_indices(30, 10, 14)
        assert len(tr) == 25

    def test_invalid_block_raises(self):
        with pytest.raises(ValueError, match="Invalid test block"):
            V.purged_train_indices(20, 15, 5)
        with pytest.raises(ValueError, match="Invalid test block"):
            V.purged_train_indices(20, 0, 20)

    def test_touch_idx_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match n_samples"):
            V.purged_train_indices(20, 0, 4, touch_idx=np.zeros(5, dtype=int))

    def test_negative_embargo_raises(self):
        with pytest.raises(ValueError, match="embargo must be >= 0"):
            V.purged_train_indices(20, 0, 4, embargo=-1)


class TestPurgedKFold:
    def test_no_training_label_leaks_into_any_test_block(self):
        n, span = 200, 10
        t1 = overlapping_touches(n, span)
        cv = V.PurgedKFold(n_splits=5, touch_idx=t1, embargo_pct=0.02)
        for train, test in cv.split(np.zeros((n, 2))):
            leaked = (train <= test[-1]) & (t1[train] >= test[0])
            assert not leaked.any(), "purge failed to remove an overlapping label"

    def test_unpurged_blocking_does_leak(self):
        """Contrast case -- shows the purge is doing real work."""
        n, span = 200, 10
        t1 = overlapping_touches(n, span)
        cv = V.PurgedKFold(n_splits=5, touch_idx=None, embargo_pct=0.0)
        total = sum(
            int(((tr <= te[-1]) & (t1[tr] >= te[0])).sum())
            for tr, te in cv.split(np.zeros((n, 2)))
        )
        assert total > 0

    def test_train_and_test_never_intersect(self):
        n = 120
        cv = V.PurgedKFold(n_splits=4, touch_idx=overlapping_touches(n, 5))
        for train, test in cv.split(np.zeros((n, 1))):
            assert not set(train) & set(test)

    def test_test_blocks_tile_the_sample_in_order(self):
        n = 100
        cv = V.PurgedKFold(n_splits=5)
        tests = [te for _, te in cv.split(np.zeros((n, 1)))]
        assert np.array_equal(np.concatenate(tests), np.arange(n))

    def test_expanding_mode_never_trains_on_the_future(self):
        n = 100
        cv = V.PurgedKFold(n_splits=5, expanding=True)
        for train, test in cv.split(np.zeros((n, 1))):
            assert (train < test[0]).all()

    def test_get_n_splits_matches_yielded_count(self):
        cv = V.PurgedKFold(n_splits=6)
        assert cv.get_n_splits() == len(list(cv.split(np.zeros((120, 1)))))

    def test_invalid_configuration_raises(self):
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            V.PurgedKFold(n_splits=1)
        with pytest.raises(ValueError, match="embargo_pct must be >= 0"):
            V.PurgedKFold(embargo_pct=-0.1)


class TestCombinatorialPurgedCV:
    def test_split_and_path_counts(self):
        cv = V.CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        assert cv.get_n_splits() == 15
        assert cv.n_paths == 5

    def test_yields_the_advertised_number_of_splits(self):
        cv = V.CombinatorialPurgedCV(n_groups=5, n_test_groups=2)
        assert len(list(cv.split(np.zeros((100, 1))))) == cv.get_n_splits()

    def test_train_and_test_never_intersect(self):
        n = 150
        cv = V.CombinatorialPurgedCV(
            n_groups=6, n_test_groups=2, touch_idx=overlapping_touches(n, 8)
        )
        for train, test in cv.split(np.zeros((n, 1))):
            assert not set(train) & set(test)

    def test_purging_holds_across_both_test_blocks(self):
        n, span = 180, 8
        t1 = overlapping_touches(n, span)
        cv = V.CombinatorialPurgedCV(n_groups=6, n_test_groups=2, touch_idx=t1)
        for train, test in cv.split(np.zeros((n, 1))):
            # Every contiguous run inside `test` is its own block to respect.
            breaks = np.flatnonzero(np.diff(test) > 1)
            starts = np.concatenate([[test[0]], test[breaks + 1]])
            ends = np.concatenate([test[breaks], [test[-1]]])
            for s, e in zip(starts, ends):
                leaked = (train <= e) & (t1[train] >= s)
                assert not leaked.any()

    def test_invalid_configuration_raises(self):
        with pytest.raises(ValueError, match="n_groups must be >= 2"):
            V.CombinatorialPurgedCV(n_groups=1)
        with pytest.raises(ValueError, match="n_test_groups must be in"):
            V.CombinatorialPurgedCV(n_groups=4, n_test_groups=4)


class TestProbabilisticSharpe:
    def test_sharpe_equal_to_benchmark_gives_half(self):
        assert V.probabilistic_sharpe_ratio(0.1, 500, benchmark_sharpe=0.1) == pytest.approx(0.5)

    def test_more_observations_increase_confidence(self):
        low = V.probabilistic_sharpe_ratio(0.1, 100)
        high = V.probabilistic_sharpe_ratio(0.1, 10_000)
        assert high > low

    def test_negative_skew_reduces_confidence(self):
        plain = V.probabilistic_sharpe_ratio(0.2, 1000)
        skewed = V.probabilistic_sharpe_ratio(0.2, 1000, skew=-1.5)
        assert skewed < plain

    def test_fat_tails_reduce_confidence(self):
        plain = V.probabilistic_sharpe_ratio(0.2, 1000, kurtosis=3.0)
        fat = V.probabilistic_sharpe_ratio(0.2, 1000, kurtosis=12.0)
        assert fat < plain

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError, match="n_obs must be >= 2"):
            V.probabilistic_sharpe_ratio(0.1, 1)


class TestExpectedMaxSharpe:
    def test_single_trial_sets_no_bar(self):
        assert V.expected_max_sharpe(1, 0.04) == 0.0

    def test_bar_rises_with_the_number_of_trials(self):
        vals = [V.expected_max_sharpe(k, 0.04) for k in (2, 10, 100, 1000)]
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_zero_variance_sets_no_bar(self):
        assert V.expected_max_sharpe(500, 0.0) == 0.0

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError, match="n_trials must be >= 1"):
            V.expected_max_sharpe(0, 0.04)
        with pytest.raises(ValueError, match="sharpe_variance must be >= 0"):
            V.expected_max_sharpe(10, -1.0)


class TestDeflatedSharpe:
    def test_confidence_collapses_as_the_search_widens(self):
        kw = dict(observed_sharpe=0.15, n_obs=1000, sharpe_variance=0.01)
        vals = [V.deflated_sharpe_ratio(n_trials=k, **kw) for k in (1, 10, 100, 1000)]
        assert all(b <= a for a, b in zip(vals, vals[1:]))
        assert vals[0] > 0.9, "a single-trial result should survive"
        assert vals[-1] < 0.1, "the best of 1000 tries should not"

    def test_single_trial_matches_the_undeflated_statistic(self):
        dsr = V.deflated_sharpe_ratio(
            0.12, 800, n_trials=1, sharpe_variance=0.02
        )
        psr = V.probabilistic_sharpe_ratio(0.12, 800)
        assert dsr == pytest.approx(psr)


class TestMinimumTrackRecordLength:
    def test_bigger_sharpe_needs_less_data(self):
        assert V.minimum_track_record_length(0.2) < V.minimum_track_record_length(0.05)

    def test_no_edge_is_never_establishable(self):
        assert V.minimum_track_record_length(0.0) == float("inf")
        assert V.minimum_track_record_length(-0.1) == float("inf")

    def test_higher_confidence_needs_more_data(self):
        assert (
            V.minimum_track_record_length(0.1, confidence=0.99)
            > V.minimum_track_record_length(0.1, confidence=0.90)
        )

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match=r"confidence must be in \(0, 1\)"):
            V.minimum_track_record_length(0.1, confidence=1.5)


class TestProbabilityOfBacktestOverfitting:
    def test_a_persistent_edge_is_not_flagged_as_overfit(self):
        rng = np.random.default_rng(42)
        perf = rng.normal(0, 0.01, size=(600, 20))
        perf[:, 7] += 0.004  # one configuration genuinely better, in every period
        out = V.probability_of_backtest_overfitting(perf, n_splits=10)
        assert out["pbo"] < 0.05
        assert out["mean_logit"] > 0

    def test_pure_noise_is_unbiased_across_datasets(self):
        """Any single PBO is noisy; the average over datasets must sit near 0.5."""
        rng = np.random.default_rng(7)
        vals = [
            V.probability_of_backtest_overfitting(
                rng.normal(0, 0.01, size=(240, 12)), n_splits=8
            )["pbo"]
            for _ in range(24)
        ]
        assert 0.35 < float(np.mean(vals)) < 0.65

    def test_reports_the_number_of_combinations_evaluated(self):
        rng = np.random.default_rng(0)
        out = V.probability_of_backtest_overfitting(
            rng.normal(0, 0.01, size=(200, 8)), n_splits=8
        )
        assert out["n_combinations"] == 70  # C(8, 4)

    def test_accepts_a_dataframe(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.normal(0, 0.01, size=(200, 6)))
        assert 0.0 <= V.probability_of_backtest_overfitting(df, n_splits=8)["pbo"] <= 1.0

    def test_lower_is_better_flips_the_ranking(self):
        rng = np.random.default_rng(5)
        perf = rng.normal(0, 0.01, size=(300, 10))
        perf[:, 2] += 0.005
        hi = V.probability_of_backtest_overfitting(perf, n_splits=8)
        lo = V.probability_of_backtest_overfitting(perf, n_splits=8, higher_is_better=False)
        assert hi["pbo"] < lo["pbo"]

    def test_invalid_input_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="n_splits must be even"):
            V.probability_of_backtest_overfitting(rng.normal(size=(100, 5)), n_splits=7)
        with pytest.raises(ValueError, match="at least 2 configurations"):
            V.probability_of_backtest_overfitting(rng.normal(size=(100, 1)))
        with pytest.raises(ValueError, match="must be 2-D"):
            V.probability_of_backtest_overfitting(rng.normal(size=100))
