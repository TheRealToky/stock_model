"""Leakage guards on the feature-store reader.

Every column in the Parquet feature store is backward-looking. Windows end
at bar ``i - 1`` and the label is read from that same bar, so a target that
is also an input hands the model its answer at the final timestep. These
tests pin the guard that refuses that configuration.

Requires duckdb (and loguru), which the reader imports at module scope, so
the whole module skips on a bare interpreter and runs inside the container.
"""

import pytest

pytest.importorskip("duckdb", reason="reader imports duckdb at module scope")
pytest.importorskip("loguru", reason="reader logging shim requires loguru")

from financials.etl_pipeline.load.reader import (  # noqa: E402
    MLDataLoader,
    StreamingSequenceDataset,
    _reject_leaky_target,
)

FEATURES = ["log_returns", "rsi", "macd_macd", "atr"]


class TestRejectLeakyTarget:
    def test_target_that_is_also_a_feature_raises(self):
        # "rsi" is not on the denylist, so this isolates the overlap check.
        with pytest.raises(ValueError, match="leaks the label"):
            _reject_leaky_target("rsi", FEATURES)

    def test_price_direction_is_rejected_when_used_as_a_feature(self):
        """Contemporaneous: the denylist catches it before the overlap check."""
        with pytest.raises(ValueError, match="not a forward-looking label"):
            _reject_leaky_target("price_direction", [*FEATURES, "price_direction"])

    def test_disjoint_target_is_allowed(self):
        _reject_leaky_target("fwd_direction_1", FEATURES)

    def test_none_target_is_allowed(self):
        """target_fn users pass target_column=None."""
        _reject_leaky_target(None, FEATURES)


class TestStreamingSequenceDatasetConstruction:
    def _loader(self, tmp_path):
        return MLDataLoader(str(tmp_path))

    def test_leaky_target_column_raises_at_construction(self, tmp_path):
        with pytest.raises(ValueError, match="leaks the label"):
            StreamingSequenceDataset(
                loader=self._loader(tmp_path),
                scaler=object(),
                feature_columns=FEATURES,
                target_column="rsi",
            )

    def test_requires_exactly_one_target_specifier(self, tmp_path):
        with pytest.raises(ValueError, match="exactly one"):
            StreamingSequenceDataset(
                loader=self._loader(tmp_path),
                scaler=object(),
                feature_columns=FEATURES,
            )
        with pytest.raises(ValueError, match="exactly one"):
            StreamingSequenceDataset(
                loader=self._loader(tmp_path),
                scaler=object(),
                feature_columns=FEATURES,
                target_column="fwd_direction_1",
                target_fn=lambda df: df["close"],
            )

    def test_target_source_columns_requires_target_fn(self, tmp_path):
        with pytest.raises(ValueError, match="only applies alongside target_fn"):
            StreamingSequenceDataset(
                loader=self._loader(tmp_path),
                scaler=object(),
                feature_columns=FEATURES,
                target_column="fwd_direction_1",
                target_source_columns=["close"],
            )

    def test_forward_shifted_target_fn_is_accepted(self, tmp_path):
        ds = StreamingSequenceDataset(
            loader=self._loader(tmp_path),
            scaler=object(),
            feature_columns=FEATURES,
            target_fn=lambda df: (df["close"].shift(-1) > df["close"]).astype("float32"),
            target_source_columns=["close"],
        )
        assert ds._target_source_columns == ["close"]

    def test_iter_sequences_rejects_leaky_target(self, tmp_path):
        loader = self._loader(tmp_path)
        with pytest.raises(ValueError, match="leaks the label"):
            next(
                loader.iter_sequences(
                    sequence_length=10,
                    feature_columns=FEATURES,
                    target_column="rsi",
                )
            )


class TestNonPredictiveTargets:
    """Store columns that describe the present, not the future."""

    @pytest.mark.parametrize(
        "col", ["close", "log_returns", "returns", "price_direction", "volume"]
    )
    def test_backward_looking_columns_are_refused(self, col):
        with pytest.raises(ValueError, match="not a forward-looking label"):
            _reject_leaky_target(col, ["rsi", "atr"])

    def test_price_direction_is_refused_even_when_not_a_feature(self, tmp_path):
        """The trap: it reads like a ready-made label but is close_t > close_t-1."""
        with pytest.raises(ValueError, match="not a forward-looking label"):
            StreamingSequenceDataset(
                loader=MLDataLoader(str(tmp_path)),
                scaler=object(),
                feature_columns=["rsi", "atr"],  # deliberately excluded
                target_column="price_direction",
            )

    def test_a_genuinely_forward_looking_column_passes(self):
        _reject_leaky_target("fwd_return_5m", ["rsi", "atr"])
