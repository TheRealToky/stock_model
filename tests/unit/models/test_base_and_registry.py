"""Tests for the model layer's Phase 2 changes.

Two defects these pin down:

* ``evaluate()`` used to report accuracy and stop, so a model could ship
  without anyone noticing it had never been backtested. The financial slot is
  now always present -- filled with a refusal when prices are missing.
* ``ModelRegistry.register()`` upserted on ``model_name``, so every retrain
  destroyed the previous run's metrics. There is now a version per run.
"""

import numpy as np
import pandas as pd
import pytest

from models.base import NO_FINANCIAL_METRICS, TASKS, BaseModel
from models.registry import _artifact_suffix, _file_sha256


class _FakeModel(BaseModel):
    """Minimal concrete model driven by a caller-supplied score."""

    def __init__(self, name="fake", task="binary", scores=None, model_type="fake"):
        self.name = name
        self.task = task
        self.model_type = model_type
        self._scores = scores

    def train(self, X, y):  # pragma: no cover - not exercised
        pass

    def predict(self, X):
        if self.task == "regression":
            return np.asarray(self._scores, dtype=float)
        return (np.asarray(self._scores, dtype=float) > 0.5).astype(int)

    def predict_proba(self, X):
        p = np.asarray(self._scores, dtype=float)
        return np.column_stack([1 - p, p])

    def get_hyperparameters(self):
        return {"k": 1}

    def save(self, path):
        with open(path, "wb") as fh:
            fh.write(b"artefact")

    @classmethod
    def load(cls, path):  # pragma: no cover - not exercised
        return cls()


def bars_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 14:30", periods=n, freq="1min", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0, 3e-4, n))
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": np.full(n, 1.0)},
        index=idx,
    )


class TestTaskSupport:
    def test_tasks_cover_the_three_target_shapes(self):
        assert set(TASKS) == {"binary", "multiclass", "regression"}

    def test_default_task_is_binary(self):
        assert _FakeModel().task == "binary"

    def test_binary_reports_classification_metrics_and_auc(self):
        scores = np.linspace(0.01, 0.99, 40)
        y = (scores > 0.5).astype(int)
        out = _FakeModel(scores=scores).evaluate(pd.DataFrame(index=range(40)), y)
        assert out["task"] == "binary"
        assert out["accuracy"] == pytest.approx(1.0)
        assert out["roc_auc"] == pytest.approx(1.0)
        assert "confusion_matrix" in out

    def test_regression_reports_error_and_directional_accuracy(self):
        preds = np.array([0.01, -0.02, 0.03, -0.01])
        model = _FakeModel(task="regression", scores=preds)
        out = model.evaluate(pd.DataFrame(index=range(4)), pd.Series(preds))
        assert out["task"] == "regression"
        assert out["rmse"] == pytest.approx(0.0)
        assert out["mae"] == pytest.approx(0.0)
        assert out["directional_accuracy"] == pytest.approx(1.0)

    def test_directional_accuracy_is_independent_of_magnitude(self):
        """A regressor can nail direction while missing size, and vice versa."""
        truth = pd.Series([0.01, -0.01, 0.02, -0.02])
        preds = np.array([5.0, -5.0, 9.0, -9.0])  # right sign, absurd scale
        out = _FakeModel(task="regression", scores=preds).evaluate(
            pd.DataFrame(index=range(4)), truth
        )
        assert out["directional_accuracy"] == pytest.approx(1.0)
        assert out["rmse"] > 1.0

    def test_multiclass_skips_auc(self):
        preds = np.array([0.1, 0.9, 0.9, 0.1])
        model = _FakeModel(task="multiclass", scores=preds)
        out = model.evaluate(pd.DataFrame(index=range(4)), pd.Series([0, 1, 1, 0]))
        assert out["task"] == "multiclass"
        assert "roc_auc" not in out


class TestFinancialMetricsAreAlwaysReported:
    def test_missing_prices_produce_an_explicit_refusal_not_a_silent_gap(self):
        scores = np.linspace(0.01, 0.99, 20)
        out = _FakeModel(scores=scores).evaluate(
            pd.DataFrame(index=range(20)), (scores > 0.5).astype(int)
        )
        assert "financial_metrics" in out, "the slot must always exist"
        assert out["financial_metrics"] == NO_FINANCIAL_METRICS
        assert "NOT COMPUTED" in out["financial_metrics"]

    def test_supplying_bars_produces_real_net_of_cost_numbers(self):
        bars = bars_frame(300)
        rng = np.random.default_rng(1)
        scores = rng.random(len(bars))
        out = _FakeModel(scores=scores).evaluate(
            pd.DataFrame(index=bars.index),
            (scores > 0.5).astype(int),
            bars=bars,
            interval="1min",
        )
        fin = out["financial_metrics"]
        assert isinstance(fin, dict)
        assert "strat_sharpe_ratio" in fin and "bh_sharpe_ratio" in fin
        assert "sharpe_edge" in fin and "beats_benchmark" in fin

    def test_evaluate_strategy_can_be_called_directly(self):
        bars = bars_frame(200)
        rng = np.random.default_rng(2)
        scores = rng.random(len(bars))
        fin = _FakeModel(scores=scores).evaluate_strategy(
            bars, pd.Series(scores, index=bars.index), interval="1min"
        )
        assert fin["label"] == "fake"
        assert fin["strat_periods_per_year"] == 98_280

    def test_costs_flow_through_to_the_reported_metrics(self):
        bars = bars_frame(600)
        rng = np.random.default_rng(3)
        scores = rng.random(len(bars))
        model = _FakeModel(scores=scores)
        cheap = model.evaluate_strategy(
            bars, pd.Series(scores, index=bars.index), interval="1min",
            commission=0.0, slippage=0.0,
        )
        dear = model.evaluate_strategy(
            bars, pd.Series(scores, index=bars.index), interval="1min",
            commission=0.001, slippage=0.0005,
        )
        assert dear["strat_sharpe_ratio"] < cheap["strat_sharpe_ratio"]


class TestRegistryHelpers:
    def test_lstm_is_registered_in_the_class_map(self):
        """Without this, load_model() on any LSTM raised 'Unknown model_type'."""
        pytest.importorskip("torch")
        from models.registry import _MODEL_CLASS_MAP, _ensure_class_map

        _ensure_class_map()
        assert "lstm" in _MODEL_CLASS_MAP
        assert {"xgboost", "random_forest"} <= set(_MODEL_CLASS_MAP)

    def test_torch_models_get_a_pt_suffix(self):
        """The registry hard-coded .joblib, mislabelling every checkpoint."""
        assert _artifact_suffix(_FakeModel(model_type="lstm")) == ".pt"
        assert _artifact_suffix(_FakeModel(model_type="xgboost")) == ".joblib"

    def test_file_hash_is_stable_and_content_sensitive(self, tmp_path):
        a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
        a.write_bytes(b"same"), b.write_bytes(b"same"), c.write_bytes(b"different")
        assert _file_sha256(str(a)) == _file_sha256(str(b))
        assert _file_sha256(str(a)) != _file_sha256(str(c))
        assert len(_file_sha256(str(a))) == 64


class TestRegistrySchema:
    """The ORM must expose the columns migration 002 adds."""

    def test_versioning_columns_exist(self):
        from models.registry import ModelRegistryRow

        cols = set(ModelRegistryRow.__table__.columns.keys())
        assert {"version", "run_id", "artifact_hash", "scaler_path", "is_latest"} <= cols

    def test_model_name_is_no_longer_unique_on_its_own(self):
        """Uniqueness moved to (model_name, version) so retrains can coexist."""
        from models.registry import ModelRegistryRow

        assert ModelRegistryRow.__table__.columns["model_name"].unique is not True

    def test_migration_002_exists_and_moves_the_constraint(self):
        from pathlib import Path

        sql = Path(
            "financials/database/migrations/002_model_registry_versioning.sql"
        ).read_text(encoding="utf-8")
        assert "DROP INDEX IF EXISTS ix_model_registry_model_name" in sql
        assert "ix_model_registry_name_version" in sql
        assert "scaler_path" in sql
