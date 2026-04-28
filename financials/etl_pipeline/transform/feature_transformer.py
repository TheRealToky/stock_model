"""Feature engineering transformer.

Wraps :class:`financials.features.engine.FeatureEngine` so the orchestrator
sees a small, stable interface and so we can plug in look-ahead-safe
warm-up handling here without polluting the engine itself.
"""

from __future__ import annotations

import pandas as pd

from financials.etl_pipeline.config.etl_config import FeatureConfig
from financials.etl_pipeline.features.selector import (
    expected_feature_columns,
    resolve_features,
)
from financials.etl_pipeline.utils.logging import get_logger
from financials.etl_pipeline.utils.validation import (
    cast_float32,
    validate_feature_frame,
)
from financials.features.engine import FeatureEngine

logger = get_logger(__name__)


class FeatureTransformer:
    """Compute features on a chunk while honouring rolling-window warmup.

    The transformer is *stateless* with respect to data: every call gets a
    fresh DataFrame and returns a new one. State that belongs in this
    class is purely configuration (which features, which columns).
    """

    def __init__(
        self,
        cfg: FeatureConfig,
        *,
        cast_float32: bool = True,
        validate: bool = True,
    ) -> None:
        self.cfg = cfg
        self._engine = FeatureEngine()
        self.feature_names = resolve_features(cfg)
        self.feature_columns = expected_feature_columns(self.feature_names)
        self._cast_float32 = cast_float32
        self._validate = validate

        logger.info(
            "FeatureTransformer initialised: {} features → {} output columns",
            len(self.feature_names),
            len(self.feature_columns),
        )

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def transform(
        self,
        ohlcv: pd.DataFrame,
        *,
        ticker: str,
        warmup_cutoff: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Compute features for *ohlcv* and return a tidy frame.

        Parameters
        ----------
        ohlcv:
            DataFrame indexed by tz-aware ``timestamp`` with the standard
            OHLCV columns. Must already include any prepended warm-up rows
            needed for rolling windows.
        ticker:
            Ticker symbol. Added as a ``symbol`` column for downstream
            partitioning and used in error messages.
        warmup_cutoff:
            If provided, rows whose index is strictly less than this
            timestamp are dropped from the *result* (not from the input
            used for computation). This is how we feed warm-up rows to the
            indicators without leaking them into the dataset.

        Returns
        -------
        pd.DataFrame
            Index reset to a column. Schema:
            ``timestamp, symbol, open, high, low, close, volume,
            adjusted_close, <feature columns...>``.
        """
        if ohlcv.empty:
            return _empty_output_frame(self.feature_columns)

        # Compute features over the full input (warm-up + payload) so the
        # rolling state is correct at the cutoff.
        with_features = self._engine.compute_features(ohlcv, self.feature_names)

        # Drop the warm-up region.
        if warmup_cutoff is not None:
            with_features = with_features[with_features.index >= warmup_cutoff]

        if with_features.empty:
            return _empty_output_frame(self.feature_columns)

        # Drop rows where any enabled feature is still NaN. This typically
        # only happens if warmup_cutoff was set too aggressively or the
        # window is longer than warmup_rows.
        if self.cfg.drop_warmup_nans and self.feature_columns:
            present = [c for c in self.feature_columns if c in with_features.columns]
            if present:
                with_features = with_features.dropna(subset=present, how="any")

        if with_features.empty:
            return _empty_output_frame(self.feature_columns)

        # Promote the index to a column for parquet, add metadata.
        out = with_features.reset_index().rename(columns={"index": "timestamp"})
        if "timestamp" not in out.columns and "Timestamp" in out.columns:
            out = out.rename(columns={"Timestamp": "timestamp"})

        out.insert(1, "symbol", ticker)
        out["date"] = out["timestamp"].dt.tz_convert("UTC").dt.date.astype("string")

        if self._validate:
            validate_feature_frame(
                out,
                expected_features=self.feature_columns,
                ticker=ticker,
            )

        if self._cast_float32:
            out = cast_float32(out)

        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_output_frame(feature_columns: list[str]) -> pd.DataFrame:
    """Return an empty frame with the canonical output schema."""
    cols = (
        ["timestamp", "symbol", "date", "open", "high", "low", "close", "volume", "adjusted_close"]
        + list(feature_columns)
    )
    return pd.DataFrame(columns=cols)
