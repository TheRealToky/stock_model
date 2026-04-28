"""ETL pipeline orchestrator.

Composes the extract / transform / load layers and adds the cross-cutting
concerns: per-ticker iteration, warm-up handling, manifest updates,
incremental high-water marks, and optional process-level parallelism.

Two entry points:

  * :meth:`ETLPipeline.run_full` -- rebuild a dataset from scratch (or
    extend it) for the configured ticker universe and date window.
  * :meth:`ETLPipeline.run_incremental` -- consult the manifest, then
    extract/transform/load only the bars that arrived since the last run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from financials.etl_pipeline.config.etl_config import ETLConfig
from financials.etl_pipeline.extract.timescale import TimescaleExtractor
from financials.etl_pipeline.load.parquet_writer import ParquetLoader
from financials.etl_pipeline.transform.feature_transformer import FeatureTransformer
from financials.etl_pipeline.utils.logging import configure_logging, get_logger
from financials.etl_pipeline.utils.manifest import Manifest
from financials.etl_pipeline.utils.parallel import map_tickers
from financials.etl_pipeline.utils.validation import (
    OHLCVValidationError,
    validate_ohlcv_chunk,
)

logger = get_logger(__name__)


@dataclass
class RunSummary:
    """Outcome of a single ETLPipeline run."""

    tickers_attempted: int
    tickers_succeeded: int
    rows_written: int
    skipped: list[str]
    failed: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tickers_attempted": self.tickers_attempted,
            "tickers_succeeded": self.tickers_succeeded,
            "rows_written": self.rows_written,
            "skipped": self.skipped,
            "failed": self.failed,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ETLPipeline:
    """Orchestrates extract → transform → load for a ticker universe.

    The pipeline is intentionally instantiated *once per run*. Internally
    it is stateless beyond the manifest, so re-running the same instance
    is identical to constructing a new one.
    """

    def __init__(self, cfg: ETLConfig) -> None:
        self.cfg = cfg
        configure_logging(cfg.runtime.log_level, json_mode=cfg.runtime.log_json)

        # Components are created lazily so child workers can rebuild them
        # in their own address space (DB engines aren't fork-safe).
        self._extractor: TimescaleExtractor | None = None
        self._transformer: FeatureTransformer | None = None
        self._loader: ParquetLoader | None = None

        # Manifest stays in the parent process. Workers report their
        # results back as plain dicts and the parent merges them.
        Path(cfg.load.output_path).mkdir(parents=True, exist_ok=True)
        self.manifest = Manifest(cfg.manifest_path)

    # ------------------------------------------------------------------
    # Component access (lazy)
    # ------------------------------------------------------------------

    @property
    def extractor(self) -> TimescaleExtractor:
        if self._extractor is None:
            self._extractor = TimescaleExtractor(self.cfg.extract)
        return self._extractor

    @property
    def transformer(self) -> FeatureTransformer:
        if self._transformer is None:
            self._transformer = FeatureTransformer(
                self.cfg.features,
                cast_float32=self.cfg.load.cast_float32,
                validate=not self.cfg.runtime.skip_validation,
            )
        return self._transformer

    @property
    def loader(self) -> ParquetLoader:
        if self._loader is None:
            self._loader = ParquetLoader(self.cfg.load)
        return self._loader

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run_full(self) -> RunSummary:
        """Process every configured ticker over the full configured window."""
        tickers = self.extractor.list_tickers()
        if not tickers:
            logger.warning("No tickers found -- nothing to do.")
            return RunSummary(0, 0, 0, [], [])

        start = _parse_date(self.cfg.extract.start_date)
        end = _parse_date(self.cfg.extract.end_date) if self.cfg.extract.end_date else None

        return self._run(tickers=tickers, start=start, end=end, mode="full")

    def run_incremental(self) -> RunSummary:
        """Pick up where the manifest left off for each known ticker.

        Tickers that have never been processed are seeded with the
        configured ``extract.start_date``.
        """
        tickers = self.extractor.list_tickers()
        if not tickers:
            logger.warning("No tickers found -- nothing to do.")
            return RunSummary(0, 0, 0, [], [])

        starts: dict[str, datetime] = {}
        fallback_start = _parse_date(self.cfg.extract.start_date)
        for ticker in tickers:
            hw = self.manifest.get_high_water(ticker)
            starts[ticker] = (hw + timedelta(microseconds=1)) if hw else fallback_start

        return self._run(tickers=tickers, start=starts, end=None, mode="incremental")

    # ------------------------------------------------------------------
    # Worker / driver
    # ------------------------------------------------------------------

    def _run(
        self,
        *,
        tickers: list[str],
        start: datetime | dict[str, datetime],
        end: datetime | None,
        mode: str,
    ) -> RunSummary:
        logger.info(
            "Starting {} ETL run: {} ticker(s), output={}",
            mode, len(tickers), self.cfg.load.output_path,
        )

        # Snapshot config so we don't pickle the whole pipeline into workers.
        cfg = self.cfg
        warmup_rows = cfg.extract.warmup_rows

        def _job(ticker: str) -> dict[str, Any]:
            ticker_start = (
                start[ticker] if isinstance(start, dict) else start
            )
            return _process_ticker(
                cfg=cfg,
                ticker=ticker,
                start=ticker_start,
                end=end,
                warmup_rows=warmup_rows,
            )

        results = map_tickers(
            _job, tickers, max_workers=cfg.runtime.num_workers,
        )

        # Merge results into manifest in the parent process.
        rows_written = 0
        succeeded: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []
        for r in results:
            ticker = r["ticker"]
            if r["status"] == "ok":
                rows_written += r["rows_written"]
                if r["last_timestamp"] is not None:
                    self.manifest.update(
                        ticker,
                        last_timestamp=r["last_timestamp"],
                        rows_written=r["rows_written"],
                    )
                if r["rows_written"] > 0:
                    succeeded.append(ticker)
                else:
                    skipped.append(ticker)
            elif r["status"] == "skip":
                skipped.append(ticker)
            else:
                failed.append(ticker)

        self.manifest.save()

        summary = RunSummary(
            tickers_attempted=len(tickers),
            tickers_succeeded=len(succeeded),
            rows_written=rows_written,
            skipped=skipped,
            failed=failed,
        )

        logger.info(
            "{} run complete: {} ok, {} skipped, {} failed, {} rows written",
            mode, len(succeeded), len(skipped), len(failed), rows_written,
        )
        return summary


# ---------------------------------------------------------------------------
# Worker function (top-level so multiprocessing can pickle it)
# ---------------------------------------------------------------------------


def _process_ticker(
    *,
    cfg: ETLConfig,
    ticker: str,
    start: datetime,
    end: datetime | None,
    warmup_rows: int,
) -> dict[str, Any]:
    """Run one full extract → transform → load cycle for a ticker.

    Each worker rebuilds its own extractor/transformer/loader so that DB
    engines and arrow buffers stay process-local. Returns a small dict
    that the parent can merge into the manifest.
    """
    # Fresh logging config in the worker process.
    configure_logging(cfg.runtime.log_level, json_mode=cfg.runtime.log_json)

    extractor = TimescaleExtractor(cfg.extract)
    transformer = FeatureTransformer(
        cfg.features,
        cast_float32=cfg.load.cast_float32,
        validate=not cfg.runtime.skip_validation,
    )
    loader = ParquetLoader(cfg.load)

    # Pull a small look-back buffer of OHLCV rows so the rolling indicators
    # have warmed up by the time the cursor reaches `start`.
    buffer_start = _pre_warmup_start(extractor, ticker, start, cfg.extract.interval, warmup_rows)
    cutoff_ts = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")

    rows_written = 0
    last_ts: datetime | None = None
    seen_any = False

    try:
        for chunk in extractor.stream_chunks(ticker, buffer_start, end):
            seen_any = True
            try:
                if not cfg.runtime.skip_validation:
                    validate_ohlcv_chunk(chunk, ticker=ticker)
            except OHLCVValidationError as exc:
                logger.warning("Validation failed for {}: {}", ticker, exc)
                continue

            features = transformer.transform(
                chunk, ticker=ticker, warmup_cutoff=cutoff_ts,
            )
            if features.empty:
                continue

            rows_written += loader.write(features, ticker=ticker)
            last_ts = features["timestamp"].max().to_pydatetime()

            # Slide the cutoff forward so subsequent chunks from THIS ticker
            # don't re-emit overlap rows.
            cutoff_ts = features["timestamp"].max() + pd.Timedelta("1us")

    except Exception:
        logger.exception("Unhandled error processing {}", ticker)
        return {"ticker": ticker, "status": "fail", "rows_written": 0, "last_timestamp": None}

    if not seen_any:
        return {"ticker": ticker, "status": "skip", "rows_written": 0, "last_timestamp": None}

    return {
        "ticker": ticker,
        "status": "ok",
        "rows_written": rows_written,
        "last_timestamp": last_ts,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(value: str | datetime) -> datetime:
    """Parse YYYY-MM-DD strings (or pass datetimes through) as tz-aware UTC."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _pre_warmup_start(
    extractor: TimescaleExtractor,
    ticker: str,
    desired_start: datetime,
    interval: str,
    warmup_rows: int,
) -> datetime:
    """Move the extract start backwards by enough bars to warm up indicators.

    Estimates the bar duration from ``interval`` (no DB round-trip needed)
    and subtracts that × warmup_rows. This is conservative -- if there are
    weekend/holiday gaps the actual warm-up will be slightly *more* than
    requested, which is fine.
    """
    bar = _interval_to_timedelta(interval)
    return desired_start - bar * warmup_rows


def _interval_to_timedelta(interval: str) -> timedelta:
    """Convert an interval string like ``"1min"`` / ``"1h"`` / ``"1d"``."""
    spec = interval.strip().lower()
    if spec.endswith("min"):
        return timedelta(minutes=int(spec[:-3]))
    if spec.endswith("h"):
        return timedelta(hours=int(spec[:-1]))
    if spec.endswith("d"):
        return timedelta(days=int(spec[:-1]))
    if spec.endswith("s"):
        return timedelta(seconds=int(spec[:-1]))
    # Unknown -> default to 1 minute, the dataset's native resolution.
    return timedelta(minutes=1)
