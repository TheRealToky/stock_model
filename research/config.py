"""Frozen research configuration: data split, universe, costs, seeds.

These constants are fixed BEFORE any modeling and are imported by every
research script so runs are reproducible and the test window is never
accidentally touched during development.
"""
from __future__ import annotations

SEED = 42

# ---------------------------------------------------------------------------
# Data split (locked up front). 1-min store spans 2021-01-04 .. 2026-04-29
# (the partial 2026-04-30 day is dropped by reslib.load_ohlcv).
# ---------------------------------------------------------------------------
# Develop / tune / walk-forward here:
TRAINVAL_START = "2021-01-04"
TRAINVAL_END = "2025-04-30"      # inclusive
# Held out; touched exactly once at the very end:
TEST_START = "2025-05-01"
TEST_END = "2026-04-29"          # inclusive

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
PRIMARY = "AAPL"
# Liquid, full-history (~520k 1-min bars each) names for robustness checks.
UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "AMD", "NFLX"]

# ---------------------------------------------------------------------------
# Cost sensitivity grid (per-leg commission, per-leg slippage), in fractions.
# The lab default (0.0010, 0.0005) -> 30 bps round-trip is the pessimistic end
# (retail-tier). For liquid mega-caps a few bp round-trip is realistic.
# ---------------------------------------------------------------------------
COST_GRID = [
    # (commission_per_leg, slippage_per_leg, label, round_trip_bps)
    (0.0,     0.0,     "frictionless",      0.0),
    (0.00005, 0.00005, "ultra_low_1bp",     2.0),
    (0.0001,  0.0001,  "low_2bp",           4.0),
    (0.00025, 0.00025, "mid_5bp",          10.0),
    (0.0005,  0.0005,  "high_10bp",        20.0),
    (0.001,   0.0005,  "lab_default_15bp", 30.0),
]

# Convenience: the lab default used for headline net-of-cost numbers.
DEFAULT_COST = (0.001, 0.0005)
