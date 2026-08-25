# Can a 1-Minute Strategy Beat Net-of-Cost Buy-and-Hold? — A Research Report

**Author:** Quant research lab (educational only — no live trading, no real orders)
**Date:** 2026-06-17
**Branch:** `feat/cl-strategy`

## TL;DR (the honest answer)

**No reliable edge was found.** On 1-minute US-equity bars (2021–2026), once costs are
applied and the test window is held out:

1. **Directional timing is hopeless net of costs.** Every existing/naive 1-min strategy
   (EMA cross, mean reversion, momentum, naive reversal) and a gradient-boosted ML
   direction classifier **loses badly** at any realistic cost. The one strategy with a
   *gross* edge (mean reversion, Sharpe 0.80 vs B&H 0.40) has per-trade alpha smaller than
   a **1 bp** round-trip cost. At 1-min, **turnover is the enemy.**
2. **A volatility-targeting overlay** — the one principled, low-turnover idea — **reliably
   reduces drawdown** (8/8 tickers out-of-sample) and is **cost-insensitive** (turnover
   ≈ 0.007/day), but does **not reliably improve Sharpe out-of-sample**: the cross-sectional
   median OOS Sharpe edge is **−0.072**, and the in-sample Sharpe ranking **does not
   persist** (the two best in-sample names went negative OOS; the worst in-sample name,
   AAPL, was the best OOS). The apparent AAPL win (Sharpe 1.16 vs 0.96) is best read as a
   **single favorable draw, not a repeatable edge.**

**Conclusion:** at this frequency, nothing reliably beats net-of-cost buy-and-hold on a
risk-adjusted basis. The only durable, repeatable benefit of the overlay is **risk
reduction** (lower drawdown / volatility at roughly matched return) — a defensible
*risk-management* tool, **not** an alpha source.

---

## 1. Data, split, costs, annualization (the ground truth)

### 1.1 Data actually in the lab
Verified directly against TimescaleDB and the on-disk Parquet feature store (not assumed):

| Item | Value |
|---|---|
| 1-min OHLCV rows | 61,581,213 across **126 tickers** |
| 1-min span | 2020-12-14 → **2026-04-30** (last day partial, ends 13:59 UTC → **dropped**) |
| Bars/ticker | ~520k (e.g. AAPL 519,692) |
| Session calendar | **390 bars/day** (AAPL median=max=390; 1,338 trading days) = **US regular hours only** (09:30–16:00 ET), no extended/24h. UTC hours 13–20 confirm RTH across DST. |
| Feature store | Hive-partitioned Parquet, 2021-01-04 → 2026-04-30, OHLCV + `adjusted_close` + 29 technical features, float32, tz-aware UTC, clean 1-min within-day continuity (only overnight gaps). |

Source: twelvedata 1-min. Prices are real (AAPL ≈ $130 in Jan-2021, post-2020 split).

### 1.2 Universe
- **Primary development ticker:** `AAPL`.
- **Robustness universe (8 liquid mega-caps, ~520k bars each):** AAPL, MSFT, NVDA, AMZN,
  META, TSLA, AMD, NFLX.

### 1.3 Split (locked in `research/config.py` *before* any modeling)
| Split | Window | Use |
|---|---|---|
| **Train + Val** | 2021-01-04 → 2025-04-30 (~4.3 yr) | EDA, tuning, walk-forward |
| **Test (sealed)** | 2025-05-01 → 2026-04-29 (~12 mo) | touched **once**, at the very end |

### 1.4 Costs (per leg; swept)
`frictionless (0)`, `1bp`, `2bp`, `5bp`, `10bp`, and the **lab default `commission=0.10% / slippage=0.05%` ≈ 30 bp round-trip** (retail-tier; the pessimistic end). Every reported number is **net of commission + slippage**, with B&H run through the identical cost model.

### 1.5 Annualization (a critical fix — see §6 Gotchas)
1-min US-RTH bars ⇒ **periods/year = 390 × 252 = 98,280**. The stock `BacktestEngine`
calls `compute_all_metrics` **without** `periods`, defaulting to 252 (daily), which
**overstates Sharpe/Sortino by √(98280/252) ≈ 19.7×**. All metrics here use 98,280 and the
fix was validated (§6).

---

## 2. Methodology & evaluation layer

A trusted evaluation library (`research/lib/reslib.py`) was built and **validated against
the canonical event-driven `BacktestEngine`** before any result was trusted:

- **Equity-path agreement:** correlation **1.000000** vs the engine for EMA / mean-reversion
  / momentum; terminal equity within **0.5–0.9%** (whole-share rounding only).
- **Metric parity:** reslib reproduces the engine Sharpe **exactly** at `periods=252`
  (−5.6842 = −5.6842), and gives −29.65 at the correct 98,280.
- **Speed:** 74× faster than the engine loop (0.024 s vs 1.808 s on 503k bars) → thousands
  of sweep runs are cheap.
- **Apples-to-apples B&H:** buy-and-hold is simulated through the *same* cost model and
  execution convention (decide at bar *i−1*, execute at bar *i* open, mark at close).
- **Alpha/beta** are computed against the asset's own close-to-close returns (not the
  strategy's own equity, which is the engine's bug).
- **Continuous-exposure extension** (for vol targeting) is a documented sizing extension;
  with exposure ≡ 1 it reproduces B&H exactly (total return 0.5960 = 0.5960).

No-look-ahead discipline: signals/features use only information through bar *i−1*; the
simulator shifts every position by one bar; the vol overlay's target is a **past-only
expanding quantile**.

---

## 3. EDA — forming hypotheses (train+val only)

(`research/scripts/p1_eda.py`, plot `research/outputs/p1_eda.png`)

1. **Direction is ~unpredictable at 1-min.** Within-day 1-min return autocorrelation is
   tiny and **sign-inconsistent** across the universe (lag-1 mean **−0.0055**, range −0.026
   to +0.008). Variance ratios for AAPL are ≈1 at every horizon (VR(2)=1.000, VR(10)=0.992,
   VR(60)=0.988) → essentially a **random walk**.
2. **Volatility is highly predictable.** |return| autocorrelation decays slowly
   (0.32 → 0.17 over 60 lags) vs return ACF ≈ 0 → strong **volatility clustering**.
3. **Strong U-shaped intraday seasonality.** AAPL mean |1-min return| ≈ 12–16 bps near the
   open, ~3.5 bps midday, ~8 bps into the close; volume mirrors it (open/close ≫ midday).
4. **Overnight vs intraday is heterogeneous and not exploitable.** Session Sharpes differ in
   sign across names (AAPL favors intraday 0.84 vs overnight −0.67; NVDA favors overnight
   1.09). Crucially the **full B&H Sharpe (avg 0.469) exceeds either component**
   (intraday 0.247, overnight 0.286) — splitting sessions destroys diversification, so
   session-timing does not beat B&H.

**Implication:** do not time direction at 1-min. The only defensible angle is a
**low-turnover volatility overlay** that exploits (2)–(3) for *risk management*.

---

## 4. Results

### 4.1 Baselines confirm the cost wall (AAPL, train+val)
B&H AAPL (train+val): **Sharpe 0.397, maxDD 0.349, total return 0.596.**

| Strategy | Gross Sharpe | Net @2bp | Net @30bp (lab) | Trades/day |
|---|---:|---:|---:|---:|
| **mean_reversion(20,2,.5)** | **+0.795** (beats B&H, DD 0.18) | −4.42 | −32.7 | 7.4 |
| naive_reversal_1bar | +0.399 | −48.0 | −239 | **95.6** |
| naive_momentum_1bar | −0.020 | −48.1 | −239 | 95.7 |
| ema_crossover(12,26) | −0.423 | −3.89 | −24.3 | 6.7 |
| sma_trend(5,30) | −0.307 | −5.08 | −32.3 | 9.3 |

Only **1 of 30** strategy×cost configs beats B&H, and it is the unrealistic frictionless
mean-reversion case. **Mean reversion has a genuine gross edge that is entirely consumed by
even a 1 bp round-trip cost.**

### 4.2 ML direction classifier (XGBoost) — a clean negative
(train 2021–23, val 2024–25-04 for threshold, sealed test; `p4_ml_route.py`)

| | AUC |
|---|---:|
| train | 0.581 |
| validation | **0.503** |
| **test** | **0.506** |

Out-of-sample AUC ≈ 0.50 ⇒ **no directional skill**; P(up) is pinned at ~0.49. With the
probability→signal adapter (long if P(up) > threshold tuned on validation), the sealed-test
result is: frictionless Sharpe 1.21 (but worse DD than B&H, so it fails the criterion); at
**2bp Sharpe −3.67**; at **30bp Sharpe −32.1** — destroyed by 11 trades/day.

### 4.3 Volatility-targeting overlay (the candidate)

Rule (frozen in `research/lib/overlays.py` before the test was touched): exposure
`e_d = clip(target_d / pred_vol_d, 0, 1)`, unlevered, rebalanced daily, where
`pred_vol_d = EMA₂₂(realized vol from 1-min returns)` (known at the prior close) and
`target_d` = **expanding median** of predicted vol (past-only). De-risk in turbulent
regimes, hold fully otherwise.

**AAPL, full metric comparison (2 bp round-trip), Sharpe @ 98,280:**

| | Sharpe | Sortino | maxDD | CAGR | TotRet | AnnVol | Alpha(ann) | Beta | AvgExpo | Turn/day |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **IS** overlay | 0.388 | 0.388 | 0.283 | 0.112 | 0.551 | 0.253 | 0.005 | 0.854 | 0.929 | 0.007 |
| **IS** B&H | 0.397 | 0.398 | 0.349 | 0.120 | 0.596 | 0.290 | 0.000 | 1.000 | 1.000 | 0.001 |
| **OOS** overlay | **1.164** | 1.235 | **0.155** | 0.315 | **0.309** | 0.222 | 0.053 | 0.896 | 0.935 | 0.007 |
| **OOS** B&H | 0.956 | 1.005 | 0.155 | 0.276 | 0.272 | 0.245 | 0.000 | 1.000 | 1.000 | 0.000 |

*(Overlay is continuous-exposure, so "#trades / avg-holding" are replaced by average
exposure and daily turnover. Turnover ≈ 0.007/day ⇒ negligible cost.)*

**Cross-ticker out-of-sample (sealed test, 2 bp), overlay vs B&H:**

| Lens | Result |
|---|---|
| maxDD ≤ B&H | **8 / 8** |
| Sharpe edge > 0 | **2 / 8** (AAPL +0.21, AMD +0.09) |
| beats B&H (Sharpe **and** DD) | **2 / 8** |
| median OOS Sharpe edge | **−0.072** |
| median OOS DD reduction | 0.003 (tiny — calm bull test regime) |

**The persistence test fails:** the in-sample Sharpe leaders **NVDA (+0.145)** and **META
(+0.130)** both went **negative OOS** (−0.028, −0.097); the in-sample laggard **AAPL
(−0.009)** was the **best OOS (+0.209)**. In-sample ranking carries **no information** about
out-of-sample Sharpe ⇒ the Sharpe "edge" is noise.

### 4.4 Cost sensitivity (sealed test) — the overlay's one structural win
Because turnover ≈ 0.007/day, the overlay is **near cost-invariant**: AAPL OOS Sharpe edge
moves only 0.210 → 0.198 as round-trip cost goes 0 → 30 bp. (Contrast the HF strategies in
§4.1, which swing from positive gross to catastrophic at 2 bp.) The universe-median edge
stays ≈ −0.07 across **all** cost levels — i.e., the negative verdict is robust to cost
assumptions, not an artifact of one cost choice.

### 4.5 Walk-forward / per-year stability (AAPL, train+val)
The locked (no-look-ahead) overlay **underperforms B&H on Sharpe in every train+val year**
(edge −0.07, −0.11, −0.03, −0.04, −0.78) while **reducing maxDD** (e.g. 2022 bear:
0.267 vs 0.311). Consistent with §4.3: drawdown reduction is the stable effect; Sharpe
improvement is not.

### Plots (all net of costs)
- `research/outputs/p1_eda.png` — return ACF, intraday vol/volume U-shape, |ret| ACF.
- `research/outputs/p5_aapl_oos.png` — AAPL **out-of-sample** equity + drawdown, overlay vs B&H.
- `research/outputs/p5_crossticker_oos.png` — cross-ticker OOS Sharpe edge and DD reduction.

---

## 5. What didn't work, and why

- **Mean reversion / reversal at 1-min.** Real *gross* edge (negative microstructure
  autocorrelation), but per-trade alpha < 1 bp while it trades 7–96×/day ⇒ costs win.
  `(1−0.003)^≈800 ≈ 0.09` — ~90% of capital lost to costs over six months at lab default.
- **EMA crossover / SMA trend.** Negative even *gross* intraday on these names; trend-
  following has nothing to follow at 1-min. Costs make it worse.
- **Session timing (overnight-only / intraday-only).** Sign of the better session is
  inconsistent across names and B&H beats either component on Sharpe → no generalizable rule.
- **ML direction classifier.** OOS AUC 0.503/0.506 — overfits training noise, no real skill;
  net-of-cost it is wiped out by turnover. A textbook confirmation that 1-min direction is
  unpredictable here.
- **Vol-target overlay for Sharpe.** Reduces drawdown reliably, but the Sharpe gain is small,
  name-dependent, and **does not persist** in-sample → out-of-sample. The honest verdict is
  "risk reducer, not alpha."

### A methodological catch worth flagging
In the *first* (exploratory) overlay pass, the target was a **global in-sample quantile** of
predicted vol — a subtle look-ahead — which showed AAPL Sharpe **+0.028**. Re-running with a
**past-only expanding quantile** flipped it to **−0.009**. The "edge" was partly the
look-ahead. This is exactly why the locked, no-look-ahead rule and the sealed test matter.

---

## 6. Codebase gotchas verified / fixed

| Gotcha | Status |
|---|---|
| **Annualization** (engine defaults to 252 for 1-min) | **Confirmed & fixed**: use 98,280; reslib matches engine exactly @252, corrects @98,280 (19.7× factor verified). |
| **Engine is long/flat, whole-share, all-in/out** | Confirmed; reslib mirrors it. Vol overlay needs sizing → **continuous-exposure extension documented** and validated (e≡1 ⇒ B&H). |
| **Alpha/beta computed vs strategy's own equity** | Confirmed bug; reslib computes alpha/beta vs the asset's close-to-close returns instead. |
| **ML→signal bridge missing** | Built: probability→long/flat adapter, threshold tuned on validation only. |
| **Alt-data (flights) is ~daily** | Not used as a 1-min signal (would be dishonest); noted as a low-frequency regime overlay only. |
| **`Momentum` expanding-quantile** | O(n²); impractical at 406k bars — itself a finding. Excluded from the 1-min sweep; vectorized naive momentum used instead. |

---

## 7. Reproducibility

- **Config (frozen):** `research/config.py` (seed=42, split dates, universe, cost grid).
- **Locked strategy logic:** `research/lib/overlays.py` (span=22, target_q=0.50, min_days=60).
- **Eval library:** `research/lib/reslib.py` (validated against `BacktestEngine`).
- **Scripts (run in the `python-app` container, in order):**
  `p0_db_inventory.py`, `p0_featurestore_probe.py`, `p0_validate_engine.py`, `p1_eda.py`,
  `p2_baselines.py`, `p3_voltarget_explore.py`, `p3b_voltarget_robust.py`,
  `p4_ml_route.py`, `p5_final_test.py`, `p6_register.py`.
- **Outputs:** `research/outputs/*.csv`, `*.txt`, `*.png`.
- **DB deliverables:** `strategy_registry` now contains `vol_target_overlay` and
  `buy_and_hold`; `backtest_results` has AAPL IS/OOS rows for both (see §8).
- **Env:** added `pyarrow`, `duckdb` to `requirements.txt` (feature-store reads).

Run example:
```bash
docker compose up -d timescaledb python-app
docker compose exec python-app python research/scripts/p5_final_test.py
```

## 8. Deliverables checklist

- [x] Strategies registered in `strategy_registry` (`vol_target_overlay`, `buy_and_hold`).
- [x] `backtest_results` rows (AAPL, in-sample & sealed out-of-sample, overlay & B&H).
- [x] Comparison tables strategy vs B&H (Sharpe, Sortino, maxDD, CAGR, total return,
      turnover/exposure), in-sample **and** out-of-sample.
- [x] Equity-curve and drawdown plots, net of costs.
- [x] Cost-sensitivity sweep.
- [x] Walk-forward / per-year + cross-ticker robustness.
- [x] Honest "what didn't work" section.
- [x] Reproducible config (seeds, params, date ranges, data versions).

**Final verdict:** a defensible, reproducible, out-of-sample comparison shows that **no
strategy reliably beats net-of-cost buy-and-hold at 1-minute frequency** on these
instruments. The volatility-targeting overlay is a genuine, cost-robust **drawdown reducer**
but not a reliable source of risk-adjusted outperformance.
