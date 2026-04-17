# Term Structure Intelligence Strategy — Design Spec

**Date:** 2026-04-17
**Status:** Approved design, pending implementation
**Target audience:** Commodity desks (Citadel Commodities, Trafigura, Man AHL), multi-strat quant funds

---

## 1. Problem Statement

The current strategy suite (7 strategies, 10 factors, 19 commodities) produces no statistically significant positive Sharpe ratios out-of-sample. The best strategy (TSMOM) has full-sample Sharpe +0.02 with 95% CI [-0.41, +0.46]. The equal-weight long commodity benchmark beats every active strategy at Sharpe +0.27.

Root causes identified through systematic audit (session 9, 2026-04-13):
1. **Curve factor contamination:** All 4 curve factors (carry, slope, curvature, term_carry) have lag0 IC 3-4x larger than lag1 IC. After decontamination, lag1 IC collapses to noise. Daily curve signals are tautological — same-day price moves shift the curve.
2. **Cross-sectional ranking with 19 assets is too thin:** Academic papers showing Sharpe 0.5-1.0 use 40-60 instruments. With 19 commodities, there is not enough dispersion for cross-sectional ranking to separate signal from noise.
3. **Standard factors are fully arbitraged:** Carry and momentum premia documented in 1970-2010 data have been compressed to near-zero in post-2010 markets by CTAs and systematic commodity funds.

## 2. Core Thesis

The commodity term structure is the market's real-time consensus on physical supply/demand tightness. Instead of treating curve metrics as factors to z-score and rank cross-sectionally, build the entire strategy around **reading the curve the way a physical trader reads it**.

A physical trader sees deep backwardation and thinks: "The market is physically tight — convenience yield is high, nearby supply is scarce, I want to be long until the curve tells me supply is easing." This project systematises that reasoning.

**Key design principles:**
- **Slow signals:** Use curve information at monthly/weekly frequency to determine positioning, not as daily return predictors. This avoids the contamination problem.
- **Time-series, not cross-sectional:** Each commodity gets an independent signal based on its own curve state. No ranking across the universe.
- **Economically motivated:** Every signal is grounded in commodity market microstructure theory (Theory of Storage, cost of carry, refinery margins). No data-mined patterns.
- **Few parameters:** Each layer has 2-4 configurable parameters, all from YAML configs. Low overfitting risk.

## 3. Architecture

Three layers, each capturing a different return source, unified by the term structure thesis:

```
Layer 1: Curve-Informed Directional (40% risk budget)
  "Which commodities are physically tight? Be long those."
  Monthly rebalance. TSMOM trend filter.

Layer 2: Curve Transition Momentum (25% risk budget)
  "Which commodities are BECOMING tight or loose? Trade the transition."
  Weekly rebalance. TSMOM confirmation gate.

Layer 3: Structural Spread Alpha (35% risk budget)
  "When related commodities disagree on tightness, trade the convergence."
  Bi-weekly rebalance. Inventory conditioning.
```

The layers are roughly uncorrelated by construction (level vs derivative vs relative value), so the combined Sharpe benefits from diversification.

## 4. Layer 1 — Curve-Informed Directional

### 4.1 Convenience Yield Estimation

From the cost-of-carry model:

```
F(T) = S * exp((r - y + c) * T)
```

Rearranging:

```
y = r + c - ln(F(T) / S) / T
```

Where:
- `S` = spot proxy (F1M from existing curve data)
- `F(T)` = futures price at tenor T (use F6M as reference — enough distance to be informative, low NaN rate)
- `r` = annualised risk-free rate (DGS3MO from FRED, already downloaded)
- `c` = storage cost proxy (per-commodity constant, calibrated as the median contango depth in IS period)
- `y` = convenience yield (the output)

Compute daily, then take the **monthly median** to smooth noise.

Implementation: `curves/convenience_yield.py`
- `estimate_storage_cost(curves, is_end="2017-12-31") -> dict[str, float]` — calibrate per-commodity storage proxy
- `compute_convenience_yield(curves, risk_free, storage_costs, tenor="F6M") -> pd.DataFrame` — daily CY, (dates x commodities)
- `monthly_convenience_yield(daily_cy) -> pd.DataFrame` — monthly median aggregation

### 4.2 Curve Regime Classification

Classify each commodity-month into 5 regimes based on convenience yield percentile (expanding window, no lookahead):

| Regime | CY Percentile | Base Position | Rationale |
|--------|---------------|---------------|-----------|
| Crisis backwardation | > 90th | +1.0 (full long) | Severe scarcity, strong roll yield |
| Mild backwardation | 70th - 90th | +0.5 | Tight market, moderate edge |
| Balanced | 30th - 70th | 0.0 (flat) | No edge, stay out |
| Mild contango | 10th - 30th | 0.0 (flat) | Negative roll yield eats returns |
| Deep contango | < 10th | -0.5 (half short) | Oversupply, negative roll yield |

Implementation: `signals/curve_regime.py`
- `classify_regime(monthly_cy, thresholds=[10, 30, 70, 90]) -> pd.DataFrame` — regime labels per commodity-month
- `regime_to_position(regimes) -> pd.DataFrame` — map regimes to base position weights

### 4.3 Trend Filter

Before taking any position, gate on 12-month TSMOM (existing signal):

| Regime Position | TSMOM > 0 | TSMOM <= 0 |
|----------------|-----------|------------|
| Long (+1.0, +0.5) | Take it | Override to flat |
| Flat (0.0) | Stay flat | Stay flat |
| Short (-0.5) | Override to flat | Take it |

Rationale: Oil in H2 2014 was in backwardation but crashing 60%. The trend filter prevents catching falling knives in structurally tight but collapsing markets.

Implementation: part of `signals/directional.py`
- `apply_trend_filter(positions, tsmom) -> pd.DataFrame` — gate positions on TSMOM sign

### 4.4 Rebalance

Monthly (first business day). Positions held for the full month. This is the slowest layer — captures the commodity risk premium and convenience yield signal without daily noise.

### 4.5 Parameters (all in `configs/strategy.yaml`)

```yaml
curve_directional:
  convenience_yield_tenor: "F6M"
  regime_thresholds: [10, 30, 70, 90]
  position_map: {crisis_backwardation: 1.0, mild_backwardation: 0.5, balanced: 0.0, mild_contango: 0.0, deep_contango: -0.5}
  trend_filter: true
  trend_lookback_days: 252
  rebalance: "monthly"
  risk_budget: 0.40
```

## 5. Layer 2 — Curve Transition Momentum

### 5.1 Signal Construction

Compute the 3-month change in convenience yield z-score:

```
transition_signal(t) = cy_zscore(t) - cy_zscore(t - 63 trading days)
```

Where `cy_zscore` is the expanding-window z-score of monthly convenience yield (same as used in Layer 1 regime classification). The monthly CY is forward-filled to daily frequency so the 63-day diff can be evaluated at weekly rebalance checkpoints.

- Large positive change (> +0.5 std) = market tightening = long
- Large negative change (< -0.5 std) = market loosening = short
- Within +/- 0.5 std = no transition = flat

The threshold of 0.5 std is applied to the expanding standard deviation of the transition signal itself.

### 5.2 TSMOM Confirmation Gate

Only take positions where the transition direction agrees with TSMOM:

| Transition | TSMOM > 0 | TSMOM <= 0 |
|-----------|-----------|------------|
| Tightening (long signal) | High conviction long | Flat (conflicting) |
| No transition | Flat | Flat |
| Loosening (short signal) | Flat (conflicting) | High conviction short |

### 5.3 Rebalance

Weekly (every Friday, execute Monday — consistent with existing execution lag). Faster than Layer 1 because transitions are more dynamic than regime levels.

### 5.4 Parameters

```yaml
curve_transition:
  lookback_days: 63
  threshold_std: 0.5
  tsmom_confirmation: true
  rebalance: "weekly"
  risk_budget: 0.25
```

### 5.5 Implementation

`signals/curve_transition.py`
- `compute_transition_signal(monthly_cy, lookback=63) -> pd.DataFrame` — 3-month CY z-score change
- `transition_to_position(transition, tsmom, threshold=0.5) -> pd.DataFrame` — position weights with confirmation gate

## 6. Layer 3 — Structural Spread Alpha

### 6.1 Energy Complex — Convenience Yield Crack Spread

Standard crack spread (`RB + HO - CL`) is well-known. The differentiated version uses **convenience yield divergence**:

```
cy_crack(t) = cy(RB, t) + cy(HO, t) - cy(CL, t)
```

When CL convenience yield is high relative to products (cy_crack is negative) -> crude is tight but products aren't -> refinery margins should expand -> go long products / short crude.

When CL convenience yield is low relative to products (cy_crack is positive) -> crude is loose, products tight -> margins should compress -> go long crude / short products.

Signal: z-score of `cy_crack` vs expanding mean. Mean-reversion trade when |z| > 1.5.

Position implementation: when cy_crack z-score < -1.5 (crude tight, products loose), go long 0.5 RB + 0.5 HO / short CL. When cy_crack z-score > +1.5 (crude loose, products tight), go long CL / short 0.5 RB + 0.5 HO. Dollar-neutral within the spread.

### 6.2 Inventory-Conditioned Energy Signals

For CL and NG (where EIA weekly inventory data exists):

- Compute inventory surprise (already built in `factors/inventory.py`)
- When inventory surprise is negative (larger-than-expected draw) AND convenience yield is rising -> amplify the long signal from Layer 1 by 1.5x
- When inventory surprise is positive (larger-than-expected build) AND convenience yield is falling -> amplify the short signal by 1.5x
- Otherwise -> no amplification

This is a conviction overlay, not a standalone signal. It adds sizing confidence to the energy positions in Layers 1/2.

### 6.3 Livestock Spread — Cattle-Hog with Seasonal Overlay

Spread: `LC_front - LH_front` (front-month prices, log difference for stationarity)

Livestock spreads mean-revert because of biological production cycles:
- Cattle cycle: ~10 years (breeding response is slow — gestation + maturation)
- Hog cycle: ~4 years (faster breeding, shorter maturation)
- When cattle is expensive relative to hogs, hog producers expand and cattle becomes relatively cheaper over the medium term

Signal: expanding z-score of the spread. Mean-reversion trade when |z| > 1.5.

Seasonal overlay:
- Compute ISO-week average of the spread over trailing 5 years (expanding, no lookahead)
- Subtract the seasonal component to get the deseasonalised z-score
- This prevents entering spread trades that are just reflecting normal seasonal patterns (e.g., grilling season lifts both, but hogs respond faster)

### 6.4 What Is NOT Included

- **Gold-silver ratio:** Too well-known among retail traders, no physical conversion margin story.
- **Platinum-palladium spread:** Thin markets, auto catalyst substitution is a multi-year story not suitable for monthly trading.
- **Grain spreads:** Would require ZL (soybean oil) and ZM (soybean meal) for the crush spread. Both were dropped from the universe because WRDS data ends in 2015 — missing the entire OOS window. ZS alone is insufficient.

### 6.5 Parameters

```yaml
structural_spreads:
  crack_spread:
    commodities: ["CL", "RB", "HO"]
    signal: "convenience_yield_divergence"
    z_threshold: 1.5
    rebalance: "biweekly"
  inventory_overlay:
    commodities: ["CL", "NG"]
    amplification: 1.5
    requires: ["inventory_surprise", "convenience_yield_change"]
  livestock_spread:
    long_leg: "LC"
    short_leg: "LH"
    z_threshold: 1.5
    seasonal_lookback_years: 5
    rebalance: "biweekly"
  risk_budget: 0.35
```

### 6.6 Implementation

`signals/spreads.py`
- `compute_cy_crack(convenience_yields) -> pd.Series` — convenience yield crack spread
- `crack_spread_signal(cy_crack, threshold=1.5) -> pd.DataFrame` — spread positions (CL/RB/HO weights)
- `inventory_overlay(positions, inventory_surprise, cy_change, amplification=1.5) -> pd.DataFrame`
- `livestock_spread_signal(lc_prices, lh_prices, seasonal_years=5, threshold=1.5) -> pd.DataFrame`

`signals/seasonal.py`
- `compute_seasonal_pattern(series, lookback_years=5) -> pd.Series` — ISO-week expanding average
- `deseasonalise(series, seasonal) -> pd.Series` — subtract seasonal component

## 7. Portfolio Construction

### 7.1 Layer Combination

Each layer produces commodity-level weights independently. Combine with risk budgeting:

1. Vol-target each layer independently to its risk budget share of the total 10% vol target:
   - Layer 1 weights -> target 4% vol (40% of 10%)
   - Layer 2 weights -> target 2.5% vol (25% of 10%)
   - Layer 3 weights -> target 3.5% vol (35% of 10%)
2. Sum the vol-targeted weights across layers
3. Apply portfolio-level constraints (position caps, sector caps)

### 7.2 Covariance Estimation

Use Ledoit-Wolf shrinkage estimator for the covariance matrix:
- Trailing 252-day returns
- Shrinkage target: identity matrix scaled by average variance
- This fixes the ill-conditioned covariance matrix problem (condition number 118 found in the audit)

Implementation: use `sklearn.covariance.LedoitWolf` or implement the analytical shrinkage formula from Ledoit & Wolf (2004).

### 7.3 Risk Constraints

| Constraint | Value | Source |
|-----------|-------|--------|
| Total vol target | 10% | `strategy.yaml` |
| Max position weight | 20% | `strategy.yaml` (existing) |
| Max sector weight | 40% | `strategy.yaml` (existing) |
| Max leverage | 2.0 | `strategy.yaml` (existing) |
| Execution lag | 1 day | `strategy.yaml` (existing) |

### 7.4 Implementation

`signals/combined_strategy.py`
- `combine_layers(layer1_weights, layer2_weights, layer3_weights, risk_budgets, returns) -> pd.DataFrame` — risk-budget-weighted combination
- `apply_ledoit_wolf_vol_target(weights, returns, target_vol, lookback=252) -> pd.DataFrame` — vol targeting with shrinkage covariance

## 8. Evaluation & Reporting

### 8.1 IS/OOS Discipline

- In-sample: 2005-01-01 to 2017-12-31 (13 years) — used for storage cost calibration, regime threshold selection
- Out-of-sample: 2018-01-01 to 2024-12-31 (7 years) — untouched until final evaluation
- All expanding-window z-scores start from the beginning of data (2003+) to maximise warm-up

### 8.2 What Gets Reported

1. **Per-layer performance:** Each layer run independently with its full risk budget. Shows which layers contribute alpha and which don't.
2. **Combined performance:** The three-layer portfolio.
3. **IS vs OOS comparison:** Sharpe, Sortino, CAGR, MaxDD for both windows.
4. **Bootstrap CIs:** 10,000 block-bootstrap samples for the combined Sharpe.
5. **Stress tests:** Performance during oil crash 2008, oil glut 2014, COVID 2020, energy spike 2022.
6. **Comparison vs benchmarks:** Equal-weight long, SPY, cash — already computed.
7. **Comparison vs old strategies:** The 7 original strategies remain in the codebase as a research baseline.

### 8.3 Honest Framing

If the combined Sharpe is 0.45 with CI [0.05, 0.85]:
- This IS statistically positive (CI excludes zero)
- It modestly beats the equal-weight long benchmark (Sharpe 0.27)
- It tells the story: term structure intelligence adds ~0.2 Sharpe over naive long exposure
- If any layer fails, report it honestly with the diagnosis

If the combined Sharpe is below 0.3:
- Report it honestly
- The project value shifts to the methodology: "I built the right framework and proved that even physically-motivated signals struggle post-2010"
- This is still a strong portfolio piece — commodity desks value honest research over inflated claims

## 9. Implementation Plan — Module List

| # | Module | New/Modify | Lines (est) | Dependencies |
|---|--------|-----------|-------------|-------------|
| 1 | `curves/convenience_yield.py` | New | ~150 | curves/metrics, FRED data |
| 2 | `signals/curve_regime.py` | New | ~120 | convenience_yield |
| 3 | `signals/directional.py` | New | ~100 | curve_regime, TSMOM |
| 4 | `signals/curve_transition.py` | New | ~100 | convenience_yield, TSMOM |
| 5 | `signals/spreads.py` | New | ~200 | convenience_yield, inventory, front-month prices |
| 6 | `signals/seasonal.py` | New | ~80 | — |
| 7 | `signals/combined_strategy.py` | New | ~150 | all signal layers, Ledoit-Wolf |
| 8 | `backtest/__main__.py` | Modify | ~50 | new strategy runner |
| 9 | `configs/strategy.yaml` | Modify | ~40 | new strategy params |
| 10 | Tests (7 test files) | New | ~600 | — |

**Total estimated new code:** ~1,600 lines (source + tests)

## 10. Realistic Expectations

| Configuration | Expected Sharpe | Basis |
|---|---|---|
| Layer 1 alone | 0.20 - 0.40 | Commodity risk premium + convenience yield timing |
| Layer 1 + Layer 2 | 0.30 - 0.55 | Transition momentum adds ~0.10-0.15 |
| All three layers | 0.40 - 0.70 | Spread alpha + diversification benefit |
| Equal-weight long (benchmark) | 0.27 | Already measured |
| SPY (benchmark) | 0.51 | Already measured |

The target is to beat the equal-weight long benchmark (Sharpe 0.27) with statistical significance and to produce a Sharpe competitive with SPY (0.51) — while telling a compelling story about physical commodity market understanding.

## 11. What Stays, What Changes

**Fully preserved (no changes):**
- Data pipeline (all 19 commodities, WRDS, EIA, CFTC, FRED)
- Curve construction (roll calendar, interpolation, builder, metrics)
- Factor engine (carry, slope, curvature, curve_momentum, TSMOM, XSMOM, inventory, positioning, macro, volatility, composites) — retained as research/diagnostic layer
- Backtest engine (engine, costs, benchmarks)
- Evaluation modules (metrics, factor_analysis, attribution, stress, bootstrap, capacity, report)
- All existing tests (396 passing)

**New additions:**
- 7 new source modules (~1,000 lines)
- 7 new test files (~600 lines)
- Config additions to strategy.yaml
- Updated backtest runner

**Narrative on the website:**
> "I tested 10 standard commodity factors across 19 markets and found they're arbitraged to near-zero in post-2010 data. Then I built what a real commodity desk runs: a systematic framework that reads the term structure the way a physical trader does — using convenience yield for positioning, regime transitions for timing, and structural spreads for relative value."
