# Visualization & Website — Design Spec

**Date:** 2026-04-19
**Status:** Approved design, pending implementation
**Scope:** Phase 6 (15 publication-quality charts) + Phase 7 (single-page editorial website)
**Reference:** Website and chart style cloned from `systematic-regime-trading` project

---

## 1. Narrative Structure

**Discovery arc:** Lead with the punchline ("10 factors, all arbitraged to zero"), walk through the research journey, arrive at the honest conclusion. The drama is in the intellectual honesty — a PM at a commodity desk has seen a hundred inflated backtests but rarely a candidate who proved their own hypothesis wrong and reported it cleanly.

**Key numbers:**
- 8 strategies tested (7 standard + TSI), 4 benchmarks (EW long, cash, SPY, AGG)
- TSI: IS Sharpe +0.39, OOS -0.30 (overfits)
- EW Long (equal-weight long-only benchmark): IS +0.22, OOS +0.35 (the robust winner — no active strategy beats it)
- TSMOM standalone: IS +0.17, OOS -0.23 (only factor with persistent lag1 IC, but not enough for positive OOS)
- SPY: OOS Sharpe +0.67, CAGR +14.23%
- 19 commodities, 5 sectors, 20 years of data (2005-2024)
- 438 tests, full IS/OOS discipline, bootstrap CIs
- Contamination discovery: lag0 IC 3-4x lag1 IC on all curve factors

---

## 2. Website Section Layout

11 sections matching the regime-trading editorial style exactly. Same CSS, same component classes, same fonts, same animations.

### §00 — Masthead

- **Ticker strip:** Key stats scrolling (19 COMMODITIES, 10 FACTORS, 438 TESTS, OOS SHARPE +0.35, 5 SECTORS, 20 YEARS, etc.)
- **Byline:** Brian Banna · Quantitative Research · Volume II
- **Title:** "Commodity Factor Trading" with italic accent on "Factor"
- **Tagline:** "Ten factors tested across nineteen commodity markets. All arbitraged to near-zero. The simplest strategy — equal-weight long — wins. Twenty years of data, no peeking."
- **Hero preview panel:** OOS metrics for EW Long benchmark (Sharpe +0.35, CAGR +5.3%, Max DD, Vol) with delta badges vs SPY
- **Badges:** 4 key stats (0.35 Sharpe, 19 commodities, 438 tests, 10 factors)

### §01 — Hook (data-theme="copper")

- **Headline:** "I tested ten commodity factors. *They're all dead.*"
- **Lead:** Standard carry, momentum, positioning signals that worked in academic papers from 1970-2010 have been arbitraged to near-zero by CTAs and systematic funds.
- **Side note:** "Why this matters" — most quant projects cherry-pick the best backtest. This one reports all variants honestly.
- **Stat cards:** Carry OOS Sharpe (-0.15), Best Factor OOS Sharpe (+0.02 TSMOM), Factors Tested (10), EW Long OOS Sharpe (+0.35)

### §02 — The Evidence (data-theme="gold")

- **Headline:** "All eight strategies, *after costs.*"
- **Side note:** Benchmarks explanation (IS 2005-2017, OOS 2018-2024, execution lag, roll costs)
- **Interactive Plotly equity curve:** All strategies + benchmarks
- **Static drawdown chart:** `02_drawdown.png`
- **HTML performance table:** All strategies with Sharpe, CAGR, Vol, Max DD, Turnover. EW Long row highlighted with star.

### §03 — The Contamination (data-theme="copper")

- **Headline:** "The contamination *no one talks about.*"
- **Lead:** All 4 curve factors have lag0 IC 3-4x larger than lag1 IC. Daily curve signals are tautological.
- **Static chart:** `05_factor_ic_decay.png` — the key research finding
- **Side note:** Academic papers use monthly rebalancing which masks this. Daily signals see the artifact.
- **Stat cards:** Carry lag0 IC, Carry lag1 IC, decay ratio, factors contaminated (4/4)

### §04 — Reading the Curve (data-theme="jade")

- **Headline:** "Reading the curve *like a physical trader.*"
- **Lead:** Convenience yield from cost-of-carry model, 5-regime classification, the term structure as supply/demand signal.
- **Interactive CY explorer:** Plotly chart with commodity toggle, regime shading
- **Static chart:** `08_curve_regime_heatmap.png`
- **Side note:** Theory of Storage explanation

### §05 — Term Structure Intelligence (data-theme="violet")

- **Headline:** "Three layers, *one thesis.*"
- **Lead:** TSI architecture — directional (40% risk budget), transition momentum (25%), structural spreads (35%).
- **Pipeline strip:** CY Estimation → Regime Classification → Directional + Transition + Spreads → Vol Target → Portfolio
- **Static chart:** `09_tsi_layer_decomposition.png`
- **Stat cards:** TSI IS Sharpe (+0.39), TSI OOS Sharpe (-0.30), IS-OOS gap (-0.69), Bootstrap CI includes zero
- **Honest conclusion paragraph:** "TSI overfits. The three-layer architecture captures in-sample dynamics that don't persist."

### §06 — What Actually Works (data-theme="jade")

- **Headline:** "The commodity risk premium *is the alpha.*"
- **Lead:** Equal-weight long-only. IS +0.22, OOS +0.35. The simplest strategy is the most robust. No active factor strategy beats passive long commodity exposure.
- **Static chart:** `04_rolling_sharpe.png`
- **Static chart:** `14_bootstrap_ci.png`
- **Side note:** "Why simplicity wins" — fewer parameters, less overfitting surface, captures the structural commodity risk premium.

### §07 — Cost Sensitivity (data-theme="amber")

- **Headline:** "How much friction *can it take?*"
- **Lead:** Commodity futures costs are typically 2-5 bps per side. Drag the slider.
- **Interactive cost slider:** Same pattern as regime-trading. Stat cards: Sharpe, CAGR, Max DD, breakeven cost.

### §08 — Stress Tests (data-theme="copper")

- **Headline:** "Through four *commodity crises.*"
- **Lead:** 2008 crash, 2014 oil glut, 2020 COVID/negative WTI, 2022 energy spike.
- **Static chart:** `10_stress_test.png`
- **Static chart:** `13_sector_attribution.png`
- **Stat cards:** Per-crisis returns for EW Long

### §09 — Architecture (data-theme="steel")

- **Headline:** "Nineteen markets, *one pipeline.*"
- **Pipeline strip:** WRDS → Curves → Factors → Signals → Backtest → Evaluation
- **Bullet list:** Data sources (WRDS institutional, EIA, CFTC, FRED, yfinance), 19 commodities × 5 sectors, 20 years daily data, 438 unit tests
- **Code snippet:** Factor computation or walk-forward loop (in `<details>` expandable)

### §10 — The Research (data-theme="steel")

- **Headline:** "The journey *is the finding.*"
- **Lead:** The research value is the methodology: honest IS/OOS discipline, contamination discovery, comprehensive factor testing.
- **Static chart:** `12_is_oos_comparison.png`
- **Static chart:** `06_factor_correlation.png`
- **Static chart:** `03_monthly_heatmap.png`

### Colophon

- "— Brian Banna"
- Links: brianbanna.com, GitHub repo

---

## 3. Charts — Phase 6 (Matplotlib PNGs)

15 publication-quality charts, all 300 DPI. Style matches regime-trading project exactly.

### 3.1 Matplotlib Style Configuration

```python
# style.py constants — identical to regime-trading
BG_COLOR   = "#0a0b0d"   # graphite
PAPER      = "#121214"   # paper card surface
FG_COLOR   = "#eae6de"   # bone
GRID_COLOR = "#2c2b28"   # subtle grid
ACCENT     = "#d4cec0"   # warm stone

UP_COLOR   = "#8ca891"   # muted sage
MID_COLOR  = "#c5b58c"   # warm tan
DOWN_COLOR = "#b87c6c"   # terracotta

DPI = 300

# rcParams — same as regime-trading setup()
plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   PAPER,
    "text.color":       FG_COLOR,
    "axes.labelcolor":  "#eae6de8c",
    "xtick.color":      "#eae6de70",
    "ytick.color":      "#eae6de70",
    "axes.edgecolor":   GRID_COLOR,
    "grid.color":       "#eae6de1e",
    "grid.alpha":       1.0,
    "grid.linewidth":   0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "font.family":      ["JetBrains Mono", "DejaVu Sans Mono", "monospace"],
    "font.size":        10,
    "axes.titlesize":   13,
    "legend.facecolor": PAPER,
    "legend.edgecolor": "#eae6de1e",
    "savefig.facecolor": BG_COLOR,
})
```

### 3.2 Custom Colormaps

- **editorial_diverging:** terracotta → paper → sage (monthly returns heatmap)
- **editorial_warm:** paper → tan → terracotta → stone (sequential warm)
- **editorial_cool:** paper → bone (correlation matrices, factor heatmaps)

### 3.3 Chart Specifications

| # | Filename | Size | Type | Data Source |
|---|----------|------|------|-------------|
| 1 | `01_cumulative_returns.png` | 14×6 | Line | `data/processed/backtest/*.parquet` |
| 2 | `02_drawdown.png` | 14×6 | Fill | Same |
| 3 | `03_monthly_heatmap.png` | 12×8 | Heatmap | EW Long monthly returns |
| 4 | `04_rolling_sharpe.png` | 14×6 | Line | 252-day rolling, IS/OOS divider |
| 5 | `05_factor_ic_decay.png` | 12×6 | Grouped bar | `evaluation.factor_analysis` |
| 6 | `06_factor_correlation.png` | 10×8 | Heatmap | 10×10 factor correlations |
| 7 | `07_convenience_yield.png` | 14×6 | Line + shading | CY data + regime classification |
| 8 | `08_curve_regime_heatmap.png` | 14×8 | Heatmap | 19 commodities × time × regime |
| 9 | `09_tsi_layer_decomposition.png` | 14×6 | Line | Per-layer cumulative returns |
| 10 | `10_stress_test.png` | 12×6 | Grouped bar | 4 crises × strategies |
| 11 | `11_cost_sensitivity.png` | 10×6 | Line | `cost_sensitivity.parquet` |
| 12 | `12_is_oos_comparison.png` | 12×6 | Grouped bar | IS vs OOS Sharpe |
| 13 | `13_sector_attribution.png` | 12×6 | Stacked bar | Attribution by sector |
| 14 | `14_bootstrap_ci.png` | 12×6 | Violin/CI | Bootstrap Sharpe distributions |
| 15 | `15_performance_table.png` | 14×8 | Table render | Performance summary |

### 3.4 Strategy Color Mapping

```python
STRATEGY_COLORS = {
    "tsmom":              ACCENT,           # gold — the winner
    "tsi":                DOWN_COLOR,       # copper — the ambitious attempt
    "xs_carry":           "#eae6de40",
    "multi_factor_ew":    "#eae6de45",
    "multi_factor_ic":    "#eae6de50",
    "regime_conditioned": "#eae6de55",
    "sector_neutral":     "#eae6de4c",
    "calendar_spread":    "#eae6de48",
}

BENCHMARK_COLORS = {
    "ew_long": "#eae6de8c",   # bone — passive benchmark
    "spy":     "#b87c6c80",   # copper muted
    "agg":     "#eae6de30",
    "cash":    "#eae6de20",
}
```

### 3.5 Crisis Annotation Helper

Shared function for shading crisis periods on time-series charts:

```python
CRISIS_PERIODS = {
    "2008 Crash":      ("2008-06-01", "2009-03-01"),
    "Oil Glut 2014":   ("2014-06-01", "2016-02-01"),
    "COVID 2020":      ("2020-02-01", "2020-06-01"),
    "Energy Spike 2022": ("2022-02-01", "2022-10-01"),
}
```

---

## 4. Interactive Elements (Plotly + JS)

Three interactive charts, all using the same Plotly theme and configuration pattern as the regime-trading site.

### 4.1 Plotly Theme Constants

```javascript
var INK    = '#0c0c0e';
var PAPER  = '#141417';
var BG     = '#0a0b0d';
var FG     = '#eae6de';
var FG_60  = 'rgba(234, 230, 222, 0.6)';
var FG_30  = 'rgba(234, 230, 222, 0.14)';
var ACCENT = '#c8a255';   // gold
var UP     = '#62af7b';   // jade
var DOWN   = '#cf664d';   // copper
var BLUE   = '#6894be';   // steel
var VIOLET = '#9b82c8';   // violet
var FONT   = "JetBrains Mono, ui-monospace, monospace";
```

### 4.2 Equity Curve (§02)

- Traces for all strategies + benchmarks
- EW Long as gold hero line (width 2.4)
- TSI as copper (width 1.6)
- SPY as bone 0.55 opacity
- Others at descending bone opacity
- Hover: strategy name, date, cumulative return
- Legend toggle, drag to zoom
- Mobile responsive (380px height on small screens)
- Same Plotly layout config as regime-trading (paper_bgcolor, gridcolor, hoverlabel, etc.)

### 4.3 Cost Sensitivity Slider (§07)

- Same slider pattern as regime-trading
- Data from `cost_sensitivity.parquet` (sweep at 0, 2, 5, 8, 10, 15, 20 bps)
- Stat cards update live: Sharpe, CAGR, Max DD, breakeven cost
- Linear interpolation between data points via `lerp()` function
- Verdict text: green if beats EW Long benchmark, amber if profitable but underperforms, red if unprofitable

### 4.4 Convenience Yield Explorer (§04)

- Plotly chart with legend toggle to show/hide commodities
- Default visible: CL, NG, GC
- CY time series as lines (one per commodity, signal-palette colors)
- Background shading via Plotly shapes for 5 regime classifications:
  - Crisis backwardation: jade at 0.15 opacity
  - Mild backwardation: jade at 0.08 opacity
  - Balanced: transparent
  - Mild contango: copper at 0.08 opacity
  - Deep contango: copper at 0.15 opacity
- Regime shading follows the selected commodity (first visible trace)
- Hover: date, CY value, regime label
- Same dark Plotly theme

---

## 5. File Structure

```
website/
  index.html                    # Single-page editorial site
  css/
    style.css                   # Copied verbatim from regime-trading
  js/
    chart_data_inline.js        # Generated by visualization/__main__.py
  assets/
    figures/                    # Generated PNGs (copied from results/figures/)

src/commodity_curve_factors/visualization/
  __init__.py                   # Already exists (empty)
  style.py                      # rcParams, palette, colormaps, figure helpers
  performance.py                # Charts 01, 02, 03, 04, 11, 12, 15
  factors.py                    # Charts 05, 06
  curves.py                     # Charts 07, 08
  tsi.py                        # Chart 09
  risk.py                       # Charts 10, 13, 14
  __main__.py                   # Generate all 15 PNGs + chart_data_inline.js

results/
  figures/                      # Output directory for all 15 PNGs
```

### 5.1 Module Dependencies

| Module | Reads | Calls |
|--------|-------|-------|
| `style.py` | `configs/evaluation.yaml` | — |
| `performance.py` | `data/processed/backtest/*.parquet` | `evaluation.metrics` |
| `factors.py` | `data/processed/factors/*.parquet` | `evaluation.factor_analysis` |
| `curves.py` | `data/processed/curves/*.parquet`, `data/processed/curve_metrics/*.parquet` | `curves.convenience_yield`, `signals.curve_regime` |
| `tsi.py` | `data/processed/backtest/tsi.parquet`, signal layer data | `signals.directional`, `signals.curve_transition`, `signals.spreads` |
| `risk.py` | `data/processed/backtest/*.parquet` | `evaluation.stress`, `evaluation.bootstrap`, `evaluation.attribution` |
| `__main__.py` | All of the above | All visualization modules |

### 5.2 Chart Data Generation

`__main__.py` generates `website/js/chart_data_inline.js` containing:

```javascript
var CHART_DATA = {
    strategies: {
        "tsmom": { dates: [...], cumulative_return: [...] },
        "tsi":   { dates: [...], cumulative_return: [...] },
        // ... all strategies + benchmarks
    }
};

var COST_DATA = {
    bps:    [0, 2, 5, 8, 10, 15, 20],
    sharpe: [...],
    cagr:   [...],
    maxdd:  [...]
};

var CY_DATA = {
    dates: [...],
    commodities: {
        "CL": { cy: [...], regime: [...] },
        "NG": { cy: [...], regime: [...] },
        // ... all 19 commodities
    },
    regime_labels: ["Deep Contango", "Mild Contango", "Balanced", "Mild Backwardation", "Crisis Backwardation"]
};
```

---

## 6. CSS & Typography

**No CSS changes required.** The `style.css` from `systematic-regime-trading/website/css/style.css` is copied verbatim. All component classes are reused:

- `.masthead`, `.ticker`, `.byline`, `.title-block`, `.hero-lower`, `.hero-preview`, `.badges`
- `.section-marker`, `.section-grid`, `.side-note`
- `.stat-row`, `.stat-card`, `.chart-container`, `.chart-caption`
- `.table-wrap`, `table`
- `.sim-container`, `.sim-slider-row`
- `.pipeline`, `.pipeline-step`, `.pipeline-arrow`
- `.reveal` (scroll animations)
- `[data-theme]` section accent overrides

Same Google Fonts import: Source Serif 4, Inter, JetBrains Mono.

### 6.1 Per-Section Theme Assignments

| Section | data-theme | Accent | Narrative tone |
|---------|-----------|--------|----------------|
| §01 Hook | copper | terracotta | "Everything failed" |
| §02 Evidence | gold | gold | Core results |
| §03 Contamination | copper | terracotta | Warning / discovery |
| §04 Reading the Curve | jade | sage | Building something new |
| §05 TSI | violet | violet | Experimental |
| §06 What Works | jade | sage | Positive outcome |
| §07 Costs | amber | amber | Sensitivity |
| §08 Stress | copper | terracotta | Crisis periods |
| §09 Architecture | steel | steel blue | Technical |
| §10 Research | steel | steel blue | Summary |

---

## 7. Makefile Targets

```makefile
figures:
	python -m commodity_curve_factors.visualization

website: figures
	@echo "Website ready at website/index.html"
```

`make figures` generates all 15 PNGs + `chart_data_inline.js`. The website is static HTML — no build step needed beyond generating the data.

---

## 8. What Is NOT Included

- No 3D surface plot (too gimmicky, adds complexity without insight)
- No separate JS files — all inline in HTML (matching reference project)
- No build toolchain (no webpack, no npm) — static HTML/CSS/JS
- No animated crisis growth chart (the regime-trading project had this because it had a clear hero result; this project's story is different)
- No server-side rendering — deploy-ready for GitHub Pages as-is
- No dark/light mode toggle — dark only (matching reference)
