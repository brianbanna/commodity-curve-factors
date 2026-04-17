# Commodity Futures Curve Modeling and Factor Trading

Extract carry, slope, curvature, and momentum signals from commodity futures term structures. Build systematic factor portfolios. Backtest with realistic roll costs.

This framework constructs daily term structures across 19 commodity markets (5 sectors: energy, metals, agriculture, softs, livestock), computes factor signals from curve shape, and runs long-short trading strategies through a futures-aware backtest that accounts for roll costs, margin, and per-commodity transaction costs. Everything is config-driven.

<!--
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()
-->

## Overview

A commodity futures price is not a single number - it is a curve, a series of prices stretching months into the future. That curve shape tells you something real about the market:

- **Backwardation** (near prices above deferred) means scarcity, strong demand
- **Contango** (near prices below deferred) means plenty of supply, storage costs getting priced in
- **Curve flattening** tends to show up before or during price rallies
- **Curve steepening** tends to show up before or during selloffs

This framework takes those dynamics and turns them into systematic trading signals across energy, metals, agriculture, softs, and livestock markets.

**What it does:**

- Downloads futures data across multiple maturities for 19 commodities
- Builds daily term structures with proper contract roll handling
- Computes 11 factor signals from curve shape, momentum, positioning, and fundamental data
- Constructs cross-sectional and time-series trading portfolios
- Runs vectorized backtests with roll costs, slippage, and margin tracking
- Evaluates performance at the factor level (IC, decay, attribution)

**What it produces:**

- Daily term structures and curve metrics for every commodity
- Factor signals: carry, slope, curvature, curve momentum, convenience yield, TSMOM, XSMOM, CFTC positioning, inventory surprise, macro exposure, volatility regime
- Net-of-cost strategy returns across multiple portfolio variants
- Performance tearsheet with factor IC analysis and cost sensitivity
- Static research website with term structure visualizations

## Research Pipeline

```
              Futures Price Data
          (yfinance + WRDS Datastream)
                      |
            Curve Construction
      (interpolation, roll handling)
                      |
             Factor Engineering
    (carry, slope, curvature, momentum)
                      |
           Inventory Overlay
          (EIA surprise signals)
                      |
          Signal Construction
   (cross-sectional rank, z-score)
                      |
        Portfolio Allocation
    (vol-targeting, sector limits)
                      |
         Futures Backtest
     (roll costs, margin, slippage)
                      |
      Performance Evaluation
  (Sharpe, IC, factor attribution)
```

## Factors

Eleven factors across three families:

**Structural curve** (from WRDS back-month contracts):

| Factor | What It Captures |
| --- | --- |
| **Carry** | Annualized roll yield (F1-F2). Positive in backwardation, negative in contango. |
| **Slope** | Full-curve tilt (F12M vs F1M). Steep contango is bearish, steep backwardation is bullish. |
| **Curvature** | Butterfly shape (F1M - 2xF6M + F12M). Fastest mean-reverting curve signal. |
| **Curve Momentum** | Rate of change in slope. Flattening precedes rallies, steepening precedes declines. |
| **Convenience Yield** | Implied benefit of holding physical commodity. Spikes signal scarcity. |

**Flow & behavioral:**

| Factor | What It Captures |
| --- | --- |
| **TSMOM** | Time-series momentum. Per-commodity trend signal with the strongest persistent IC. |
| **XSMOM** | Cross-sectional momentum. Relative strength ranking across the universe. |
| **CFTC Positioning** | Net speculative positioning from Commitments of Traders. Contrarian at extremes. |

**Fundamental & macro:**

| Factor | What It Captures |
| --- | --- |
| **Inventory Surprise** | Deviation from 5-year seasonal norm (EIA energy data). Draws support backwardation. |
| **Macro Exposure** | Regression-based sensitivity to rates, inflation, and USD. |
| **Volatility Regime** | VIX-based regime classification (calm / moderate / turbulent). |

All factors z-scored with expanding windows so there is no lookahead.

## Strategies

Two strategy generations reflecting the research arc of this project.

### Generation 1 -- Research baseline

7 strategies using the 10 standard commodity factors. Key finding: standard factors are arbitraged to near-zero post-2010. TSMOM is the only factor with persistent lag-1 IC.

| Strategy | Method |
| --- | --- |
| Cross-Sectional Carry | Long top-N carry, short bottom-N |
| Multi-Factor | Composite z-score ranking across all factors |
| Sector-Neutral | Within-sector long/short (strips sector beta) |
| Time-Series Carry | Per-commodity threshold signals |
| Calendar Spread Carry | Long front / short back per commodity |
| Regime-Conditioned | Factor weights shift with VIX regime |
| MinVar + TSMOM | Minimum-variance allocation filtered by trend |

### Generation 2 -- Term Structure Intelligence

3-layer strategy that reads the term structure like a physical trader:

| Layer | Signal | Logic |
| --- | --- | --- |
| **L1: Directional** | Convenience yield + TSMOM filter | CY-based positioning, trend confirmation gates entry |
| **L2: Curve Regime** | Curve regime transition momentum | Trade the shift between contango and backwardation regimes |
| **L3: Structural Spreads** | CY crack spread, inventory-conditioned energy, deseasonalised livestock spread | Relative-value within and across sectors |

All strategies include:

- Weekly rebalancing (signal Friday, execute Monday)
- Per-commodity vol targeting at 10% annualized
- Sector exposure caps at 40%
- Per-commodity cost model (commission + slippage + roll costs)
- Cost sensitivity sweep from 0 to 20 bps

Benchmarks: equal-weight long-only commodity basket, cash (for long-short).

## Data

| Dataset | Source | Frequency |
| --- | --- | --- |
| Continuous futures (front-month) | yfinance | Daily |
| Back-month contracts (curve construction) | WRDS Datastream | Daily |
| US crude / gas / product inventories | EIA API | Weekly |
| Speculative positioning | CFTC COT | Weekly |
| Dollar index, rates, inflation | FRED | Daily |

`make data` handles all downloads. API keys for EIA and FRED are read from environment variables; WRDS credentials required for back-month data.

## Commodity Universe

| Sector | Commodities |
| --- | --- |
| Energy | WTI Crude (CL), Brent Crude (BZ), Natural Gas (NG), Heating Oil (HO), Gasoline (RB) |
| Metals | Gold (GC), Silver (SI), Copper (HG), Platinum (PL), Palladium (PA) |
| Agriculture | Corn (ZC), Soybeans (ZS), Wheat (ZW), Soybean Oil (ZL) |
| Softs | Coffee (KC), Sugar (SB), Cocoa (CC), Cotton (CT) |
| Livestock | Live Cattle (LE), Lean Hogs (HE) |

19 commodities across 5 sectors. Curve depth varies - monthly contracts for energy, seasonal delivery months for agriculture and livestock.

## Key Research Outputs

- Daily term structures and curve metrics for every commodity in the universe
- Factor performance with information coefficients and decay curves
- Strategy backtests with proper futures roll cost modeling
- Factor attribution broken down by commodity, sector, and time period
- Transaction cost sensitivity with breakeven identification
- Performance tearsheet with bootstrap confidence intervals
- Research website with term structure heatmaps and factor charts

## Quickstart

```bash
git clone https://github.com/brianbanna/commodity-curve-factors.git
cd commodity-curve-factors

pip install -e .

# Run the full pipeline
make all

# Or run stages individually
make data          # Download futures, inventory, and macro data
make curves        # Construct daily term structures
make factors       # Compute all 11 factor signals
make signals       # Generate portfolio signals and weights
make backtest      # Run strategy backtests with roll costs
make evaluate      # Compute performance metrics and factor analysis
make report        # Generate tearsheet and charts

# Run tests
make test
```

Requires Python 3.10+. Free API keys needed for EIA and FRED. WRDS institutional credentials needed for back-month data.

## Project Structure

```
commodity-curve-factors/
├── configs/
│   ├── universe.yaml          # Commodity specs, contract months, sectors
│   ├── curve.yaml             # Tenors, interpolation, roll rules
│   ├── factors.yaml           # Factor definitions, z-score windows
│   ├── inventory.yaml         # EIA series, seasonal params
│   ├── strategy.yaml          # Portfolio weights, rebalance, constraints
│   ├── backtest.yaml          # Per-commodity costs, margin, execution lag
│   └── evaluation.yaml        # Metrics, benchmarks, stress test periods
│
├── src/
│   └── commodity_curve_factors/
│       ├── data/              # yfinance, WRDS, EIA, CFTC, FRED loaders
│       ├── curves/            # Term structure builder, interpolation, rolls
│       ├── factors/           # Carry, slope, curvature, momentum, inventory
│       ├── signals/           # Ranking, thresholds, calendar spreads, regime
│       ├── backtest/          # Futures engine, roll costs, benchmarks
│       ├── evaluation/        # Metrics, factor IC, attribution, reporting
│       ├── visualization/     # Curve plots, factor charts, tearsheet
│       └── utils/             # Config loader, paths, constants
│
├── notebooks/                 # Step-by-step research notebooks
├── data/                      # Raw + processed data (git-ignored)
├── results/                   # Figures, tables, tearsheets
├── website/                   # Static research site (GitHub Pages)
├── tests/                     # Unit and integration tests
├── Makefile                   # Pipeline orchestration
└── pyproject.toml             # Dependencies
```

All parameters live in YAML configs. No magic numbers in source code.

## License

MIT

## Disclaimer

Research code for educational and demonstration purposes. Not investment advice. Backtested performance does not guarantee future results.
