# Commodity Futures Curve Modeling and Factor Trading

Extract carry, slope, curvature, and momentum signals from commodity futures term structures. Build systematic factor portfolios. Backtest with realistic roll costs.

This framework constructs daily term structures across 13 commodity markets, computes factor signals from curve shape, and runs long-short trading strategies through a futures-aware backtest that accounts for roll costs, margin, and per-commodity transaction costs. Everything is config-driven.

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

This framework takes those dynamics and turns them into systematic trading signals across energy, metals, and agricultural markets.

**What it does:**

- Downloads futures data across multiple maturities for 13 commodities
- Builds daily term structures with proper contract roll handling
- Computes five factor signals from curve shape and fundamental data
- Constructs cross-sectional and time-series trading portfolios
- Runs vectorized backtests with roll costs, slippage, and margin tracking
- Evaluates performance at the factor level (IC, decay, attribution)

**What it produces:**

- Daily term structures and curve metrics for every commodity
- Factor signals: carry, slope, curvature, momentum, inventory surprise
- Net-of-cost strategy returns across multiple portfolio variants
- Performance tearsheet with factor IC analysis and cost sensitivity
- Static research website with term structure visualizations

## Research Pipeline

```
              Futures Price Data
            (Stooq + Nasdaq Data Link)
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

Five factors extracted from curve structure and fundamentals:

| Factor | Formula | What It Captures |
| --- | --- | --- |
| **Carry** | `(F1 - F2) / F2 x 12` | Annualized roll yield. Positive in backwardation, negative in contango. The single biggest return driver in commodity futures. |
| **Slope** | `(F12M - F1M) / F1M` | Full-curve tilt. Steep contango is bearish, steep backwardation is bullish. Tends to mean-revert over weeks. |
| **Curvature** | `F1M - 2 x F6M + F12M` | Butterfly shape - convexity vs concavity of the term structure. The fastest mean-reverting curve signal. |
| **Curve Momentum** | `slope(t) - slope(t-L)` | How fast the curve shape is changing. Flattening tends to precede rallies, steepening tends to precede declines. |
| **Inventory Surprise** | `(actual - seasonal_avg) / std` | Deviation from 5-year seasonal norm. Draws support backwardation, builds support contango. |

All factors z-scored with expanding windows so there is no lookahead.

## Strategies

Six portfolio strategies built on top of the factor signals:

| Strategy | Method | Characteristics |
| --- | --- | --- |
| Cross-Sectional Carry | Long top 3 carry, short bottom 3 | The classic commodity carry trade |
| Multi-Factor | Composite z-score ranking | Diversified across all five factors |
| Sector-Neutral | Within-sector long/short | Strips out sector beta |
| Time-Series Carry | Per-commodity threshold signals | No cross-sectional dependence |
| Calendar Spread Carry | Long front / short back per commodity | Lower margin, direct curve expression |
| Regime-Conditioned | Factor weights shift with vol regime | Heavier carry in calm, heavier momentum in turbulent |

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
| Continuous futures (front + second month) | Stooq | Daily |
| Back-month contracts (3M-12M) | Nasdaq Data Link | Daily |
| US crude / gas / product inventories | EIA API | Weekly |
| Speculative positioning | CFTC COT | Weekly |
| Dollar index, rates, inflation | FRED | Daily |

`make data` handles all downloads. API keys for Nasdaq Data Link and EIA are read from environment variables.

## Commodity Universe

| Sector | Commodities |
| --- | --- |
| Energy | WTI Crude (CL), Natural Gas (NG), Heating Oil (HO), Gasoline (RB) |
| Metals | Gold (GC), Silver (SI), Copper (HG) |
| Agriculture | Corn (ZC), Soybeans (ZS), Wheat (ZW) |
| Softs | Coffee (KC), Sugar (SB), Cocoa (CC) |

13 commodities across 4 sectors. Curve depth varies - monthly contracts for energy, seasonal delivery months for agriculture.

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
make factors       # Compute carry, slope, curvature, momentum, inventory
make signals       # Generate portfolio signals and weights
make backtest      # Run strategy backtests with roll costs
make evaluate      # Compute performance metrics and factor analysis
make report        # Generate tearsheet and charts

# Run tests
make test
```

Requires Python 3.10+. Free API keys needed for Nasdaq Data Link and EIA.

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
│       ├── data/              # Stooq, Nasdaq Data Link, EIA, FRED loaders
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
