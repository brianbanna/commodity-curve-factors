# Commodity Futures Curve Modeling and Factor Trading

**A Quantitative Research Framework**

Extract carry, slope, curvature, and momentum signals from commodity futures term structures. Build systematic factor portfolios. Backtest with realistic roll costs.

A config-driven framework that constructs daily term structures across 13 commodity markets, computes factor signals from curve shape, and evaluates long-short trading strategies with futures-aware backtesting including roll costs, margin tracking, and per-commodity transaction cost modeling.

<!--
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()
-->

## Overview

A commodity futures price is not a single number. It is a **curve** — a series of prices stretching months into the future. The shape of that curve encodes fundamental information about the market:

- **Backwardation** (near prices above deferred) signals scarcity and strong demand
- **Contango** (near prices below deferred) signals abundance and storage costs
- **Curve flattening** often precedes or accompanies price rallies
- **Curve steepening** often precedes or accompanies price declines

This framework extracts systematic trading signals from these curve dynamics across energy, metals, and agricultural commodity markets.

**What it does:**

- Downloads futures data across multiple maturities for 13 commodities
- Constructs daily term structures with proper contract roll handling
- Computes five factor signals from curve shape and fundamental data
- Builds cross-sectional and time-series trading portfolios
- Runs vectorized backtests with roll costs, slippage, and margin tracking
- Evaluates performance with factor-level analysis (IC, decay, attribution)

**What it produces:**

- Daily term structures and curve metrics for all commodities
- Factor signals: carry, slope, curvature, momentum, inventory surprise
- Net-of-cost strategy returns for multiple portfolio variants
- Full performance tearsheet with factor IC analysis and cost sensitivity
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
         (EIA, USDA surprise signals)
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

Five systematic factors extracted from curve structure and fundamentals:

| Factor | Formula | What It Captures |
| --- | --- | --- |
| **Carry** | `(F1 - F2) / F2 × 12` | Annualized roll yield. Positive in backwardation, negative in contango. The dominant return driver in commodity futures. |
| **Slope** | `(F12M - F1M) / F1M` | Full-curve tilt. Steep contango = bearish, steep backwardation = bullish. Mean-reverts over weeks. |
| **Curvature** | `F1M - 2×F6M + F12M` | Butterfly shape. Convexity vs concavity of the term structure. Fastest mean-reverting curve signal. |
| **Curve Momentum** | `slope(t) - slope(t-L)` | Rate of change in curve shape. Flattening precedes rallies, steepening precedes declines. |
| **Inventory Surprise** | `(actual - seasonal_avg) / std` | Deviation from 5-year seasonal norm. Draws support backwardation, builds support contango. |

All factors are z-scored using expanding windows to prevent lookahead.

## Strategies

Factor signals drive six portfolio strategies:

| Strategy | Method | Characteristics |
| --- | --- | --- |
| Cross-Sectional Carry | Long top 3 carry, short bottom 3 | Classic commodity carry trade |
| Multi-Factor | Composite z-score ranking | Diversified across all factors |
| Sector-Neutral | Within-sector long/short | Removes sector beta |
| Time-Series Carry | Per-commodity threshold signals | No cross-sectional dependence |
| Calendar Spread Carry | Long front / short back per commodity | Lower margin, direct curve expression |
| Regime-Conditioned | Factor weights adjusted by volatility regime | Carry in calm, momentum in turbulent |

All strategies include:

- Weekly rebalancing (signal Friday, execute Monday)
- Per-commodity volatility targeting (10% annualized)
- Sector exposure limits (max 40% per sector)
- Per-commodity cost model (commission + slippage + roll costs)
- Cost sensitivity analysis from 0 to 20 bps

Benchmarks: equal-weight long-only commodity basket, cash (for long-short strategies).

## Data

| Dataset | Source | Frequency |
| --- | --- | --- |
| Continuous futures (front + second month) | Stooq | Daily |
| Back-month contracts (3M-12M) | Nasdaq Data Link | Daily |
| US crude / gas / product inventories | EIA API | Weekly |
| Crop production estimates | USDA WASDE | Monthly |
| Speculative positioning | CFTC COT | Weekly |
| Dollar index, rates, inflation | FRED | Daily |

Data downloads automatically via `make data`. API keys for Nasdaq Data Link and EIA are read from environment variables.

## Commodity Universe

| Sector | Commodities |
| --- | --- |
| Energy | WTI Crude (CL), Natural Gas (NG), Heating Oil (HO), Gasoline (RB) |
| Metals | Gold (GC), Silver (SI), Copper (HG) |
| Agriculture | Corn (ZC), Soybeans (ZS), Wheat (ZW) |
| Softs | Coffee (KC), Sugar (SB), Cocoa (CC) |

13 commodities across 4 sectors. Curve depth varies by commodity (monthly contracts for energy, seasonal for agriculture).

## Key Research Outputs

- Daily term structures and curve metrics for all commodities
- Factor performance analysis with information coefficients and decay curves
- Strategy backtests with futures-aware roll cost modeling
- Factor attribution by commodity, sector, and time period
- Transaction cost sensitivity and breakeven analysis
- Full performance tearsheet with bootstrap confidence intervals
- Static research website with term structure heatmaps

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

Requires Python 3.10+. API keys needed for Nasdaq Data Link and EIA (both free).

## Project Structure

```
commodity-curve-factors/
├── configs/
│   ├── universe.yaml          # Commodity specs, contract months, sectors
│   ├── curve.yaml             # Tenors, interpolation, roll rules
│   ├── factors.yaml           # Factor definitions, z-score windows
│   ├── inventory.yaml         # EIA/USDA series, seasonal params
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

All parameters are externalized to YAML configs. No magic numbers in source code.

## License

MIT

## Disclaimer

Research code for educational and demonstration purposes. Not investment advice. Backtested performance does not guarantee future results.
