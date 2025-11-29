# Commodity Futures Curve Modeling and Factor Trading

A quantitative trading research framework for modeling and trading term-structure factors in global commodity futures markets.

This repository focuses on extracting and analyzing carry, slope, and curvature\* factors from futures curves and testing their ability to forecast returns and risk premia.

## Overview

This project develops a systematic approach to commodity curve factor trading by:

- Building continuous futures curves from raw contract data
- Extracting structural factors (carry, slope, curvature) that drive term structure dynamics
- Applying statistical models (PCA, GARCH, HMM) to identify regimes and forecast volatility
- Constructing long-short factor portfolios with volatility targeting
- Backtesting strategies with performance evaluation and risk analysis

## Project Structure

```
commodity-curve-factors/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data_pipeline/          # Data download, cleaning, and curve construction
│   │   ├── __init__.py
│   │   └── load_data.py
│   ├── features/                # Curve feature extraction
│   │   ├── __init__.py
│   │   └── curve_features.py
│   ├── models/                  # Factor modeling (PCA, GARCH, HMM)
│   │   ├── __init__.py
│   │   ├── pca_model.py
│   │   ├── garch_model.py
│   │   └── hmm_model.py
│   ├── trading/                 # Strategy logic and signal generation
│   │   ├── __init__.py
│   │   └── factor_strategy.py
│   ├── backtesting/             # Backtest engine and performance metrics
│   │   ├── __init__.py
│   │   └── backtest_engine.py
│   ├── utils/                   # Visualization and results management
│   │   ├── __init__.py
│   │   ├── visualization.py
│   │   └── results/
│   │       ├── figures/         # Saved plots and charts
│   │       ├── tables/          # Performance metrics tables
│   │       └── reports/         # Generated analysis reports
│   └── config/                  # Global parameters and settings
│       ├── __init__.py
│       └── settings.yaml
├── notebooks/                   # Ordered research notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_curve_features.ipynb
│   ├── 03_factor_modeling.ipynb
│   ├── 04_strategy_backtest.ipynb
│   └── 05_results_analysis.ipynb
└── tests/                       # Unit tests and validation
    ├── test_data_integrity.py
    ├── test_factor_computation.py
    ├── test_strategy_performance.py
    └── test_model_convergence.py
```
