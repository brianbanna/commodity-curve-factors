"""
Visualization utilities for curves, factors, and performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List


def plot_curve_term_structure(
    prices: pd.DataFrame,
    date: str,
    commodity: str,
    save_path: Optional[str] = None
) -> None:
    """
    Plot futures curve term structure for a given date.

    Parameters:
        prices: DataFrame with futures prices by maturity.
        date: Date to plot (YYYY-MM-DD).
        commodity: Commodity name for title.
        save_path: Optional path to save figure.
    """
    pass


def plot_factor_time_series(
    factors: pd.DataFrame,
    factors_to_plot: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot factor time series.

    Parameters:
        factors: DataFrame with factor time series.
        factors_to_plot: List of factor names to plot (default: all).
        save_path: Optional path to save figure.
    """
    pass


def plot_pca_loadings(
    loadings: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot PCA factor loadings by maturity.

    Parameters:
        loadings: DataFrame with PCA loadings.
        save_path: Optional path to save figure.
    """
    pass


def plot_performance_attribution(
    returns: pd.Series,
    factor_returns: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot performance attribution by factor.

    Parameters:
        returns: Series of strategy returns.
        factor_returns: DataFrame with factor return contributions.
        save_path: Optional path to save figure.
    """
    pass


def plot_regime_performance(
    returns: pd.Series,
    regimes: pd.Series,
    save_path: Optional[str] = None
) -> None:
    """
    Plot strategy performance by regime.

    Parameters:
        returns: Series of strategy returns.
        regimes: Series with regime labels.
        save_path: Optional path to save figure.
    """
    pass

