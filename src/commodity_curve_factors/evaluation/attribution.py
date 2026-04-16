"""Return attribution by sector, time period, and regime."""

import logging

import numpy as np
import pandas as pd

from commodity_curve_factors.evaluation.metrics import (
    compute_all_metrics,
    sharpe_ratio,
)
from commodity_curve_factors.utils.constants import SECTORS

logger = logging.getLogger(__name__)


def attribution_by_sector(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose strategy returns by sector contribution.

    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights (dates x commodities).
    returns : pd.DataFrame
        Daily returns (dates x commodities).

    Returns
    -------
    pd.DataFrame
        Columns: sector, contribution (annualised), pct_of_total.
    """
    common_idx = weights.index.intersection(returns.index)
    common_cols = weights.columns.intersection(returns.columns)
    w = weights.loc[common_idx, common_cols]
    r = returns.loc[common_idx, common_cols]

    total_ret = (w * r).sum(axis=1)
    total_annual = float(total_ret.mean()) * 252

    rows = []
    for sector, symbols in SECTORS.items():
        cols = [s for s in symbols if s in common_cols]
        if not cols:
            continue
        sector_ret = (w[cols] * r[cols]).sum(axis=1)
        contrib = float(sector_ret.mean()) * 252
        pct = contrib / total_annual if total_annual != 0 else 0.0
        rows.append({"sector": sector, "contribution": contrib, "pct_of_total": pct})

    result = pd.DataFrame(rows)
    logger.info("attribution_by_sector: %d sectors", len(result))
    return result


def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """Rolling annualised Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    window : int
        Rolling window in trading days.

    Returns
    -------
    pd.Series
        Rolling Sharpe with DatetimeIndex.
    """
    roll_mean = returns.rolling(window, min_periods=window // 2).mean()
    roll_std = returns.rolling(window, min_periods=window // 2).std()
    result = (roll_mean / roll_std) * np.sqrt(252)
    result.name = "rolling_sharpe"
    return result


def attribution_by_regime(
    returns: pd.Series,
    regimes: pd.Series,
) -> dict[str, dict[str, float]]:
    """Compute performance metrics within each VIX regime.

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns.
    regimes : pd.Series
        Regime labels (e.g. "calm", "moderate", "turbulent") aligned to returns.

    Returns
    -------
    dict[str, dict[str, float]]
        Regime name -> performance metrics dict.
    """
    common_idx = returns.index.intersection(regimes.index)
    r = returns.loc[common_idx]
    reg = regimes.loc[common_idx]

    result = {}
    for label in sorted(reg.unique()):
        mask = reg == label
        subset = r[mask]
        if len(subset) < 20:
            continue
        result[label] = {
            "sharpe": sharpe_ratio(subset),
            "n_days": len(subset),
            "mean_return": float(subset.mean()) * 252,
            "volatility": float(subset.std()) * np.sqrt(252),
        }

    logger.info("attribution_by_regime: %d regimes", len(result))
    return result


def attribution_by_year(returns: pd.Series) -> pd.DataFrame:
    """Annual performance breakdown.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        One row per year with columns: year, return, sharpe, max_drawdown.
    """
    rows = []
    for year, group in returns.groupby(returns.index.year):
        metrics = compute_all_metrics(group)
        rows.append({
            "year": year,
            "return": metrics["cagr"],
            "sharpe": metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
        })

    result = pd.DataFrame(rows)
    logger.info("attribution_by_year: %d years", len(result))
    return result
