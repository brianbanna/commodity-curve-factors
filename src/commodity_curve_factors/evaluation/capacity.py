"""Strategy capacity analysis via volume participation."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def volume_participation(
    weights: pd.DataFrame,
    volume: pd.DataFrame,
    aum: float,
) -> pd.DataFrame:
    """Compute daily volume participation rate per commodity.

    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights (dates x commodities).
    volume : pd.DataFrame
        Daily volume in contracts or notional (dates x commodities).
    aum : float
        Assets under management in same units as volume × price.

    Returns
    -------
    pd.DataFrame
        Participation rate (dates x commodities).
    """
    common_idx = weights.index.intersection(volume.index)
    common_cols = weights.columns.intersection(volume.columns)
    w = weights.loc[common_idx, common_cols].abs()
    v = volume.loc[common_idx, common_cols].replace(0, np.nan)
    participation = (w * aum) / v
    return participation


def capacity_curve(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    aum_range: list[float] | None = None,
) -> pd.DataFrame:
    """Estimate Sharpe ratio degradation at increasing AUM.

    Uses square-root market impact model: cost = k * sqrt(participation).

    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights (dates x commodities).
    returns : pd.DataFrame
        Daily returns (dates x commodities).
    volume : pd.DataFrame
        Daily volume.
    aum_range : list[float] or None
        AUM levels to test. Default [1e6, 5e6, 10e6, 50e6, 100e6, 500e6].

    Returns
    -------
    pd.DataFrame
        Columns: aum, sharpe, impact_bps.
    """
    if aum_range is None:
        aum_range = [1e6, 5e6, 10e6, 50e6, 100e6, 500e6]

    common_idx = weights.index.intersection(returns.index).intersection(volume.index)
    common_cols = weights.columns.intersection(returns.columns).intersection(volume.columns)
    w = weights.loc[common_idx, common_cols]
    r = returns.loc[common_idx, common_cols]
    v = volume.loc[common_idx, common_cols].replace(0, np.nan)

    base_ret = (w * r).sum(axis=1)
    impact_coefficient = 0.1  # 10 bps at 100% participation

    rows = []
    for aum in aum_range:
        participation = (w.abs() * aum) / v
        impact = impact_coefficient * np.sqrt(participation.clip(upper=1.0))
        daily_impact = (impact * w.abs()).sum(axis=1)
        adj_ret = base_ret - daily_impact
        std = float(adj_ret.std())
        sharpe = float(adj_ret.mean() / std * np.sqrt(252)) if std > 0 else 0.0
        mean_impact_bps = float(daily_impact.mean()) * 10000
        rows.append({"aum": aum, "sharpe": sharpe, "impact_bps": mean_impact_bps})

    result = pd.DataFrame(rows)
    logger.info("capacity_curve: %d AUM levels", len(result))
    return result


def estimate_max_capacity(
    weights: pd.DataFrame,
    volume: pd.DataFrame,
    max_participation: float = 0.01,
) -> float:
    """Estimate maximum AUM at a given participation constraint.

    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights (dates x commodities).
    volume : pd.DataFrame
        Daily volume.
    max_participation : float
        Maximum allowed fraction of daily volume. Default 1%.

    Returns
    -------
    float
        Estimated max AUM.
    """
    common_idx = weights.index.intersection(volume.index)
    common_cols = weights.columns.intersection(volume.columns)
    w = weights.loc[common_idx, common_cols].abs()
    v = volume.loc[common_idx, common_cols]

    # For each day, max AUM = min over commodities of (max_participation * volume / weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        max_aum_per_day = (max_participation * v / w).replace([np.inf, -np.inf], np.nan).min(axis=1)

    result = float(max_aum_per_day.median())
    logger.info(
        "estimate_max_capacity: median max AUM=%.0f at %.1f%% participation",
        result,
        max_participation * 100,
    )
    return result
