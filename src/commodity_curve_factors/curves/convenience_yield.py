"""Convenience yield estimation from the cost-of-carry model.

The cost-of-carry model relates spot and futures prices:

    F(T) = S * exp((r - y + c) * T)

Rearranging for convenience yield:

    y = r + c - ln(F(T) / S) / T

where S = F1M (spot proxy), F(T) = futures at tenor T, r = risk-free rate,
c = storage cost, y = convenience yield.

High convenience yield signals physical market tightness (scarcity premium).
Low/negative convenience yield signals surplus.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MONTHS_PER_YEAR = 12


def estimate_storage_cost(
    curves: dict[str, pd.DataFrame],
    is_end: str = "2017-12-31",
    tenor: str = "F6M",
) -> dict[str, float]:
    """Calibrate per-commodity storage cost proxy from in-sample contango depth.

    Storage cost is estimated as the median annualised contango
    ``ln(F(T)/S) / T`` over the in-sample period. For backwardated
    commodities this will be negative — floored at 0.0.

    Parameters
    ----------
    curves : dict[str, pd.DataFrame]
        Keyed by commodity symbol. Each DataFrame has columns including
        ``"F1M"`` and the specified *tenor* column, with a DatetimeIndex.
    is_end : str
        Last date of the in-sample calibration window.
    tenor : str
        Futures tenor column to use. Default ``"F6M"``.

    Returns
    -------
    dict[str, float]
        Per-commodity annualised storage cost estimate.
    """
    tenor_months = int(tenor.replace("F", "").replace("M", ""))
    t_years = tenor_months / _MONTHS_PER_YEAR

    result: dict[str, float] = {}
    for sym, df in curves.items():
        is_df = df.loc[:is_end]
        if "F1M" not in is_df.columns or tenor not in is_df.columns:
            logger.warning("estimate_storage_cost: %s missing F1M or %s", sym, tenor)
            result[sym] = 0.0
            continue
        ratio = is_df[tenor] / is_df["F1M"]
        ratio = ratio[ratio > 0]
        if len(ratio) == 0:
            result[sym] = 0.0
            continue
        contango_depth = np.log(ratio) / t_years
        cost = max(0.0, float(contango_depth.median()))
        result[sym] = cost

    logger.info(
        "estimate_storage_cost: %d commodities, is_end=%s, tenor=%s",
        len(result),
        is_end,
        tenor,
    )
    return result


def compute_convenience_yield(
    curves: dict[str, pd.DataFrame],
    risk_free: pd.Series,
    storage_costs: dict[str, float],
    tenor: str = "F6M",
) -> pd.DataFrame:
    """Compute daily convenience yield for each commodity.

    Parameters
    ----------
    curves : dict[str, pd.DataFrame]
        Keyed by commodity symbol. Each has ``"F1M"`` and *tenor* columns.
    risk_free : pd.Series
        Annualised risk-free rate (percentage, e.g. 2.0 for 2%).
        DatetimeIndex aligned to curve dates.
    storage_costs : dict[str, float]
        Per-commodity annualised storage cost from ``estimate_storage_cost``.
    tenor : str
        Futures tenor column. Default ``"F6M"``.

    Returns
    -------
    pd.DataFrame
        Daily convenience yield (dates x commodities). Annualised fraction.
    """
    tenor_months = int(tenor.replace("F", "").replace("M", ""))
    t_years = tenor_months / _MONTHS_PER_YEAR

    all_dates = sorted(set().union(*(df.index for df in curves.values())))
    idx = pd.DatetimeIndex(all_dates)

    rf = risk_free.reindex(idx).ffill() / 100.0  # convert percentage to fraction

    cy_dict: dict[str, pd.Series] = {}
    for sym, df in curves.items():
        if "F1M" not in df.columns or tenor not in df.columns:
            continue
        spot = df["F1M"].reindex(idx)
        fut = df[tenor].reindex(idx)
        c = storage_costs.get(sym, 0.0)
        r = rf

        ratio = fut / spot
        # Guard against non-positive ratios
        valid = ratio > 0
        log_ratio = pd.Series(np.nan, index=idx)
        log_ratio[valid] = np.log(ratio[valid])

        y = r + c - log_ratio / t_years
        cy_dict[sym] = y

    result = pd.DataFrame(cy_dict, index=idx)
    logger.info(
        "compute_convenience_yield: %d commodities, %d dates, %.1f%% non-NaN",
        len(cy_dict),
        len(result),
        result.notna().mean().mean() * 100,
    )
    return result


def monthly_convenience_yield(daily_cy: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily convenience yield to monthly median.

    Parameters
    ----------
    daily_cy : pd.DataFrame
        Daily convenience yield (dates x commodities).

    Returns
    -------
    pd.DataFrame
        Monthly convenience yield indexed by month-end date.
    """
    result = daily_cy.resample("ME").median()
    logger.info(
        "monthly_convenience_yield: %d months, %d commodities",
        len(result),
        len(result.columns),
    )
    return result
