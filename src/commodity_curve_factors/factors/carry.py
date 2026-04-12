"""Carry factor.

Measures the annualised roll return embedded in each commodity's term structure.
Positive z-score means the market is more backwardated than its historical
average (earns positive carry); negative means deeper contango than average.

The raw metric is computed by
:func:`~commodity_curve_factors.curves.metrics.compute_carry`
and then z-scored with an expanding window via
:func:`~commodity_curve_factors.factors.transforms.expanding_zscore_df`.
"""

import logging

import pandas as pd

from commodity_curve_factors.curves.metrics import compute_carry
from commodity_curve_factors.factors.transforms import expanding_zscore_df

logger = logging.getLogger(__name__)


def compute_carry_factor(
    curves: dict[str, pd.DataFrame],
    min_periods: int = 252,
) -> pd.DataFrame:
    """Carry factor: expanding z-score of annualized carry per commodity.

    Raw carry = (F1M - F2M) / F2M * 12  (from curves/metrics.py)

    Parameters
    ----------
    curves : dict[str, pd.DataFrame]
        Output of ``builder.load_curves()`` — keyed by commodity symbol,
        each DataFrame has DatetimeIndex and tenor columns (F1M, F2M, ...).
    min_periods : int
        Minimum observations for the expanding z-score. Default 252.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, columns = commodity symbols, values = z-scored carry.
        Positive z-score = more backwardated than historical average.
    """
    raw = pd.DataFrame({sym: compute_carry(curve) for sym, curve in curves.items()})

    factor: pd.DataFrame = expanding_zscore_df(raw, min_periods=min_periods)

    logger.info(
        "compute_carry_factor: %d commodities, %d dates, %.1f%% non-NaN",
        len(factor.columns),
        len(factor),
        factor.notna().mean().mean() * 100,
    )
    return factor
