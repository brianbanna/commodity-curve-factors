"""Slope factor.

Measures the steepness of each commodity's forward curve from front to 12-month
tenor. Positive z-score means the curve is more steeply in contango than its
historical average; negative means more backwardated than average.

The raw metric is computed by
:func:`~commodity_curve_factors.curves.metrics.compute_slope`
and then z-scored with an expanding window via
:func:`~commodity_curve_factors.factors.transforms.expanding_zscore_df`.
"""

import logging

import pandas as pd

from commodity_curve_factors.curves.metrics import compute_slope
from commodity_curve_factors.factors.transforms import expanding_zscore_df

logger = logging.getLogger(__name__)


def compute_slope_factor(
    curves: dict[str, pd.DataFrame],
    min_periods: int = 252,
) -> pd.DataFrame:
    """Slope factor: expanding z-score of (F12M - F1M) / F1M per commodity.

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
        DatetimeIndex, columns = commodity symbols, values = z-scored slope.
        Positive z-score = steeper contango than historical average.
        Negative z-score = deeper backwardation than historical average.
    """
    raw = pd.DataFrame({sym: compute_slope(curve) for sym, curve in curves.items()})

    factor: pd.DataFrame = expanding_zscore_df(raw, min_periods=min_periods)

    logger.info(
        "compute_slope_factor: %d commodities, %d dates, %.1f%% non-NaN",
        len(factor.columns),
        len(factor),
        factor.notna().mean().mean() * 100,
    )
    return factor
