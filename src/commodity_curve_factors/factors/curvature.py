"""Curvature factor.

Measures the butterfly shape of each commodity's forward curve: whether the
6-month point is above or below the line connecting the 1-month and 12-month
tenors. Positive z-score means the mid-curve is concave-up (bowed below the
front-to-back line) relative to history; negative means concave-down (humped).

The raw metric is computed by
:func:`~commodity_curve_factors.curves.metrics.compute_curvature`
and then z-scored with an expanding window via
:func:`~commodity_curve_factors.factors.transforms.expanding_zscore_df`.
"""

import logging

import pandas as pd

from commodity_curve_factors.curves.metrics import compute_curvature
from commodity_curve_factors.factors.transforms import expanding_zscore_df

logger = logging.getLogger(__name__)


def compute_curvature_factor(
    curves: dict[str, pd.DataFrame],
    min_periods: int = 252,
) -> pd.DataFrame:
    """Curvature factor: expanding z-score of F1M - 2*F6M + F12M per commodity.

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
        DatetimeIndex, columns = commodity symbols, values = z-scored curvature.
        Positive z-score = mid-curve bowed below the F1M–F12M line (concave-up)
        more than historical average.
    """
    raw = pd.DataFrame({sym: compute_curvature(curve) for sym, curve in curves.items()})

    factor = expanding_zscore_df(raw, min_periods=min_periods)

    logger.info(
        "compute_curvature_factor: %d commodities, %d dates, %.1f%% non-NaN",
        len(factor.columns),
        len(factor),
        factor.notna().mean().mean() * 100,
    )
    return factor
