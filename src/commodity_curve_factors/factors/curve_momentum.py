"""Curve momentum factor.

Measures the recent change in each commodity's term-structure slope.  A
positive signal means the curve has been steepening (contango deepening or
backwardation flattening) over the lookback window, which empirically predicts
continuation of price trends driven by shifting storage demand.

The raw slope is computed by
:func:`~commodity_curve_factors.curves.metrics.compute_slope`
(``(F12M - F1M) / F1M``), the change is taken over ``lookback_days`` trading
days via ``Series.diff``, and the result is z-scored with an expanding window
via :func:`~commodity_curve_factors.factors.transforms.expanding_zscore_df`.
"""

import logging

import pandas as pd

from commodity_curve_factors.curves.metrics import compute_slope
from commodity_curve_factors.factors.transforms import expanding_zscore_df

logger = logging.getLogger(__name__)


def compute_curve_momentum_factor(
    curves: dict[str, pd.DataFrame],
    lookback_days: int = 60,
    min_periods: int = 252,
) -> pd.DataFrame:
    """Curve momentum factor: expanding z-score of slope change.

    curve_momentum(t) = slope(t) - slope(t - lookback_days)

    Where slope = (F12M - F1M) / F1M. A positive curve momentum means
    the term structure is steepening (contango deepening or backwardation
    flattening). The raw difference is then z-scored with an expanding window.

    Parameters
    ----------
    curves : dict[str, pd.DataFrame]
        Output of ``builder.load_curves()``.
    lookback_days : int
        Number of trading days to look back for the slope change. Default 60.
    min_periods : int
        Minimum observations for the expanding z-score. Default 252.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, columns = commodity symbols, values = z-scored
        curve momentum.
    """
    slope_raw = pd.DataFrame({sym: compute_slope(curve) for sym, curve in curves.items()})

    slope_change = slope_raw.diff(lookback_days)

    factor: pd.DataFrame = expanding_zscore_df(slope_change, min_periods=min_periods)

    logger.info(
        "compute_curve_momentum_factor: %d commodities, %d dates, %.1f%% non-NaN"
        " (lookback_days=%d)",
        len(factor.columns),
        len(factor),
        factor.notna().mean().mean() * 100,
        lookback_days,
    )
    return factor
