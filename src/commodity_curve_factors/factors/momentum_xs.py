"""Cross-sectional momentum (XSMOM) factor.

Ranks each commodity's trailing return against all other commodities on the
same day, producing a signal in [0, 1].  A rank of 1.0 means this commodity
had the highest trailing return in the cross-section that day (strongest
relative momentum); 0.0 means the weakest.

The trailing return is sourced from
:func:`~commodity_curve_factors.factors.momentum_ts.compute_trailing_return`
and cross-sectionally ranked with
:func:`~commodity_curve_factors.factors.transforms.cross_sectional_rank`.
"""

import logging

import pandas as pd

from commodity_curve_factors.factors.momentum_ts import compute_trailing_return
from commodity_curve_factors.factors.transforms import cross_sectional_rank

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def xsmom_signal(
    prices: pd.DataFrame,
    lookback_days: int = 126,
) -> pd.DataFrame:
    """Cross-sectional momentum: trailing return ranked to [0, 1] each day.

    At each date the trailing log return for every commodity is ranked within
    the cross-section.  NaN prices are excluded from ranking on that date
    (they remain NaN in the output).

    Parameters
    ----------
    prices : pd.DataFrame
        DatetimeIndex, columns = commodity symbols, values = front-month prices.
    lookback_days : int
        Lookback window for the trailing log return.  Default 126 (~6 months).

    Returns
    -------
    pd.DataFrame
        Same shape, index, and column names as *prices*.  Values in [0, 1]
        where non-NaN.  NaN where the price or its lagged counterpart is NaN.
    """
    returns = compute_trailing_return(prices, lookback_days)
    signal: pd.DataFrame = returns.apply(cross_sectional_rank, axis=1)

    logger.info(
        "xsmom_signal: lookback=%d, %d commodities, %d dates, %.1f%% non-NaN",
        lookback_days,
        len(signal.columns),
        len(signal),
        signal.notna().mean().mean() * 100,
    )
    return signal
