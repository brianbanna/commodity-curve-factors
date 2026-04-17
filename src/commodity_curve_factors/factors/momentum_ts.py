"""Time-series momentum (TSMOM) factor.

Measures each commodity's own trend by comparing the current price to its
price ``lookback_days`` ago using log returns.  The raw trailing return is
then z-scored against its own expanding history, producing a signal that is
positive when a commodity is in a stronger uptrend than its historical average.

References
----------
Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum", JFE.
"""

import logging

import numpy as np
import pandas as pd

from commodity_curve_factors.factors.transforms import expanding_zscore_df

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_LOOKBACKS: list[int] = [21, 63, 126, 252]


def compute_trailing_return(
    prices: pd.DataFrame,
    lookback_days: int,
) -> pd.DataFrame:
    """Trailing log return over ``lookback_days`` trading days.

    Computes::

        r(t) = log(price(t) / price(t - lookback_days))

    Parameters
    ----------
    prices : pd.DataFrame
        DatetimeIndex, columns = commodity symbols, values = front-month prices.
    lookback_days : int
        Number of trading days to look back.  The first ``lookback_days`` rows
        of the output are NaN.

    Returns
    -------
    pd.DataFrame
        Same shape, index, and column names as *prices*.
    """
    returns = pd.DataFrame(
        np.log(prices / prices.shift(lookback_days)), index=prices.index, columns=prices.columns
    )
    logger.debug(
        "compute_trailing_return: lookback=%d, shape=%s, non-NaN=%.1f%%",
        lookback_days,
        returns.shape,
        returns.notna().mean().mean() * 100,
    )
    return returns


def tsmom_signal(
    prices: pd.DataFrame,
    lookback_days: int = 252,
    min_periods: int = 252,
) -> pd.DataFrame:
    """Time-series momentum signal: expanding z-score of trailing log return.

    Positive values indicate the commodity is trending up more strongly than
    its own historical average.

    Parameters
    ----------
    prices : pd.DataFrame
        DatetimeIndex, columns = commodity symbols, values = front-month prices.
    lookback_days : int
        Lookback window for the trailing log return.  Default 252 (~1 year).
    min_periods : int
        Minimum non-NaN observations required before a z-score is emitted.
        Default 252.

    Returns
    -------
    pd.DataFrame
        Same shape, index, and column names as *prices*.  NaN where
        insufficient history or where input prices are NaN.
    """
    raw = compute_trailing_return(prices, lookback_days)
    signal: pd.DataFrame = expanding_zscore_df(raw, min_periods=min_periods)

    logger.info(
        "tsmom_signal: lookback=%d, %d commodities, %d dates, %.1f%% non-NaN",
        lookback_days,
        len(signal.columns),
        len(signal),
        signal.notna().mean().mean() * 100,
    )
    return signal


def tsmom_multi_horizon(
    prices: pd.DataFrame,
    lookbacks: list[int] | None = None,
    min_periods: int = 252,
) -> dict[int, pd.DataFrame]:
    """TSMOM signal at multiple horizons.

    Parameters
    ----------
    prices : pd.DataFrame
        DatetimeIndex, columns = commodity symbols, values = front-month prices.
    lookbacks : list[int] or None
        Lookback windows in trading days.  Default ``[21, 63, 126, 252]``.
    min_periods : int
        Minimum non-NaN observations for each expanding z-score.  Default 252.

    Returns
    -------
    dict[int, pd.DataFrame]
        Keys are lookback values; values are TSMOM signal DataFrames with the
        same index and column names as *prices*.
    """
    if lookbacks is None:
        lookbacks = _DEFAULT_LOOKBACKS

    result: dict[int, pd.DataFrame] = {}
    for lb in lookbacks:
        result[lb] = tsmom_signal(prices, lookback_days=lb, min_periods=min_periods)

    logger.info(
        "tsmom_multi_horizon: horizons=%s, %d commodities, %d dates",
        lookbacks,
        len(prices.columns),
        len(prices),
    )
    return result
