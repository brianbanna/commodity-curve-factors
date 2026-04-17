"""Seasonal pattern extraction and deseasonalisation.

ISO-week-based seasonal adjustment:

- ``compute_seasonal_pattern`` — average by ISO week number over a trailing
  window of years.  Uses only data up to and including the last observation,
  so there is no lookahead within the estimation window.
- ``deseasonalise`` — subtract the estimated seasonal component from a daily
  series using ISO week mapping.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_seasonal_pattern(
    series: pd.Series,
    lookback_years: int = 5,
) -> pd.Series:
    """Estimate the seasonal pattern as an ISO-week average.

    Computes the mean value of *series* for each ISO week number (1-53),
    using only the trailing ``lookback_years`` of data (or all data if fewer
    years are available).  The result is a per-week average indexed by week
    number.

    Parameters
    ----------
    series : pd.Series
        Daily (or any frequency) time series with a :class:`pandas.DatetimeIndex`.
    lookback_years : int
        Number of trailing calendar years to include in the average.  Default 5.

    Returns
    -------
    pd.Series
        ISO week number → average value.  Index dtype is int64 (week numbers
        1 through 52 or 53).  Length is 50-54 depending on the ISO calendar.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")

    end_date = series.index.max()
    start_date = end_date - pd.DateOffset(years=lookback_years)
    window = series.loc[series.index >= start_date]

    # Assign ISO week numbers
    iso_weeks = window.index.isocalendar().week.astype(int)
    temp = pd.Series(window.values, index=iso_weeks, name=series.name)
    pattern = temp.groupby(level=0).mean()
    pattern.index.name = "iso_week"

    logger.info(
        "compute_seasonal_pattern: lookback=%d yr, %d observations, %d week buckets",
        lookback_years,
        len(window),
        len(pattern),
    )
    return pattern


def deseasonalise(
    series: pd.Series,
    seasonal_pattern: pd.Series,
) -> pd.Series:
    """Remove the seasonal component from a daily series.

    Maps each observation to its ISO week number and subtracts the
    corresponding value from ``seasonal_pattern``.  Observations whose
    week number is not found in the pattern are left unchanged (NaN
    subtraction → NaN, which signals missing seasonal info).

    Parameters
    ----------
    series : pd.Series
        Daily time series with a :class:`pandas.DatetimeIndex`.
    seasonal_pattern : pd.Series
        ISO week → average value, as returned by ``compute_seasonal_pattern``.

    Returns
    -------
    pd.Series
        Deseasonalised series with the same index and name as *series*.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")

    iso_weeks = series.index.isocalendar().week.astype(int)
    seasonal_component = iso_weeks.map(seasonal_pattern)
    seasonal_component = pd.Series(seasonal_component.values, index=series.index, dtype=float)

    result = series - seasonal_component
    result.name = series.name

    logger.info(
        "deseasonalise: %d observations, %.1f%% seasonal component applied",
        len(series),
        seasonal_component.notna().mean() * 100,
    )
    return result
