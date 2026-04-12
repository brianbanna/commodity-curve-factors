"""Mathematical primitives used by every factor module.

Four pure transform functions:

- ``expanding_zscore``     -- no-lookahead expanding-window z-score (Series)
- ``expanding_zscore_df``  -- column-wise wrapper for DataFrames
- ``cross_sectional_rank`` -- rank one cross-section, scaled to [0, 1]
- ``percentile_rank``      -- rolling empirical CDF rank, scaled to [0, 1]

All functions are stateless and side-effect-free.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def expanding_zscore(
    series: pd.Series,
    min_periods: int = 252,
) -> pd.Series:
    """Expanding-window z-score with no lookahead.

    At each time t the z-score is computed as::

        z(t) = (x(t) - mean(x[0:t+1])) / std(x[0:t+1], ddof=1)

    The expanding mean and std at position t include x(t) itself but
    nothing beyond t, so there is strictly no lookahead bias.

    Parameters
    ----------
    series : pd.Series
        Input time series.  Any index type is accepted.
    min_periods : int
        Minimum number of non-NaN observations required before a z-score
        is emitted.  Positions with fewer observations return NaN.
        Default 252 (~1 trading year).

    Returns
    -------
    pd.Series
        Same index and name as *series*.  NaN where insufficient history
        or where the input is NaN.  Where the expanding std is zero
        (constant sub-series), the z-score is 0.0 rather than NaN/inf.
    """
    exp = series.expanding(min_periods=min_periods)
    mean = exp.mean()
    std = exp.std(ddof=1)
    z = (series - mean) / std
    # Constant sub-series → std is exactly 0 (not NaN) → replace with 0.0.
    # We must NOT replace positions where std is NaN (below min_periods).
    z = z.where(~(std == 0), 0.0)
    # Restore NaN where the original series was NaN
    z = z.where(series.notna(), np.nan)
    return z


def expanding_zscore_df(
    df: pd.DataFrame,
    min_periods: int = 252,
) -> pd.DataFrame:
    """Apply ``expanding_zscore`` to each column of a DataFrame independently.

    Each column is z-scored using its own expanding mean and std (time-series
    z-score, not cross-sectional normalization).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.  Columns are processed independently.
    min_periods : int
        Passed to ``expanding_zscore`` for every column.

    Returns
    -------
    pd.DataFrame
        Same shape, index, and column names as *df*.
    """
    return df.apply(expanding_zscore, min_periods=min_periods)


def cross_sectional_rank(row: pd.Series) -> pd.Series:
    """Rank values within a cross-section, scaled to [0, 1].

    NaN values are excluded from ranking (they remain NaN in output).
    Ties receive the average of the ranks they would occupy (method="average").

    Parameters
    ----------
    row : pd.Series
        One cross-section — e.g. one day's factor values across commodities.

    Returns
    -------
    pd.Series
        Same index as *row*.  Values in [0, 1] where 0.0 is the lowest
        non-NaN value and 1.0 is the highest.  Single non-NaN value → 0.5.
        All NaN → all NaN.
    """
    n_valid = int(row.notna().sum())

    if n_valid == 0:
        return row.copy()

    if n_valid == 1:
        # Single observation: no ranking possible → canonical midpoint
        return row.where(row.isna(), 0.5)

    ranked = row.rank(method="average", na_option="keep")
    # Scale: rank 1 → 0.0, rank n_valid → 1.0
    scaled = (ranked - 1) / (n_valid - 1)
    return scaled


def percentile_rank(
    series: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Rolling percentile rank of the current value within its trailing window.

    At each time t, computes the fraction of the trailing ``window``
    observations (including t) that are less than or equal to x(t).
    This is the empirical CDF evaluated at x(t), producing a value in [0, 1].

    Used by the CFTC positioning factor: "what percentile is this week's
    net speculative position relative to the last ``window`` weeks?"

    Parameters
    ----------
    series : pd.Series
        Input time series.
    window : int
        Lookback window size (number of observations, inclusive of current).
        Default 252 (~1 trading year).

    Returns
    -------
    pd.Series
        Same index as *series*.  NaN where fewer than ``window`` non-NaN
        observations are available.  Values in [0, 1].
    """

    def _pctile(arr: np.ndarray) -> float:
        """Fraction of window values that are <= the current (last) value."""
        current = arr[-1]
        if np.isnan(current):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        return float(np.sum(valid <= current)) / len(valid)

    return series.rolling(window=window, min_periods=window).apply(_pctile, raw=True)
