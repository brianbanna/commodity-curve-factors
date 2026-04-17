"""Structural spread signals: crack spread and livestock spread.

Four public functions:

- ``compute_cy_crack``         -- convenience-yield crack spread (CY-based)
- ``crack_spread_signal``      -- position weights for CL/RB/HO dollar-neutral spread
- ``inventory_overlay``        -- amplify positions when inventory & CY agree
- ``livestock_spread_signal``  -- mean-reversion signal on LC/LH log spread
"""

import logging

import numpy as np
import pandas as pd

from commodity_curve_factors.factors.transforms import expanding_zscore
from commodity_curve_factors.signals.seasonal import (
    compute_seasonal_pattern,
    deseasonalise,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_cy_crack(convenience_yields: pd.DataFrame) -> pd.Series:
    """Compute the crack-spread convenience yield.

    The crack spread CY is defined as::

        cy_crack = cy(RB) + cy(HO) - cy(CL)

    A positive value indicates that petroleum products are expensive
    relative to crude (loose crude, tight products).

    Parameters
    ----------
    convenience_yields : pd.DataFrame
        DataFrame with at minimum columns ``CL``, ``RB``, ``HO`` representing
        the convenience yield of crude oil, reformulated gasoline and heating oil.

    Returns
    -------
    pd.Series
        Crack-spread convenience yield with the same index as the input.
    """
    required = {"CL", "RB", "HO"}
    missing = required - set(convenience_yields.columns)
    if missing:
        raise ValueError(f"convenience_yields is missing required columns: {missing}")

    cy_crack = convenience_yields["RB"] + convenience_yields["HO"] - convenience_yields["CL"]
    cy_crack.name = "cy_crack"

    logger.info(
        "compute_cy_crack: %d observations, mean=%.4f, std=%.4f",
        len(cy_crack),
        cy_crack.mean(),
        cy_crack.std(),
    )
    return cy_crack


def crack_spread_signal(
    cy_crack: pd.Series,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Convert the crack CY series into dollar-neutral positions for CL, RB, HO.

    An expanding z-score of *cy_crack* is computed (no lookahead).  Position
    logic (mean-reverting interpretation):

    * ``z < -threshold``  (crude tight, products cheap) →
      CL = -1.0, RB = +0.5, HO = +0.5
    * ``z > +threshold``  (crude loose, products expensive) →
      CL = +1.0, RB = -0.5, HO = -0.5
    * otherwise → all 0.0

    NaN rows in the z-score propagate as NaN in all three position columns.

    Parameters
    ----------
    cy_crack : pd.Series
        Crack-spread convenience yield, typically from ``compute_cy_crack``.
    threshold : float
        Absolute z-score threshold for triggering a position.  Default 1.5.

    Returns
    -------
    pd.DataFrame
        Columns ``CL``, ``RB``, ``HO``.  Each row sums to 0.0 (dollar-neutral)
        when the position is active.
    """
    z = expanding_zscore(cy_crack, min_periods=252)

    cl = pd.Series(np.nan, index=cy_crack.index, dtype=float)
    rb = pd.Series(np.nan, index=cy_crack.index, dtype=float)
    ho = pd.Series(np.nan, index=cy_crack.index, dtype=float)

    valid = z.notna()

    # Default flat where z is valid
    cl[valid] = 0.0
    rb[valid] = 0.0
    ho[valid] = 0.0

    crude_tight = valid & (z < -threshold)
    cl[crude_tight] = -1.0
    rb[crude_tight] = 0.5
    ho[crude_tight] = 0.5

    crude_loose = valid & (z > threshold)
    cl[crude_loose] = 1.0
    rb[crude_loose] = -0.5
    ho[crude_loose] = -0.5

    result = pd.DataFrame({"CL": cl, "RB": rb, "HO": ho}, index=cy_crack.index)

    n_tight = int(crude_tight.sum())
    n_loose = int(crude_loose.sum())
    logger.info(
        "crack_spread_signal: threshold=%.2f, crude_tight=%d, crude_loose=%d, NaN=%d",
        threshold,
        n_tight,
        n_loose,
        int(z.isna().sum()),
    )
    return result


def inventory_overlay(
    positions: pd.DataFrame,
    inventory_surprise: pd.Series,
    cy_change: pd.Series,
    amplification: float = 1.5,
) -> pd.DataFrame:
    """Amplify positions when inventory surprise and CY change agree.

    Two amplification conditions are checked on common columns:

    * **Draw + tightening**: ``inv_surprise < 0`` AND ``cy_change > 0``
      → multiply long positions by *amplification*.
    * **Build + loosening**: ``inv_surprise > 0`` AND ``cy_change < 0``
      → multiply short positions by *amplification*.

    Only columns present in both *positions* and the overlay series are
    affected.  Rows where neither condition holds are returned unchanged.

    Parameters
    ----------
    positions : pd.DataFrame
        Current position weights (output of ``crack_spread_signal`` or similar).
    inventory_surprise : pd.Series
        Signed inventory surprise; negative means a draw, positive means a build.
    cy_change : pd.Series
        Change in convenience yield; positive means tightening.
    amplification : float
        Multiplier applied when a condition fires.  Default 1.5.

    Returns
    -------
    pd.DataFrame
        Same shape as *positions* with selectively amplified values.
    """
    result = positions.copy()

    # Align series to positions index
    inv = inventory_surprise.reindex(positions.index)
    cy_chg = cy_change.reindex(positions.index)

    draw_tight = (inv < 0) & (cy_chg > 0)
    build_loose = (inv > 0) & (cy_chg < 0)

    common_cols = [c for c in positions.columns if c in positions.columns]

    for col in common_cols:
        pos = result[col]

        # Draw + tightening: amplify longs
        result.loc[draw_tight, col] = pos[draw_tight].where(
            pos[draw_tight] <= 0, pos[draw_tight] * amplification
        )

        # Build + loosening: amplify shorts
        result.loc[build_loose, col] = pos[build_loose].where(
            pos[build_loose] >= 0, pos[build_loose] * amplification
        )

    n_draw = int(draw_tight.sum())
    n_build = int(build_loose.sum())
    logger.info(
        "inventory_overlay: amplification=%.2f, draw+tight=%d, build+loose=%d",
        amplification,
        n_draw,
        n_build,
    )
    return result


def livestock_spread_signal(
    lc_prices: pd.Series,
    lh_prices: pd.Series,
    seasonal_years: int = 5,
    threshold: float = 1.5,
    rolling_window: int = 504,
) -> pd.DataFrame:
    """Mean-reversion signal on the live cattle / lean hog log spread.

    The spread is::

        spread = ln(LC) - ln(LH)

    If at least ``seasonal_years * 252`` observations are available, the
    spread is deseasonalised before z-scoring.  A **rolling** z-score is
    used (instead of expanding) because the cattle-hog spread has structural
    shifts that make the expanding mean inappropriate.

    Position logic (mean-reversion):

    * ``z > threshold``  (cattle expensive relative to hogs) →
      LC = -0.5, LH = +0.5
    * ``z < -threshold`` (cattle cheap relative to hogs) →
      LC = +0.5, LH = -0.5
    * otherwise → 0.0

    NaN rows in z-score propagate as NaN in both position columns.

    Parameters
    ----------
    lc_prices : pd.Series
        Live cattle price series.  Must be positive.
    lh_prices : pd.Series
        Lean hog price series.  Must be positive.
    seasonal_years : int
        Minimum years of history required to apply seasonal adjustment.
        Default 5.
    threshold : float
        Absolute z-score threshold for triggering a position.  Default 1.5.
    rolling_window : int
        Rolling window for z-score (trading days). Default 504 (~2 years).
        Uses rolling instead of expanding to adapt to structural shifts
        in the cattle-hog relationship.

    Returns
    -------
    pd.DataFrame
        Columns ``LC``, ``LH``.  Each row sums to 0.0 when the position is
        active.
    """
    spread = np.log(lc_prices) - np.log(lh_prices)
    spread.name = "lc_lh_spread"

    min_obs = seasonal_years * 252
    if len(spread.dropna()) > min_obs:
        logger.info(
            "livestock_spread_signal: applying seasonal adjustment (n=%d > min_obs=%d)",
            len(spread.dropna()),
            min_obs,
        )
        pattern = compute_seasonal_pattern(spread, lookback_years=seasonal_years)
        spread = deseasonalise(spread, pattern)
    else:
        logger.info(
            "livestock_spread_signal: skipping seasonal adjustment (n=%d <= min_obs=%d)",
            len(spread.dropna()),
            min_obs,
        )

    # Rolling z-score (adapts to structural shifts in the spread)
    roll_mean = spread.rolling(rolling_window, min_periods=rolling_window // 2).mean()
    roll_std = spread.rolling(rolling_window, min_periods=rolling_window // 2).std()
    z = (spread - roll_mean) / roll_std.replace(0, np.nan)

    lc = pd.Series(np.nan, index=spread.index, dtype=float)
    lh = pd.Series(np.nan, index=spread.index, dtype=float)

    valid = z.notna()

    lc[valid] = 0.0
    lh[valid] = 0.0

    cattle_expensive = valid & (z > threshold)
    lc[cattle_expensive] = -0.5
    lh[cattle_expensive] = 0.5

    cattle_cheap = valid & (z < -threshold)
    lc[cattle_cheap] = 0.5
    lh[cattle_cheap] = -0.5

    result = pd.DataFrame({"LC": lc, "LH": lh}, index=spread.index)

    n_long_lh = int(cattle_expensive.sum())
    n_long_lc = int(cattle_cheap.sum())
    logger.info(
        "livestock_spread_signal: threshold=%.2f, long_LH=%d, long_LC=%d, NaN=%d",
        threshold,
        n_long_lh,
        n_long_lc,
        int(z.isna().sum()),
    )
    return result
