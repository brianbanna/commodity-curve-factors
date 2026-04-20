"""Layer 2: Curve transition momentum with TSMOM confirmation gate.

Detects structural shifts in the convenience-yield curve (tightening vs.
loosening) and converts them into directional positions.

Signal pipeline
---------------
1. Forward-fill monthly CY onto a business-day grid.
2. Expanding z-score the daily CY series (no lookahead).
3. Compute ``lookback``-day difference of the z-scored series — positive
   values signal tightening (market moving toward backwardation), negative
   values signal loosening (market moving toward contango).
4. Apply an expanding-std threshold and a TSMOM confirmation gate:
   - Tightening (signal > +threshold) AND TSMOM > 0  → long  (+1)
   - Loosening  (signal < −threshold) AND TSMOM ≤ 0  → short (−1)
   - Conflicts or below threshold                     → flat   (0)
"""

import logging

import numpy as np
import pandas as pd

from commodity_curve_factors.factors.transforms import expanding_zscore_df

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_transition_signal(
    monthly_cy: pd.DataFrame,
    lookback: int = 63,
) -> pd.DataFrame:
    """Curve transition momentum signal on a daily business-day grid.

    Parameters
    ----------
    monthly_cy : pd.DataFrame
        Monthly convenience yield (month-end dates x commodities).
    lookback : int
        Number of business days over which to measure the change in
        z-scored CY.  Default 63 (~3 months).

    Returns
    -------
    pd.DataFrame
        Daily transition signal indexed on business days spanning the range
        of ``monthly_cy``.  NaN where the expanding z-score has insufficient
        history (< 252 observations) or within the first ``lookback`` days.
    """
    # Step 1: forward-fill monthly CY to a daily business-day grid
    start = monthly_cy.index.min()
    end = monthly_cy.index.max()
    daily_idx = pd.bdate_range(start, end)

    combined_idx = daily_idx.union(monthly_cy.index).sort_values()
    daily_cy = monthly_cy.reindex(combined_idx).ffill().reindex(daily_idx)

    # Step 2: expanding z-score (no lookahead, min 252 observations)
    z = expanding_zscore_df(daily_cy, min_periods=252)

    # Step 3: lookback-day diff of z-scored series
    signal = z.diff(lookback)

    logger.info(
        "compute_transition_signal: lookback=%d, %d daily rows, %d commodities, %.1f%% non-NaN",
        lookback,
        len(signal),
        len(signal.columns),
        signal.notna().mean().mean() * 100,
    )
    return signal


def transition_to_position(
    transition: pd.DataFrame,
    tsmom: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Map transition signal to {-1, 0, +1} positions with TSMOM gate.

    At each date the expanding standard deviation of the transition signal is
    computed; positions are opened when the signal exceeds ±``threshold``
    expanding std.  The TSMOM series acts as a confirmation gate: longs require
    TSMOM > 0, shorts require TSMOM ≤ 0.

    Parameters
    ----------
    transition : pd.DataFrame
        Transition momentum signal from ``compute_transition_signal``.
    tsmom : pd.DataFrame
        Daily TSMOM signal (same columns as ``transition``).  Aligned by
        index and columns before comparison.
    threshold : float
        Expanding-std multiplier for the signal threshold.  Default 0.5.

    Returns
    -------
    pd.DataFrame
        Position weights in {-1, 0, +1} with the same index and columns as
        ``transition``.  NaN where both ``transition`` and ``tsmom`` are NaN.
    """
    result = pd.DataFrame(0.0, index=transition.index, columns=transition.columns)

    tsmom_aligned = tsmom.reindex(index=transition.index, columns=transition.columns)

    for col in transition.columns:
        sig = transition[col]
        ts = (
            tsmom_aligned[col]
            if col in tsmom_aligned.columns
            else pd.Series(np.nan, index=transition.index)
        )

        # Expanding std of the transition signal (for dynamic threshold)
        exp_std = sig.expanding(min_periods=63).std(ddof=1)
        band = threshold * exp_std

        long_signal = sig > band
        short_signal = sig < -band
        trend_up = ts > 0
        trend_down = ts <= 0

        pos = pd.Series(0.0, index=transition.index)
        pos = pos.where(~(long_signal & trend_up), 1.0)
        pos = pos.where(~(short_signal & trend_down), -1.0)

        # Where both signal and tsmom are NaN → NaN
        both_nan = sig.isna() & ts.isna()
        pos = pos.where(~both_nan, np.nan)

        result[col] = pos

    logger.info(
        "transition_to_position: threshold=%.2f, %d rows, %d commodities",
        threshold,
        len(result),
        len(result.columns),
    )
    return result
