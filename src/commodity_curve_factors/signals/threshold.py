"""Time-series threshold signal.

Converts per-commodity z-scores into directional signals (+1 / 0 / -1)
using a symmetric z-score threshold.  Used primarily for the TSMOM strategy.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def threshold_signal(
    zscore: pd.DataFrame,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Per-commodity time-series signal: +1 if z > threshold, -1 if z < -threshold, 0 otherwise.

    For TSMOM: if a commodity has been going up (z > 0), go long (+1).
    If going down (z < 0), go short (-1). Zero if ambiguous.

    Parameters
    ----------
    zscore : pd.DataFrame
        DatetimeIndex × commodity columns, values = factor z-scores.
    threshold : float
        Symmetric entry threshold. Values between -threshold and +threshold
        (exclusive on both sides) produce a flat signal (0). Default 0.0
        means any non-zero z-score generates a directional signal.

    Returns
    -------
    pd.DataFrame
        Same shape, index, and columns as *zscore*.
        Values ∈ {-1, 0, +1}.  NaN inputs remain NaN in the output.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")

    result = pd.DataFrame(
        np.where(zscore.isna(), np.nan, 0.0),
        index=zscore.index,
        columns=zscore.columns,
    )

    result = result.where(zscore.isna() | ~(zscore > threshold), 1.0)
    result = result.where(zscore.isna() | ~(zscore < -threshold), -1.0)

    logger.info(
        "threshold_signal: %d dates, %d commodities, threshold=%.3f",
        len(zscore),
        zscore.shape[1],
        threshold,
    )

    return result
