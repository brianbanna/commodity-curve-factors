"""Regime classifier based on VIX level.

Classifies market regimes into three states — calm, moderate, turbulent —
using VIX as a proxy for volatility/fear.  Used by the regime-conditioned
strategy to tilt factor weights across environments.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS = [15.0, 25.0]
_LABELS = ["calm", "moderate", "turbulent"]


def classify_regime(
    vix: pd.Series,
    thresholds: list[float] | None = None,
) -> pd.Series:
    """Classify VIX into regime labels.

    Default thresholds [15, 25]:
    - VIX < 15: "calm"
    - 15 <= VIX < 25: "moderate"
    - VIX >= 25: "turbulent"

    Parameters
    ----------
    vix : pd.Series
        VIX level series with DatetimeIndex.  NaN values are preserved.
    thresholds : list[float] | None
        Two monotonically increasing thresholds that define the boundaries
        between the three regimes.  Defaults to [15, 25].

    Returns
    -------
    pd.Series
        Same index as *vix*, values ∈ {"calm", "moderate", "turbulent"}.
        NaN VIX → NaN regime (dtype object).
    """
    if thresholds is None:
        thresholds = _DEFAULT_THRESHOLDS

    if len(thresholds) != 2:
        raise ValueError(f"thresholds must have exactly 2 elements, got {len(thresholds)}")

    lo, hi = thresholds
    if lo >= hi:
        raise ValueError(f"thresholds must be strictly increasing, got [{lo}, {hi}]")

    result = pd.Series(np.nan, index=vix.index, dtype=object)
    valid = vix.notna()

    result[valid & (vix < lo)] = "calm"
    result[valid & (vix >= lo) & (vix < hi)] = "moderate"
    result[valid & (vix >= hi)] = "turbulent"

    logger.info(
        "classify_regime: %d dates, thresholds=[%.1f, %.1f], "
        "calm=%d, moderate=%d, turbulent=%d, nan=%d",
        len(vix),
        lo,
        hi,
        int((result == "calm").sum()),
        int((result == "moderate").sum()),
        int((result == "turbulent").sum()),
        int(result.isna().sum()),
    )

    return result
