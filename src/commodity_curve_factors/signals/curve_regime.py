"""Curve regime classification based on convenience yield percentiles.

Classifies each commodity-month into one of five regimes using expanding-window
percentile ranks of convenience yield. No lookahead — each observation only
sees data up to and including itself.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS = [10, 30, 70, 90]

_DEFAULT_POSITION_MAP: dict[str, float] = {
    "crisis_backwardation": 1.0,
    "mild_backwardation": 0.5,
    "balanced": 0.0,
    "mild_contango": 0.0,
    "deep_contango": -0.5,
}

_REGIME_NAMES = [
    "deep_contango",
    "mild_contango",
    "balanced",
    "mild_backwardation",
    "crisis_backwardation",
]


def classify_regime(
    monthly_cy: pd.DataFrame,
    thresholds: list[int] | None = None,
) -> pd.DataFrame:
    """Classify each commodity-month into a curve regime.

    Uses expanding-window percentile rank of convenience yield.

    Parameters
    ----------
    monthly_cy : pd.DataFrame
        Monthly convenience yield (dates x commodities).
    thresholds : list[int] or None
        Percentile boundaries [p1, p2, p3, p4] defining 5 regimes.
        Default ``[10, 30, 70, 90]``.

    Returns
    -------
    pd.DataFrame
        Regime labels (dates x commodities). Values are one of:
        ``"deep_contango"``, ``"mild_contango"``, ``"balanced"``,
        ``"mild_backwardation"``, ``"crisis_backwardation"``.
    """
    if thresholds is None:
        thresholds = _DEFAULT_THRESHOLDS.copy()

    result = pd.DataFrame(index=monthly_cy.index, columns=monthly_cy.columns, dtype=object)

    for col in monthly_cy.columns:
        series = monthly_cy[col]
        for i in range(len(series)):
            val = series.iloc[i]
            if np.isnan(val):
                continue
            history = series.iloc[: i + 1].dropna()
            if len(history) < 12:  # need at least 12 months for meaningful percentiles
                continue
            pct = (history < val).sum() / len(history) * 100

            if pct < thresholds[0]:
                result.iloc[i, result.columns.get_loc(col)] = "deep_contango"
            elif pct < thresholds[1]:
                result.iloc[i, result.columns.get_loc(col)] = "mild_contango"
            elif pct < thresholds[2]:
                result.iloc[i, result.columns.get_loc(col)] = "balanced"
            elif pct < thresholds[3]:
                result.iloc[i, result.columns.get_loc(col)] = "mild_backwardation"
            else:
                result.iloc[i, result.columns.get_loc(col)] = "crisis_backwardation"

    logger.info("classify_regime: %d months, %d commodities", len(result), len(result.columns))
    return result


def regime_to_position(
    regimes: pd.DataFrame,
    position_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Map regime labels to base position weights.

    Parameters
    ----------
    regimes : pd.DataFrame
        Regime labels from ``classify_regime``.
    position_map : dict[str, float] or None
        Mapping from regime name to position weight.
        Default: crisis_backwardation=1.0, mild_backwardation=0.5,
        balanced=0.0, mild_contango=0.0, deep_contango=-0.5.

    Returns
    -------
    pd.DataFrame
        Position weights (dates x commodities).
    """
    if position_map is None:
        position_map = _DEFAULT_POSITION_MAP.copy()

    # Map labels to floats; unmapped labels (including NaN cells) become NaN
    result = regimes.apply(lambda col: col.map(position_map))
    logger.info("regime_to_position: %d rows, %d commodities", len(result), len(result.columns))
    return result
