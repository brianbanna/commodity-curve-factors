"""Curve metrics derived from daily interpolated term structures.

All formulas are taken verbatim from ``configs/curve.yaml`` ``metrics:`` block
and must not diverge from that source of truth.

Current formulas (as of curve.yaml):
  slope:       (F12M - F1M) / F1M
  front_slope: (F3M - F1M) / F1M
  curvature:   F1M - 2*F6M + F12M
  carry:       (F1M - F2M) / F2M * 12
  term_carry:  (F1M - F12M) / F12M

Note: ``compute_curve_momentum`` is NOT implemented here — it is a temporal
factor that belongs in the ``factors/`` subpackage (Task 3.5).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_METRIC_NAMES = ("slope", "front_slope", "curvature", "carry", "term_carry")


def compute_slope(curve: pd.DataFrame) -> pd.Series:
    """Slope of the term structure: (F12M - F1M) / F1M.

    Parameters
    ----------
    curve : pd.DataFrame
        DatetimeIndex × tenor columns (must include ``F1M`` and ``F12M``).

    Returns
    -------
    pd.Series
        Daily slope values; NaN where either tenor is NaN.
    """
    return (curve["F12M"] - curve["F1M"]) / curve["F1M"]


def compute_front_slope(curve: pd.DataFrame) -> pd.Series:
    """Front slope: (F3M - F1M) / F1M.

    Parameters
    ----------
    curve : pd.DataFrame
        Must include columns ``F1M`` and ``F3M``.

    Returns
    -------
    pd.Series
        Daily front-slope values.
    """
    return (curve["F3M"] - curve["F1M"]) / curve["F1M"]


def compute_curvature(curve: pd.DataFrame) -> pd.Series:
    """Butterfly curvature: F1M - 2*F6M + F12M.

    Parameters
    ----------
    curve : pd.DataFrame
        Must include columns ``F1M``, ``F6M``, and ``F12M``.

    Returns
    -------
    pd.Series
        Daily curvature values.
    """
    return curve["F1M"] - 2 * curve["F6M"] + curve["F12M"]


def compute_carry(curve: pd.DataFrame) -> pd.Series:
    """Annualised carry: (F1M - F2M) / F2M * 12.

    Parameters
    ----------
    curve : pd.DataFrame
        Must include columns ``F1M`` and ``F2M``.

    Returns
    -------
    pd.Series
        Daily carry values (annualised).
    """
    return (curve["F1M"] - curve["F2M"]) / curve["F2M"] * 12


def compute_term_carry(curve: pd.DataFrame) -> pd.Series:
    """Term carry: (F1M - F12M) / F12M.

    Parameters
    ----------
    curve : pd.DataFrame
        Must include columns ``F1M`` and ``F12M``.

    Returns
    -------
    pd.Series
        Daily term-carry values.
    """
    return (curve["F1M"] - curve["F12M"]) / curve["F12M"]


def compute_all_metrics(
    curves: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Compute all 5 curve metrics across the commodity universe.

    For each metric, produces a wide DataFrame (DatetimeIndex × commodity
    symbols) aligned on the union of all dates.

    Parameters
    ----------
    curves : dict[str, pd.DataFrame]
        Keyed by commodity symbol; values are daily curve DataFrames from
        :func:`~commodity_curve_factors.curves.builder.build_curve`.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``"slope"``, ``"front_slope"``, ``"curvature"``, ``"carry"``,
        ``"term_carry"``.  Each value is a DataFrame (date × commodity).
    """
    _compute_fn = {
        "slope": compute_slope,
        "front_slope": compute_front_slope,
        "curvature": compute_curvature,
        "carry": compute_carry,
        "term_carry": compute_term_carry,
    }

    per_metric: dict[str, dict[str, pd.Series]] = {name: {} for name in _METRIC_NAMES}

    for symbol, curve in curves.items():
        for name, fn in _compute_fn.items():
            try:
                per_metric[name][symbol] = fn(curve)
            except KeyError as exc:
                logger.warning("Cannot compute %s for %s — missing column %s", name, symbol, exc)

    result: dict[str, pd.DataFrame] = {}
    for name in _METRIC_NAMES:
        if per_metric[name]:
            df = pd.DataFrame(per_metric[name])
            df.index.name = "trade_date"
            result[name] = df
            logger.info("Metric %s: shape=%s", name, df.shape)
        else:
            result[name] = pd.DataFrame()

    return result
