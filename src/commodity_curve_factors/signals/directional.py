"""Layer 1: Directional positioning with trend filter.

Chains curve-regime classification → position weights → TSMOM trend gate →
monthly-to-daily resampling to produce a daily directional weight matrix.

Signal logic
------------
- Long weights (> 0) are zeroed when TSMOM ≤ 0 (no uptrend confirmation).
- Short weights (< 0) are zeroed when TSMOM > 0 (no downtrend confirmation).
- Zero weights pass through unchanged.
"""

import logging

import pandas as pd

from commodity_curve_factors.signals.curve_regime import classify_regime, regime_to_position

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_trend_filter(
    positions: pd.DataFrame,
    tsmom: pd.DataFrame,
) -> pd.DataFrame:
    """Zero out positions that conflict with the TSMOM trend direction.

    Parameters
    ----------
    positions : pd.DataFrame
        Raw position weights (dates x commodities).  Values are typically
        in {-1.0, -0.5, 0.0, 0.5, 1.0}.
    tsmom : pd.DataFrame
        Time-series momentum signal (dates x commodities).  Positive values
        indicate an uptrend; non-positive values indicate no uptrend.

    Returns
    -------
    pd.DataFrame
        Filtered positions.  Long positions (> 0) where TSMOM ≤ 0 are set
        to 0.0.  Short positions (< 0) where TSMOM > 0 are set to 0.0.
    """
    result = positions.copy()

    # Align on shared columns and index
    shared_cols = positions.columns.intersection(tsmom.columns)
    tsmom_aligned = tsmom.reindex(index=positions.index, columns=positions.columns)

    # Zero longs when TSMOM <= 0
    long_mask = result[shared_cols] > 0
    trend_neg = tsmom_aligned[shared_cols] <= 0
    result[shared_cols] = result[shared_cols].where(~(long_mask & trend_neg), 0.0)

    # Zero shorts when TSMOM > 0
    short_mask = result[shared_cols] < 0
    trend_pos = tsmom_aligned[shared_cols] > 0
    result[shared_cols] = result[shared_cols].where(~(short_mask & trend_pos), 0.0)

    logger.info(
        "apply_trend_filter: %d rows, %d cols, %.1f%% zeroed by filter",
        len(result),
        len(result.columns),
        (result[shared_cols] == 0.0).mean().mean() * 100,
    )
    return result


def resample_weights_monthly(
    monthly_weights: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Forward-fill monthly weights onto a daily business-day index.

    Each month-end weight is applied to every business day in that month (and
    subsequent months until the next month-end observation).

    Parameters
    ----------
    monthly_weights : pd.DataFrame
        Month-end weights (month-end dates x commodities).
    daily_index : pd.DatetimeIndex
        Target daily index (typically business days).

    Returns
    -------
    pd.DataFrame
        Daily weights with shape ``(len(daily_index), n_commodities)``.
        Rows before the first monthly observation are NaN.
    """
    combined_index = daily_index.union(monthly_weights.index).sort_values()
    expanded = monthly_weights.reindex(combined_index).ffill()
    result = expanded.reindex(daily_index)

    logger.info(
        "resample_weights_monthly: %d monthly → %d daily rows",
        len(monthly_weights),
        len(result),
    )
    return result


def build_directional_weights(
    monthly_cy: pd.DataFrame,
    tsmom: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    thresholds: list[int] | None = None,
    position_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build daily directional position weights from convenience yield and TSMOM.

    Pipeline:
    1. ``classify_regime`` — classify each commodity-month by CY percentile.
    2. ``regime_to_position`` — map regime labels to base position weights.
    3. ``apply_trend_filter`` — zero conflicting positions using monthly TSMOM.
    4. ``resample_weights_monthly`` — forward-fill to the target daily index.

    Parameters
    ----------
    monthly_cy : pd.DataFrame
        Monthly convenience yield (month-end dates x commodities).
    tsmom : pd.DataFrame
        TSMOM signal on the *same monthly* frequency as ``monthly_cy``
        (month-end dates x commodities).  Resampled to monthly internally if
        needed by aligning on ``monthly_cy`` index.
    daily_index : pd.DatetimeIndex
        Target daily business-day index for the output.
    thresholds : list[int] or None
        Percentile thresholds forwarded to ``classify_regime``.
    position_map : dict[str, float] or None
        Position map forwarded to ``regime_to_position``.

    Returns
    -------
    pd.DataFrame
        Daily position weights (``len(daily_index)`` rows x commodities).
    """
    regimes = classify_regime(monthly_cy, thresholds=thresholds)
    positions = regime_to_position(regimes, position_map=position_map)

    # Align TSMOM to the monthly index of positions
    tsmom_monthly = tsmom.reindex(index=positions.index, method="ffill")

    filtered = apply_trend_filter(positions, tsmom_monthly)
    daily_weights = resample_weights_monthly(filtered, daily_index)

    logger.info(
        "build_directional_weights: %d daily rows, %d commodities",
        len(daily_weights),
        len(daily_weights.columns),
    )
    return daily_weights
