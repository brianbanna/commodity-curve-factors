"""Layer 1: Curve-informed directional positioning.

Long-biased regime tilt: scales a 1/N long-only base allocation by
convenience yield regime multipliers and a multiplicative TSMOM trend tilt.

The strategy is almost always long (capturing the commodity risk premium),
varying SIZE based on fundamental curve signals and trend confirmation.
This avoids the problem of being out of the market too much — the previous
binary trend filter zeroed 60%+ of positions, giving up the risk premium.
"""

import logging

import pandas as pd

from commodity_curve_factors.signals.curve_regime import classify_regime, regime_to_position

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_trend_tilt(
    positions: pd.DataFrame,
    tsmom: pd.DataFrame,
    trend_up_mult: float = 1.2,
    trend_down_mult: float = 0.7,
) -> pd.DataFrame:
    """Apply a multiplicative TSMOM trend tilt to regime positions.

    Instead of zeroing positions that conflict with the trend (which removes
    most of the commodity risk premium), this scales positions up when trend
    confirms and down when it disagrees.

    Parameters
    ----------
    positions : pd.DataFrame
        Regime-based position weights (dates x commodities).
    tsmom : pd.DataFrame
        TSMOM signal (dates x commodities). Sign is what matters.
    trend_up_mult : float
        Multiplier when TSMOM > 0. Default 1.2.
    trend_down_mult : float
        Multiplier when TSMOM <= 0. Default 0.7.

    Returns
    -------
    pd.DataFrame
        Tilted positions.
    """
    result = positions.copy()
    tsmom_aligned = tsmom.reindex(index=positions.index, columns=positions.columns)

    shared_cols = positions.columns.intersection(tsmom.columns)
    trend_pos = tsmom_aligned[shared_cols] > 0

    # Multiplicative tilt: up when trend confirms, down when it disagrees
    multiplier = pd.DataFrame(trend_down_mult, index=positions.index, columns=shared_cols)
    multiplier[trend_pos] = trend_up_mult

    result[shared_cols] = result[shared_cols] * multiplier

    logger.info(
        "apply_trend_tilt: %d rows, %d cols, up=%.2f, down=%.2f",
        len(result),
        len(result.columns),
        trend_up_mult,
        trend_down_mult,
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
    trend_up_mult: float = 1.2,
    trend_down_mult: float = 0.7,
) -> pd.DataFrame:
    """Build daily long-biased directional weights from convenience yield and TSMOM.

    Pipeline:
    1. ``classify_regime`` — classify each commodity-month by CY percentile.
    2. ``regime_to_position`` — map regime labels to regime multipliers.
    3. Apply 1/N base weight scaled by regime multiplier.
    4. ``apply_trend_tilt`` — multiplicative TSMOM tilt (scale, not zero-out).
    5. ``resample_weights_monthly`` — forward-fill to the target daily index.

    Parameters
    ----------
    monthly_cy : pd.DataFrame
        Monthly convenience yield (month-end dates x commodities).
    tsmom : pd.DataFrame
        TSMOM signal (dates x commodities). Resampled to monthly internally.
    daily_index : pd.DatetimeIndex
        Target daily business-day index for the output.
    thresholds : list[int] or None
        Percentile thresholds forwarded to ``classify_regime``.
    position_map : dict[str, float] or None
        Regime-to-multiplier mapping. Default: crisis_backwardation=1.5,
        mild_backwardation=1.25, balanced=1.0, mild_contango=0.5,
        deep_contango=0.0.
    trend_up_mult : float
        TSMOM > 0 multiplier. Default 1.2.
    trend_down_mult : float
        TSMOM <= 0 multiplier. Default 0.7.

    Returns
    -------
    pd.DataFrame
        Daily position weights (``len(daily_index)`` rows x commodities).
    """
    if position_map is None:
        position_map = {
            "crisis_backwardation": 1.5,
            "mild_backwardation": 1.25,
            "balanced": 1.0,
            "mild_contango": 0.5,
            "deep_contango": 0.0,
        }

    regimes = classify_regime(monthly_cy, thresholds=thresholds)
    regime_multipliers = regime_to_position(regimes, position_map=position_map)

    # Apply 1/N base weight scaled by regime multiplier
    n_commodities = len(monthly_cy.columns)
    base_weight = 1.0 / n_commodities
    positions = regime_multipliers * base_weight

    # Multiplicative TSMOM tilt (scale, not zero-out)
    tsmom_monthly = tsmom.reindex(index=positions.index, method="ffill")
    tilted = apply_trend_tilt(
        positions,
        tsmom_monthly,
        trend_up_mult=trend_up_mult,
        trend_down_mult=trend_down_mult,
    )

    daily_weights = resample_weights_monthly(tilted, daily_index)

    logger.info(
        "build_directional_weights: %d daily rows, %d commodities, mean_weight=%.4f",
        len(daily_weights),
        len(daily_weights.columns),
        daily_weights.mean().mean(),
    )
    return daily_weights
