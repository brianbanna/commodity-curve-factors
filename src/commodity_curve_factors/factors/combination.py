"""Factor combination methods.

Combines individual factor signals into composite signals using three
strategies: equal-weight (nanmean), IC-weighted (trailing cross-sectional
rank correlation), and regime-conditioned (VIX-based weight switching).

All methods are NaN-tolerant: a (date, commodity) cell is NaN only when ALL
contributing factors are NaN for that cell.

Usage:
    python -m commodity_curve_factors.factors.combination
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def equal_weight_composite(
    factors: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Equal-weight composite: nanmean across available factors per commodity per day.

    Parameters
    ----------
    factors : dict[str, pd.DataFrame]
        Keyed by factor name (e.g. "carry", "slope"), each DataFrame has
        DatetimeIndex and commodity columns. Factors may have different
        date ranges and different NaN patterns (e.g. inventory is NaN for
        non-energy commodities).

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (union of all factor dates), columns = union of all
        commodity columns. Values are the nanmean of all available factor
        z-scores for that (date, commodity). NaN only if ALL factors are
        NaN for that cell.
    """
    if not factors:
        logger.warning("equal_weight_composite: no factors provided — returning empty DataFrame")
        return pd.DataFrame()

    dfs = list(factors.values())

    # Build a union date index and union column set
    all_dates = dfs[0].index
    all_cols: set[str] = set(dfs[0].columns)
    for df in dfs[1:]:
        all_dates = all_dates.union(df.index)
        all_cols.update(df.columns)

    sorted_cols = sorted(all_cols)

    # Stack all factors into a 3D array: (dates, commodities, factors)
    # Reindex each factor to the union index/columns first
    aligned = []
    for df in dfs:
        reindexed = df.reindex(index=all_dates, columns=sorted_cols)
        aligned.append(reindexed.to_numpy(dtype=float))

    stacked = np.stack(aligned, axis=2)  # shape: (n_dates, n_cols, n_factors)

    # nanmean across the factor dimension
    with np.errstate(all="ignore"):
        composite_arr = np.nanmean(stacked, axis=2)

    # Where ALL factors are NaN, nanmean returns NaN naturally (no warning
    # needed — we suppress warnings above). However, np.nanmean of all-NaN
    # raises RuntimeWarning and returns NaN, which is the desired behavior.
    # Re-mask: if every factor is NaN for a cell, force it back to NaN.
    all_nan_mask = np.all(np.isnan(stacked), axis=2)
    composite_arr[all_nan_mask] = np.nan

    result = pd.DataFrame(composite_arr, index=all_dates, columns=sorted_cols)
    result.index.name = "Date"

    logger.info(
        "equal_weight_composite: %d factors, %d dates, %d commodities, %.1f%% non-NaN",
        len(factors),
        len(result),
        len(result.columns),
        result.notna().mean().mean() * 100,
    )
    return result


def ic_weighted_composite(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    lookback: int = 252,
    min_observations: int = 60,
    rebalance_freq: str = "ME",
) -> pd.DataFrame:
    """IC-weighted composite: weight each factor by its trailing information coefficient.

    The IC for a factor is the cross-sectional rank correlation between the
    factor score and the forward 1-day return, averaged over a trailing window.

    Parameters
    ----------
    factors : dict[str, pd.DataFrame]
        Same as ``equal_weight_composite``.
    forward_returns : pd.DataFrame
        1-day forward returns (same index/columns as the factor DataFrames).
        IMPORTANT: these must be SHIFTED forward by 1 day to avoid lookahead.
        The caller is responsible for the shift.
    lookback : int
        Trailing window for IC computation. Default 252.
    min_observations : int
        Minimum non-NaN ICs required before weighting. If fewer, fall back
        to equal weight for that month. Default 60.
    rebalance_freq : str
        How often to recompute weights. Default "ME" (month-end).

    Returns
    -------
    pd.DataFrame
        Composite factor signal.
    """
    if not factors:
        logger.warning("ic_weighted_composite: no factors provided — returning empty DataFrame")
        return pd.DataFrame()

    factor_names = list(factors.keys())
    dfs = list(factors.values())

    # Build union index and columns (same approach as equal_weight)
    all_dates = dfs[0].index
    all_cols: set[str] = set(dfs[0].columns)
    for df in dfs[1:]:
        all_dates = all_dates.union(df.index)
        all_cols.update(df.columns)

    sorted_cols = sorted(all_cols)
    all_dates = all_dates.sort_values()

    # Align all factors and forward returns to (all_dates, sorted_cols)
    aligned: dict[str, pd.DataFrame] = {}
    for name, df in factors.items():
        aligned[name] = df.reindex(index=all_dates, columns=sorted_cols)

    fwd = forward_returns.reindex(index=all_dates, columns=sorted_cols)

    # Compute daily cross-sectional IC for each factor
    # IC(t) = spearmanr(factor[t], fwd_returns[t]) across commodities
    daily_ic: dict[str, pd.Series] = {}
    for name, df in aligned.items():
        ic_vals = np.full(len(all_dates), np.nan)
        factor_arr = df.to_numpy(dtype=float)
        fwd_arr = fwd.to_numpy(dtype=float)

        for i in range(len(all_dates)):
            f_row = factor_arr[i]
            r_row = fwd_arr[i]
            # Keep only cells where both factor and return are non-NaN
            valid = ~(np.isnan(f_row) | np.isnan(r_row))
            n_valid = int(valid.sum())
            if n_valid < 2:
                continue
            with np.errstate(all="ignore"):
                corr, _ = spearmanr(f_row[valid], r_row[valid])
            if not np.isnan(corr):
                ic_vals[i] = corr

        daily_ic[name] = pd.Series(ic_vals, index=all_dates)

    ic_df = pd.DataFrame(daily_ic)  # shape: (n_dates, n_factors)

    # Identify rebalance dates using the specified frequency
    rebalance_dates = pd.date_range(
        start=all_dates[0],
        end=all_dates[-1],
        freq=rebalance_freq,
    )

    # Build weight array per day: same weight held between rebalance dates
    n_dates = len(all_dates)
    n_factors = len(factor_names)
    weights = np.full((n_dates, n_factors), 1.0 / n_factors)  # default: equal weight

    current_weights = np.ones(n_factors) / n_factors
    rebalance_set = set(rebalance_dates.normalize())

    for i, dt in enumerate(all_dates):
        if dt.normalize() in rebalance_set:
            # Compute trailing IC mean over lookback window
            start_i = max(0, i - lookback)
            window_ic = ic_df.iloc[start_i:i].to_numpy(dtype=float)  # (lookback, n_factors)

            # Count valid (non-NaN) IC observations per factor
            valid_counts = np.sum(~np.isnan(window_ic), axis=0)
            min_count = int(np.min(valid_counts))

            if min_count < min_observations:
                # Insufficient data → equal weight
                current_weights = np.ones(n_factors) / n_factors
            else:
                mean_ic = np.nanmean(window_ic, axis=0)
                # Clamp negative IC to 0
                mean_ic_pos = np.maximum(mean_ic, 0.0)
                total = float(np.sum(mean_ic_pos))
                if total == 0.0:
                    # All ICs non-positive → equal weight
                    current_weights = np.ones(n_factors) / n_factors
                else:
                    current_weights = mean_ic_pos / total

        weights[i] = current_weights

    # Compute weighted composite
    # For each date i: composite[i, j] = sum_k(weights[i,k] * factor[k][i,j])
    # We need to renormalize weights per cell to handle NaN factors
    factor_arr_3d = np.stack(
        [aligned[name].to_numpy(dtype=float) for name in factor_names],
        axis=2,
    )  # shape: (n_dates, n_cols, n_factors)

    # Broadcast weights: shape (n_dates, 1, n_factors)
    weights_3d = weights[:, np.newaxis, :]  # (n_dates, 1, n_factors)
    weights_broadcast = np.broadcast_to(weights_3d, factor_arr_3d.shape).copy()

    # Zero out weights where factor is NaN, then renormalize
    nan_mask = np.isnan(factor_arr_3d)
    weights_broadcast[nan_mask] = 0.0
    weight_sum = weights_broadcast.sum(axis=2, keepdims=True)

    # Avoid division by zero (all factors NaN for a cell)
    safe_weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
    weights_norm = weights_broadcast / safe_weight_sum

    # Replace NaN factor values with 0 for the dot product (already zero-weighted)
    factor_filled = np.where(nan_mask, 0.0, factor_arr_3d)
    composite_arr = np.sum(weights_norm * factor_filled, axis=2)

    # Re-apply NaN where ALL factors were NaN
    all_nan_mask = np.all(np.isnan(factor_arr_3d), axis=2)
    composite_arr[all_nan_mask] = np.nan

    result = pd.DataFrame(composite_arr, index=all_dates, columns=sorted_cols)
    result.index.name = "Date"

    logger.info(
        "ic_weighted_composite: %d factors, %d dates, %d commodities, %.1f%% non-NaN",
        len(factors),
        len(result),
        len(result.columns),
        result.notna().mean().mean() * 100,
    )
    return result


def regime_conditioned_composite(
    factors: dict[str, pd.DataFrame],
    vix: pd.Series,
    vix_thresholds: list[float] | None = None,
    weights_by_regime: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Regime-conditioned composite: VIX regime determines factor weights.

    VIX regimes (from configs/strategy.yaml):
    - calm: VIX < 15
    - moderate: 15 <= VIX < 25
    - turbulent: VIX >= 25

    Each regime has a pre-specified weight vector for the factors.

    Parameters
    ----------
    factors : dict[str, pd.DataFrame]
        Factor DataFrames. Keys must match the keys in ``weights_by_regime``.
    vix : pd.Series
        Daily VIX levels (Close column from the macro data).
    vix_thresholds : list[float], optional
        [low, high] thresholds. Default [15, 25].
    weights_by_regime : dict, optional
        Maps regime name -> factor name -> weight. Default from strategy.yaml.

    Returns
    -------
    pd.DataFrame
        Composite factor signal.
    """
    if vix_thresholds is None:
        vix_thresholds = [15.0, 25.0]

    if weights_by_regime is None:
        # Default from strategy.yaml
        weights_by_regime = {
            "calm": {
                "carry": 0.40,
                "slope": 0.25,
                "momentum": 0.15,
                "curvature": 0.10,
                "inventory": 0.10,
            },
            "moderate": {
                "carry": 0.25,
                "slope": 0.20,
                "momentum": 0.25,
                "curvature": 0.15,
                "inventory": 0.15,
            },
            "turbulent": {
                "carry": 0.10,
                "slope": 0.15,
                "momentum": 0.35,
                "curvature": 0.20,
                "inventory": 0.20,
            },
        }

    if not factors:
        logger.warning(
            "regime_conditioned_composite: no factors provided — returning empty DataFrame"
        )
        return pd.DataFrame()

    low_thresh, high_thresh = float(vix_thresholds[0]), float(vix_thresholds[1])

    dfs = list(factors.values())
    all_dates = dfs[0].index
    all_cols: set[str] = set(dfs[0].columns)
    for df in dfs[1:]:
        all_dates = all_dates.union(df.index)
        all_cols.update(df.columns)

    sorted_cols = sorted(all_cols)
    all_dates = all_dates.sort_values()

    # Align all factors to common index/columns
    aligned: dict[str, pd.DataFrame] = {}
    for name, df in factors.items():
        aligned[name] = df.reindex(index=all_dates, columns=sorted_cols)

    # Align VIX to the same date index
    vix_aligned = vix.reindex(all_dates)

    n_dates = len(all_dates)
    n_cols = len(sorted_cols)
    composite_arr = np.full((n_dates, n_cols), np.nan)

    factor_arrays: dict[str, np.ndarray] = {
        name: aligned[name].to_numpy(dtype=float) for name in aligned
    }

    for i in range(n_dates):
        vix_val = vix_aligned.iloc[i]

        # Classify VIX regime
        if np.isnan(vix_val):
            # Fall back to equal weight if VIX is missing
            regime = None
        elif vix_val < low_thresh:
            regime = "calm"
        elif vix_val < high_thresh:
            regime = "moderate"
        else:
            regime = "turbulent"

        if regime is None or regime not in weights_by_regime:
            # Equal weight fallback
            regime_weights: dict[str, float] = {name: 1.0 / len(factors) for name in factors}
        else:
            regime_weights = weights_by_regime[regime]

        # Compute weighted sum for each commodity column
        row_weights = np.zeros(n_cols)
        row_value = np.zeros(n_cols)
        any_valid = np.zeros(n_cols, dtype=bool)

        for fname, weight in regime_weights.items():
            if fname not in factor_arrays:
                continue
            factor_row = factor_arrays[fname][i]
            valid = ~np.isnan(factor_row)
            row_weights[valid] += weight
            row_value[valid] += weight * factor_row[valid]
            any_valid |= valid

        # Normalize by actual weight sum (handles NaN factors in this regime)
        safe_weights = np.where(row_weights == 0.0, 1.0, row_weights)
        row_result = row_value / safe_weights
        row_result[~any_valid] = np.nan
        composite_arr[i] = row_result

    result = pd.DataFrame(composite_arr, index=all_dates, columns=sorted_cols)
    result.index.name = "Date"

    logger.info(
        "regime_conditioned_composite: %d factors, %d dates, %d commodities, %.1f%% non-NaN",
        len(factors),
        len(result),
        len(result.columns),
        result.notna().mean().mean() * 100,
    )
    return result


def _default_weights_by_regime() -> dict[str, dict[str, Any]]:
    """Load regime weights from strategy.yaml."""
    from commodity_curve_factors.utils.config import load_config

    strategy = load_config("strategy")
    return strategy.get("regime_conditioned", {}).get("weights_by_regime", {})


def main() -> None:
    """Smoke-test combination functions with small synthetic DataFrames."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    cols = ["CL", "GC", "NG"]

    carry = pd.DataFrame(rng.standard_normal((300, 3)), index=dates, columns=cols)
    slope = pd.DataFrame(rng.standard_normal((300, 3)), index=dates, columns=cols)
    factors = {"carry": carry, "slope": slope}

    ew = equal_weight_composite(factors)
    logger.info("equal_weight_composite shape: %s", ew.shape)

    fwd = pd.DataFrame(rng.standard_normal((300, 3)), index=dates, columns=cols)
    ic = ic_weighted_composite(factors, fwd)
    logger.info("ic_weighted_composite shape: %s", ic.shape)

    vix = pd.Series(rng.uniform(10, 35, 300), index=dates)
    regime = regime_conditioned_composite(
        factors,
        vix,
        vix_thresholds=[15.0, 25.0],
        weights_by_regime={
            "calm": {"carry": 0.6, "slope": 0.4},
            "moderate": {"carry": 0.5, "slope": 0.5},
            "turbulent": {"carry": 0.3, "slope": 0.7},
        },
    )
    logger.info("regime_conditioned_composite shape: %s", regime.shape)


if __name__ == "__main__":
    main()
