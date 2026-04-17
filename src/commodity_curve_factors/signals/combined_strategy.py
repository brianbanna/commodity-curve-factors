"""Combined multi-layer strategy with Ledoit-Wolf covariance vol targeting.

Provides two public functions:

- ``apply_ledoit_wolf_vol_target`` — scale a weight matrix to a target
  annualised portfolio volatility using Ledoit-Wolf shrinkage covariance.
- ``combine_layers`` — combine multiple strategy layers with risk budgets,
  vol-targeting each layer individually before summing.
"""

import logging
import math

import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


def apply_ledoit_wolf_vol_target(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_vol: float = 0.10,
    lookback: int = 252,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """Scale weights to a target annualised portfolio volatility via Ledoit-Wolf.

    For each row starting at index ``lookback``:

    1. Collect the trailing ``lookback``-day return window.
    2. Drop columns with fewer than ``lookback // 2`` non-NaN observations.
    3. Skip (leave weights unchanged) if fewer than 2 valid columns remain or
       if the absolute weight sum is near zero.
    4. Fit ``sklearn.covariance.LedoitWolf`` on the return window.
    5. Compute portfolio variance :math:`w^T \\Sigma w` and derive the daily
       target vol scalar.
    6. Cap the scalar so gross leverage does not exceed ``max_leverage``.

    Rows prior to the warm-up period are returned as-is (unscaled).

    Parameters
    ----------
    weights : pd.DataFrame
        DatetimeIndex × commodity columns of raw portfolio weights.
    returns : pd.DataFrame
        DatetimeIndex × commodity columns of daily returns.  Must cover the
        same date range as *weights*.
    target_vol : float
        Annualised target portfolio volatility (e.g. 0.10 = 10 %).
    lookback : int
        Trailing window length in trading days for the covariance estimate.
    max_leverage : float
        Hard cap on gross leverage (sum of absolute weights).

    Returns
    -------
    pd.DataFrame
        Scaled weights, same shape as *weights*.
    """
    daily_target = target_vol / math.sqrt(_TRADING_DAYS_PER_YEAR)

    # Work on a copy so we never mutate the caller's DataFrame
    scaled = weights.copy()

    # Align returns to the weight index/columns where possible
    ret_aligned = returns.reindex(index=weights.index, columns=weights.columns)

    n = len(weights)

    for i in range(lookback, n):
        w_row = weights.iloc[i]

        # Gross leverage guard — skip if nothing is traded
        abs_sum = w_row.abs().sum()
        if abs_sum < 1e-12:
            continue

        # Trailing return window
        ret_window = ret_aligned.iloc[i - lookback : i]

        # Drop columns with insufficient history
        min_obs = lookback // 2
        valid_cols = ret_window.columns[ret_window.notna().sum() >= min_obs]
        if len(valid_cols) < 2:
            continue

        ret_window = ret_window[valid_cols].copy()
        w_valid = w_row[valid_cols].values.astype(float)

        # Fill remaining NaNs with zero for covariance estimation
        ret_window = ret_window.fillna(0.0)

        try:
            lw = LedoitWolf()
            lw.fit(ret_window.values)
            cov = lw.covariance_
        except Exception:
            logger.debug("LedoitWolf fit failed at row %d, skipping", i)
            continue

        port_var = float(w_valid @ cov @ w_valid)
        if port_var <= 0.0:
            continue

        port_vol = math.sqrt(port_var)  # daily portfolio vol
        raw_scalar = daily_target / port_vol

        # Cap so gross leverage <= max_leverage
        max_scalar = max_leverage / abs_sum
        scalar = min(raw_scalar, max_scalar)

        scaled.iloc[i] = weights.iloc[i] * scalar

    logger.info(
        "apply_ledoit_wolf_vol_target: target_vol=%.2f lookback=%d max_leverage=%.2f",
        target_vol,
        lookback,
        max_leverage,
    )

    return scaled


def combine_layers(
    layer_weights: list[pd.DataFrame],
    risk_budgets: list[float],
    returns: pd.DataFrame,
    target_vol: float = 0.10,
    lookback: int = 252,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """Combine multiple strategy layers with risk budgeting.

    Each layer is vol-targeted independently to ``target_vol * budget`` with a
    leverage cap of ``max_leverage * budget``, then all layers are summed.

    Parameters
    ----------
    layer_weights : list[pd.DataFrame]
        One DataFrame of weights per layer.
    risk_budgets : list[float]
        Non-negative budget for each layer (need not sum to 1 — they are
        normalised internally).
    returns : pd.DataFrame
        DatetimeIndex × commodity columns of daily returns used for covariance
        estimation.
    target_vol : float
        Aggregate annualised target portfolio volatility.
    lookback : int
        Lookback window passed through to ``apply_ledoit_wolf_vol_target``.
    max_leverage : float
        Aggregate leverage cap passed through (scaled by budget per layer).

    Returns
    -------
    pd.DataFrame
        Combined weights on the union of all layer indices and columns.
    """
    if len(layer_weights) != len(risk_budgets):
        raise ValueError("layer_weights and risk_budgets must have the same length")

    # Normalise budgets
    total_budget = sum(risk_budgets)
    if total_budget <= 0.0:
        raise ValueError("risk_budgets must sum to a positive value")
    norm_budgets = [b / total_budget for b in risk_budgets]

    # Build union index and columns
    all_indices = layer_weights[0].index
    all_cols = set(layer_weights[0].columns)
    for lw in layer_weights[1:]:
        all_indices = all_indices.union(lw.index)
        all_cols = all_cols.union(lw.columns)
    all_cols_sorted = sorted(all_cols)

    combined = pd.DataFrame(0.0, index=all_indices, columns=all_cols_sorted)

    for lw, budget in zip(layer_weights, norm_budgets):
        if budget <= 0.0:
            continue

        layer_target_vol = target_vol * budget
        layer_max_leverage = max_leverage * budget

        # Reindex to the union shape before vol-targeting
        lw_reindexed = lw.reindex(index=all_indices, columns=all_cols_sorted).fillna(0.0)

        layer_scaled = apply_ledoit_wolf_vol_target(
            weights=lw_reindexed,
            returns=returns,
            target_vol=layer_target_vol,
            lookback=lookback,
            max_leverage=layer_max_leverage,
        )

        combined = combined.add(layer_scaled, fill_value=0.0)

    logger.info(
        "combine_layers: %d layers, budgets=%s, target_vol=%.2f",
        len(layer_weights),
        [f"{b:.3f}" for b in norm_budgets],
        target_vol,
    )

    return combined
