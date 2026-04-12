"""Portfolio construction utilities.

Converts raw signal weights into risk-managed portfolio weights by applying
a chain of constraints:
    1. Volatility targeting — scale to a target annualised portfolio vol.
    2. Position limits — cap individual weights at ±max_weight.
    3. Sector limits — cap aggregate sector weight at max_sector.
    4. Execution lag — shift weights forward by lag_days.
"""

import logging

import numpy as np
import pandas as pd

from commodity_curve_factors.utils.constants import SECTORS, TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)

_DEFAULT_MAX_LEVERAGE = 2.0
_DEFAULT_TARGET_VOL = 0.10
_DEFAULT_MAX_WEIGHT = 0.20
_DEFAULT_MAX_SECTOR = 0.40
_DEFAULT_LAG_DAYS = 1


def apply_vol_target(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_vol: float = _DEFAULT_TARGET_VOL,
    lookback: int = 60,
    max_leverage: float = _DEFAULT_MAX_LEVERAGE,
) -> pd.DataFrame:
    """Scale weights so the portfolio's trailing realized vol matches target.

    At each date t the scalar is::

        scalar(t) = target_vol / realized_vol(t, lookback)

    where realized_vol is the annualised rolling standard deviation of
    portfolio returns over the past ``lookback`` trading days.
    The scalar is capped at ``max_leverage`` to prevent extreme leverage.

    Parameters
    ----------
    weights : pd.DataFrame
        DatetimeIndex × commodity columns, raw portfolio weights.
    returns : pd.DataFrame
        DatetimeIndex × commodity columns, daily returns for the same
        universe.  Must share columns with *weights*.
    target_vol : float
        Annualised target portfolio volatility (e.g. 0.10 = 10%). Default 0.10.
    lookback : int
        Rolling window (trading days) for realized vol estimation. Default 60.
    max_leverage : float
        Maximum allowed scalar. Prevents unbounded leverage when vol is very
        low. Default 2.0.

    Returns
    -------
    pd.DataFrame
        Scaled weights, same shape as *weights*.
    """
    # Align weights and returns on overlapping index
    w, r = weights.align(returns, join="left", axis=0)
    w, r = w.align(r, join="inner", axis=1)

    # Portfolio daily return = sum of weight_i * return_i
    port_ret = (w * r).sum(axis=1)

    # Rolling annualised vol
    rolling_vol = port_ret.rolling(window=lookback, min_periods=lookback).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )

    scalar = (target_vol / rolling_vol).clip(upper=max_leverage)
    # Where rolling_vol is NaN (not enough history), default scalar to 1.0
    scalar = scalar.fillna(1.0)

    scaled = weights.multiply(scalar, axis=0)

    logger.info(
        "apply_vol_target: target_vol=%.2f, lookback=%d, "
        "mean_scalar=%.3f, max_scalar=%.3f",
        target_vol,
        lookback,
        float(scalar.mean()),
        float(scalar.max()),
    )

    return scaled


def apply_position_limits(
    weights: pd.DataFrame,
    max_weight: float = _DEFAULT_MAX_WEIGHT,
) -> pd.DataFrame:
    """Cap individual position weights at ±max_weight, then renormalize.

    Each weight is clipped to the interval [-max_weight, +max_weight].
    Renormalization preserves the long/short structure by scaling each side
    separately so that the gross weights remain proportional.

    Parameters
    ----------
    weights : pd.DataFrame
        DatetimeIndex × commodity columns.
    max_weight : float
        Maximum absolute weight per position. Default 0.20.

    Returns
    -------
    pd.DataFrame
        Clipped and renormalized weights, same shape as *weights*.
    """
    clipped = weights.clip(lower=-max_weight, upper=max_weight)

    logger.info(
        "apply_position_limits: max_weight=%.2f, positions clipped: %d",
        max_weight,
        int((weights.abs() > max_weight).sum().sum()),
    )

    return clipped


def apply_sector_limits(
    weights: pd.DataFrame,
    max_sector: float = _DEFAULT_MAX_SECTOR,
    sectors: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Cap total absolute weight per sector at max_sector, then renormalize.

    For each sector and each date, if the total absolute weight exceeds
    ``max_sector``, all weights in that sector are scaled down proportionally.

    Parameters
    ----------
    weights : pd.DataFrame
        DatetimeIndex × commodity columns.
    max_sector : float
        Maximum total absolute weight per sector. Default 0.40.
    sectors : dict[str, str] | None
        Mapping commodity symbol → sector name. If None, falls back to the
        canonical mapping from ``constants.SECTORS``.

    Returns
    -------
    pd.DataFrame
        Sector-limited weights, same shape as *weights*.
    """
    if sectors is None:
        # Build flat symbol → sector map from constants
        sectors = {sym: sec for sec, syms in SECTORS.items() for sym in syms}

    result = weights.copy()

    # Group commodities by sector; only process columns present in weights
    sector_groups: dict[str, list[str]] = {}
    for sym in weights.columns:
        sec = sectors.get(sym)
        if sec is not None:
            sector_groups.setdefault(sec, []).append(sym)

    for sector, cols in sector_groups.items():
        sector_w = result[cols]
        sector_abs_sum = sector_w.abs().sum(axis=1)
        # Where sum exceeds max_sector, scale down
        needs_scaling = sector_abs_sum > max_sector
        if needs_scaling.any():
            scale = (max_sector / sector_abs_sum).where(needs_scaling, 1.0)
            result[cols] = sector_w.multiply(scale, axis=0)

    logger.info(
        "apply_sector_limits: max_sector=%.2f, sectors processed: %d",
        max_sector,
        len(sector_groups),
    )

    return result


def apply_execution_lag(
    weights: pd.DataFrame,
    lag_days: int = _DEFAULT_LAG_DAYS,
) -> pd.DataFrame:
    """Shift weights forward by lag_days (signal Friday → execute Monday).

    Parameters
    ----------
    weights : pd.DataFrame
        DatetimeIndex × commodity columns, signal-date weights.
    lag_days : int
        Number of rows to shift. Default 1. Positive = forward shift, so
        signal on day t takes effect on day t+lag_days.

    Returns
    -------
    pd.DataFrame
        Lagged weights, same shape as *weights*.  First ``lag_days`` rows
        become NaN (no signal yet).
    """
    if lag_days < 0:
        raise ValueError(f"lag_days must be >= 0, got {lag_days}")

    lagged = weights.shift(lag_days)

    logger.info("apply_execution_lag: lag_days=%d", lag_days)

    return lagged


def build_portfolio(
    raw_weights: pd.DataFrame,
    returns: pd.DataFrame,
    strategy_config: dict,
    universe_config: dict,
) -> pd.DataFrame:
    """Chain all constraints: vol target → position limits → sector limits → execution lag.

    Reads parameters from strategy_config["constraints"] and
    strategy_config["execution"].

    Parameters
    ----------
    raw_weights : pd.DataFrame
        DatetimeIndex × commodity columns, raw signal weights before any
        risk management.
    returns : pd.DataFrame
        DatetimeIndex × commodity columns, daily returns for vol targeting.
    strategy_config : dict
        Strategy YAML config.  Expected keys:
        ``constraints`` (vol_target, max_position_weight, max_sector_weight,
        max_leverage) and ``execution`` (lag_days).
    universe_config : dict
        Universe YAML config (``commodities`` key with ``sector`` per entry).

    Returns
    -------
    pd.DataFrame
        Fully constrained and lagged portfolio weights.
    """
    constraints = strategy_config.get("constraints", {})
    execution = strategy_config.get("execution", {})

    target_vol = float(constraints.get("vol_target", _DEFAULT_TARGET_VOL))
    max_weight = float(constraints.get("max_position_weight", _DEFAULT_MAX_WEIGHT))
    max_sector = float(constraints.get("max_sector_weight", _DEFAULT_MAX_SECTOR))
    max_leverage = float(constraints.get("max_leverage", _DEFAULT_MAX_LEVERAGE))
    lag_days = int(execution.get("lag_days", _DEFAULT_LAG_DAYS))

    # Build commodity → sector map from universe config
    sectors: dict[str, str] = {}
    for sym, spec in universe_config.get("commodities", {}).items():
        if isinstance(spec, dict) and "sector" in spec:
            sectors[sym] = spec["sector"]

    w = apply_vol_target(raw_weights, returns, target_vol=target_vol, max_leverage=max_leverage)
    w = apply_position_limits(w, max_weight=max_weight)
    w = apply_sector_limits(w, max_sector=max_sector, sectors=sectors)
    w = apply_execution_lag(w, lag_days=lag_days)

    logger.info("build_portfolio: pipeline complete, shape=%s", w.shape)

    return w
