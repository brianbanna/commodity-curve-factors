"""Cost sensitivity analysis for backtested strategies.

Reruns a strategy at varying transaction cost levels to quantify how much
of the edge survives rising costs.

Usage:
    from commodity_curve_factors.backtest.sensitivity import run_cost_sensitivity
"""

import logging

import numpy as np
import pandas as pd

from commodity_curve_factors.backtest.engine import run_backtest

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


def _sharpe(net_return: pd.Series) -> float:
    """Annualised Sharpe ratio (assumes zero risk-free rate)."""
    std = float(net_return.std())
    if std == 0.0 or len(net_return) == 0:
        return 0.0
    return float(net_return.mean()) / std * np.sqrt(_TRADING_DAYS_PER_YEAR)


def _cagr(cumulative: pd.Series) -> float:
    """Compound annual growth rate from a cumulative wealth series starting at 1.0."""
    n = len(cumulative)
    if n == 0:
        return 0.0
    years = n / _TRADING_DAYS_PER_YEAR
    end_val = float(cumulative.iloc[-1])
    if end_val <= 0.0 or years <= 0.0:
        return 0.0
    return end_val ** (1.0 / years) - 1.0


def _max_drawdown(drawdown: pd.Series) -> float:
    """Maximum drawdown (most negative value in the drawdown series)."""
    if drawdown.empty:
        return 0.0
    return float(drawdown.min())


def run_cost_sensitivity(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps_list: list[int],
) -> pd.DataFrame:
    """Run the same strategy at different cost levels.

    At each cost level, all per-commodity overrides are overridden with the
    same flat cost for commission + slippage; roll costs are kept at 0
    (the flat ``cost_bps`` is already an all-in proxy).

    Parameters
    ----------
    weights : pd.DataFrame
        Daily portfolio weights (DatetimeIndex × commodity columns).
        Must already be constrained and lagged (output of
        ``portfolio.build_portfolio``).
    returns : pd.DataFrame
        Daily log returns (same structure as *weights*).
    cost_bps_list : list[int]
        Cost levels to sweep, expressed as total transaction cost in basis
        points (commission + slippage, per unit of turnover).

    Returns
    -------
    pd.DataFrame
        One row per cost level with columns:
        ``cost_bps``, ``sharpe``, ``cagr``, ``max_drawdown``, ``cumulative``.
        ``cumulative`` is the final wealth index value (starts at 1.0).
    """
    if not cost_bps_list:
        logger.warning("run_cost_sensitivity: empty cost_bps_list — returning empty DataFrame")
        return pd.DataFrame(columns=["cost_bps", "sharpe", "cagr", "max_drawdown", "cumulative"])

    rows = []

    for bps in cost_bps_list:
        # Build a uniform cost config: commission absorbs all the bps, slippage and roll = 0
        cost_config: dict = {
            "default": {
                "commission_bps": float(bps),
                "slippage_bps": 0.0,
                "roll_cost_bps": 0.0,
            },
            "per_commodity": {},
        }

        result = run_backtest(weights, returns, cost_config)

        if result.empty:
            logger.warning("run_cost_sensitivity: empty backtest at cost_bps=%d", bps)
            rows.append(
                {
                    "cost_bps": bps,
                    "sharpe": np.nan,
                    "cagr": np.nan,
                    "max_drawdown": np.nan,
                    "cumulative": np.nan,
                }
            )
            continue

        sh = _sharpe(result["net_return"])
        cg = _cagr(result["cumulative"])
        md = _max_drawdown(result["drawdown"])
        cum = float(result["cumulative"].iloc[-1])

        rows.append(
            {
                "cost_bps": bps,
                "sharpe": sh,
                "cagr": cg,
                "max_drawdown": md,
                "cumulative": cum,
            }
        )

        logger.info(
            "cost_sensitivity: cost_bps=%d → sharpe=%.3f, cagr=%.3f, max_dd=%.3f",
            bps,
            sh,
            cg,
            md,
        )

    df = pd.DataFrame(rows)
    logger.info("run_cost_sensitivity: swept %d cost levels", len(rows))
    return df
