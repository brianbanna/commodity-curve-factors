"""Vectorized backtest engine for futures factor strategies.

Runs daily portfolio simulations with realistic transaction costs (commission,
slippage, roll costs) and produces a full performance time series.

Usage:
    python -m commodity_curve_factors.backtest.engine
"""

import logging

import numpy as np
import pandas as pd

from commodity_curve_factors.backtest.costs import apply_costs

logger = logging.getLogger(__name__)


def compute_portfolio_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Series:
    """Vectorized portfolio return: sum(w_i * r_i) per day.

    Aligns weights and returns on their shared index and columns.
    Missing columns in either are dropped (inner join).

    Parameters
    ----------
    weights : pd.DataFrame
        Daily portfolio weights (DatetimeIndex × commodity columns).
    returns : pd.DataFrame
        Daily log returns (same structure as *weights*).

    Returns
    -------
    pd.Series
        Daily gross portfolio returns.
    """
    w, r = weights.align(returns, join="inner", axis=0)
    w, r = w.align(r, join="inner", axis=1)

    gross = (w * r).sum(axis=1)

    logger.debug(
        "compute_portfolio_returns: %d days, mean=%.5f, std=%.5f",
        len(gross),
        float(gross.mean()),
        float(gross.std()),
    )
    return gross


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """Daily turnover: sum of absolute weight changes.

    Defined as ``sum_i |w_i(t) - w_i(t-1)|`` for each day *t*.

    Parameters
    ----------
    weights : pd.DataFrame
        Daily portfolio weights (DatetimeIndex × commodity columns).

    Returns
    -------
    pd.Series
        Daily turnover.  First row is NaN (no prior weights).
    """
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = np.nan
    return turnover


def run_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_config: dict,
    roll_schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run a vectorized backtest with transaction costs.

    Parameters
    ----------
    weights : pd.DataFrame
        Daily portfolio weights (DatetimeIndex × commodity columns).
        Already constrained by ``portfolio.build_portfolio``.
    returns : pd.DataFrame
        Daily log returns (same structure as *weights*).
    cost_config : dict
        From ``configs/backtest.yaml``.  Contains ``"default"`` and
        ``"per_commodity"`` cost dicts with keys:
        ``commission_bps``, ``slippage_bps``, ``roll_cost_bps``.
    roll_schedule : pd.DataFrame, optional
        Output of ``build_roll_schedule``.  If provided, roll costs are applied
        on days when the front contract rolls for a held position.

    Returns
    -------
    pd.DataFrame
        Columns: ``[gross_return, cost, net_return, cumulative, drawdown,
        turnover]``.

        - ``gross_return`` = ``sum(w * r)`` per day
        - ``cost``         = commission + slippage + roll_cost
        - ``net_return``   = gross − cost
        - ``cumulative``   = ``exp(cumsum(net_return))``, starting at 1.0
        - ``drawdown``     = ``cumulative / cumulative.cummax() − 1``
        - ``turnover``     = ``sum(|Δw|)``
    """
    # Align index and columns (inner join)
    w, r = weights.align(returns, join="inner", axis=0)
    w, r = w.align(r, join="inner", axis=1)

    if w.empty:
        logger.warning("run_backtest: no overlapping dates/columns — returning empty DataFrame")
        empty = pd.DataFrame(
            columns=["gross_return", "cost", "net_return", "cumulative", "drawdown", "turnover"]
        )
        return empty

    # Gross return
    gross_return = (w * r).sum(axis=1)

    # Transaction + roll costs
    cost, net_return = apply_costs(gross_return, w, cost_config, roll_schedule)

    # Cumulative wealth (starting at 1.0)
    cumulative = np.exp(net_return.cumsum())

    # Drawdown from peak
    drawdown = cumulative / cumulative.cummax() - 1

    # Turnover
    turnover = compute_turnover(w)

    result = pd.DataFrame(
        {
            "gross_return": gross_return,
            "cost": cost,
            "net_return": net_return,
            "cumulative": cumulative,
            "drawdown": drawdown,
            "turnover": turnover,
        }
    )

    logger.info(
        "run_backtest: %d days, annualised_gross=%.3f, annualised_net=%.3f, "
        "max_dd=%.3f, mean_turnover=%.4f",
        len(result),
        float(gross_return.mean() * 252),
        float(net_return.mean() * 252),
        float(drawdown.min()),
        float(turnover.mean(skipna=True)),
    )
    return result


def main() -> None:
    """Smoke-test the engine with synthetic data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    cols = ["CL", "GC", "NG"]

    weights = pd.DataFrame(rng.uniform(-0.1, 0.1, (252, 3)), index=dates, columns=cols)
    returns = pd.DataFrame(rng.normal(0, 0.01, (252, 3)), index=dates, columns=cols)

    cost_config = {
        "default": {"commission_bps": 3, "slippage_bps": 2, "roll_cost_bps": 2},
        "per_commodity": {
            "CL": {"commission_bps": 2, "slippage_bps": 1, "roll_cost_bps": 1},
        },
    }

    result = run_backtest(weights, returns, cost_config)
    logger.info("Backtest result shape: %s", result.shape)
    logger.info("Final cumulative: %.4f", float(result["cumulative"].iloc[-1]))
    logger.info("Max drawdown: %.4f", float(result["drawdown"].min()))


if __name__ == "__main__":
    main()
