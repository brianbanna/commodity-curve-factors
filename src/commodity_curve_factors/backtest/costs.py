"""Transaction cost model for futures backtests.

Provides per-commodity cost look-up (with default fallback) and helpers to
compute daily transaction costs (from turnover) and roll costs (from the roll
schedule).

Usage:
    python -m commodity_curve_factors.backtest.costs
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_cost(
    commodity: str,
    cost_type: str,
    config: dict,
) -> float:
    """Look up a per-commodity cost, falling back to the default.

    Parameters
    ----------
    commodity : str
        Commodity symbol (e.g. ``"CL"``).
    cost_type : str
        One of ``"commission_bps"``, ``"slippage_bps"``, ``"roll_cost_bps"``.
    config : dict
        The ``costs`` section from ``configs/backtest.yaml``.  Expected
        keys: ``"default"`` (dict) and optionally ``"per_commodity"`` (dict of
        dicts keyed by commodity symbol).

    Returns
    -------
    float
        Cost value in basis points.
    """
    per_commodity = config.get("per_commodity", {})
    commodity_costs = per_commodity.get(commodity, {})

    if cost_type in commodity_costs:
        value = float(commodity_costs[cost_type])
        logger.debug("get_cost: %s %s = %.1f bps (per-commodity)", commodity, cost_type, value)
        return value

    default = config.get("default", {})
    value = float(default.get(cost_type, 0.0))
    logger.debug("get_cost: %s %s = %.1f bps (default)", commodity, cost_type, value)
    return value


def compute_transaction_costs(
    weights: pd.DataFrame,
    config: dict,
) -> pd.Series:
    """Daily transaction cost from portfolio weight turnover.

    For each commodity *i* and each day *t*::

        cost_i(t) = |w_i(t) - w_i(t-1)| * (commission_bps_i + slippage_bps_i) / 10000

    The total daily cost is the sum across all commodities.

    Parameters
    ----------
    weights : pd.DataFrame
        Daily portfolio weights (DatetimeIndex × commodity columns).
    config : dict
        The ``costs`` section from ``configs/backtest.yaml``.

    Returns
    -------
    pd.Series
        Daily transaction costs (same index as *weights*).  First row is 0
        because there is no prior weight to diff against.
    """
    delta_w = weights.diff().abs()

    cost_rate = pd.Series(
        {
            col: (get_cost(col, "commission_bps", config) + get_cost(col, "slippage_bps", config))
            / 10_000
            for col in weights.columns
        }
    )

    # Broadcast cost_rate across the time dimension
    daily_costs = (delta_w * cost_rate).sum(axis=1)
    daily_costs = daily_costs.fillna(0.0)

    logger.debug(
        "compute_transaction_costs: mean_daily=%.6f, total=%.4f",
        float(daily_costs.mean()),
        float(daily_costs.sum()),
    )
    return daily_costs


def compute_roll_costs(
    weights: pd.DataFrame,
    roll_schedule: pd.DataFrame,
    config: dict,
) -> pd.Series:
    """Daily roll costs applied on front-contract roll days.

    On days when a commodity's front contract rolls (identified via
    *roll_schedule*), the roll cost is::

        roll_cost_i(t) = |w_i(t)| * roll_cost_bps_i / 10000

    if the commodity has a nonzero position on that day.

    Parameters
    ----------
    weights : pd.DataFrame
        Daily portfolio weights (DatetimeIndex × commodity columns).
    roll_schedule : pd.DataFrame
        Output of ``build_roll_schedule``.  Must have columns
        ``["trade_date", "front_dsmnem"]`` (or ``"front_futcode"``).
        A roll is detected when ``front_dsmnem`` changes between consecutive
        rows.
    config : dict
        The ``costs`` section from ``configs/backtest.yaml``.

    Returns
    -------
    pd.Series
        Daily roll costs indexed by the weights index.  Days without a roll
        have cost 0.
    """
    roll_costs = pd.Series(0.0, index=weights.index)

    if roll_schedule is None or roll_schedule.empty:
        return roll_costs

    # Identify roll days: dates where front_dsmnem changes
    sched = roll_schedule.copy()
    sched = sched.dropna(subset=["front_dsmnem"])
    sched = sched.set_index("trade_date").sort_index()

    # A roll occurs when the front contract identifier changes
    rolled = sched["front_dsmnem"].ne(sched["front_dsmnem"].shift(1))
    roll_dates = set(sched.index[rolled])

    for date in weights.index:
        if date not in roll_dates:
            continue
        for col in weights.columns:
            w = weights.at[date, col]
            if pd.isna(w) or w == 0.0:
                continue
            rate = get_cost(col, "roll_cost_bps", config) / 10_000
            roll_costs.at[date] += abs(w) * rate

    logger.debug(
        "compute_roll_costs: %d roll days found, total_cost=%.4f",
        len(roll_dates),
        float(roll_costs.sum()),
    )
    return roll_costs


def apply_costs(
    gross_returns: pd.Series,
    weights: pd.DataFrame,
    config: dict,
    roll_schedule: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Compute total costs and net returns.

    Combines transaction costs (turnover-based) and roll costs (roll-day-based)
    into a total daily cost series, then subtracts from gross returns.

    Parameters
    ----------
    gross_returns : pd.Series
        Daily gross portfolio returns.
    weights : pd.DataFrame
        Daily portfolio weights (DatetimeIndex × commodity columns).
    config : dict
        The ``costs`` section from ``configs/backtest.yaml``.
    roll_schedule : pd.DataFrame, optional
        Output of ``build_roll_schedule``.  If None, roll costs are zero.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        ``(costs, net_returns)`` where ``net_returns = gross_returns - costs``.
    """
    txn_costs = compute_transaction_costs(weights, config)

    if roll_schedule is not None:
        rc = compute_roll_costs(weights, roll_schedule, config)
        total_costs = txn_costs.add(rc, fill_value=0.0)
    else:
        total_costs = txn_costs

    # Align costs to gross_returns index
    total_costs = total_costs.reindex(gross_returns.index, fill_value=0.0)
    net_returns = gross_returns - total_costs

    logger.info(
        "apply_costs: mean_daily_cost=%.6f, total_cost=%.4f",
        float(total_costs.mean()),
        float(total_costs.sum()),
    )
    return total_costs, net_returns


def main() -> None:
    """Smoke-test cost look-ups with a tiny example."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    config = {
        "default": {"commission_bps": 3, "slippage_bps": 2, "roll_cost_bps": 2},
        "per_commodity": {
            "CL": {"commission_bps": 2, "slippage_bps": 1, "roll_cost_bps": 1},
        },
    }
    logger.info("CL commission: %.0f bps", get_cost("CL", "commission_bps", config))
    logger.info("NG commission: %.0f bps", get_cost("NG", "commission_bps", config))
    logger.info("Unknown commission: %.0f bps", get_cost("XX", "commission_bps", config))


if __name__ == "__main__":
    main()
