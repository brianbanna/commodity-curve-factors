"""Benchmark strategies for performance comparison.

Provides equal-weight long-only, cash (risk-free), and market benchmarks
(SPY / AGG) against which factor strategies are evaluated.

Usage:
    python -m commodity_curve_factors.backtest.benchmarks
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def equal_weight_long(returns: pd.DataFrame) -> pd.Series:
    """Equal-weight long-only benchmark: 1/N weight on each commodity.

    Assigns a constant weight of ``1 / N`` to each of the *N* columns on every
    day and returns the resulting daily portfolio returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns (DatetimeIndex × commodity columns).

    Returns
    -------
    pd.Series
        Daily portfolio returns.  NaN rows in *returns* are propagated as the
        mean of available columns on that day.
    """
    n_assets = returns.shape[1]
    if n_assets == 0:
        logger.warning("equal_weight_long: no columns in returns — returning zeros")
        return pd.Series(0.0, index=returns.index)

    weight = 1.0 / n_assets
    port_ret = returns.mean(axis=1) * (weight * n_assets)  # equivalent to returns.mean(axis=1)

    logger.debug(
        "equal_weight_long: n=%d, mean_return=%.5f",
        n_assets,
        float(port_ret.mean()),
    )
    return port_ret


def cash_benchmark(
    dates: pd.DatetimeIndex,
    annual_rate: float = 0.02,
) -> pd.Series:
    """Cash (risk-free) benchmark: constant daily return = annual_rate / 252.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Date index for the returned series.
    annual_rate : float
        Annualised risk-free rate (e.g. 0.02 for 2%). Default 0.02.

    Returns
    -------
    pd.Series
        Daily returns, constant at ``annual_rate / 252``.
    """
    daily_rate = annual_rate / 252.0
    series = pd.Series(daily_rate, index=dates, name="cash")

    logger.debug(
        "cash_benchmark: annual_rate=%.4f, daily_rate=%.6f, n=%d",
        annual_rate,
        daily_rate,
        len(series),
    )
    return series


def load_market_benchmarks(
    macro_data: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.Series]:
    """Load SPY and AGG total return series.

    If *macro_data* is provided, extracts returns from the ``"spy"`` and
    ``"agg"`` DataFrames already in memory.  Otherwise calls
    :func:`~commodity_curve_factors.data.macro_loader.load_macro_data` to
    read from disk.

    Daily returns are computed as ``Close.pct_change()``; the first row
    (always NaN) is dropped.

    Parameters
    ----------
    macro_data : dict, optional
        Pre-loaded macro data dict (keyed by series name, values are
        DataFrames with a ``Close`` column).

    Returns
    -------
    dict[str, pd.Series]
        Keys ``"SPY"`` and ``"AGG"``, values = daily return Series.  A key is
        omitted if the underlying data is unavailable.
    """
    if macro_data is None:
        from commodity_curve_factors.data.macro_loader import load_macro_data

        macro_data = load_macro_data()

    result: dict[str, pd.Series] = {}

    key_map = {"spy": "SPY", "agg": "AGG"}

    for raw_key, output_key in key_map.items():
        df = macro_data.get(raw_key)
        if df is None or df.empty:
            logger.warning("load_market_benchmarks: no data for %s", raw_key)
            continue

        if "Close" not in df.columns:
            # Try Adj Close or first column as fallback
            price_col = "Adj Close" if "Adj Close" in df.columns else df.columns[0]
            logger.debug(
                "load_market_benchmarks: %s — 'Close' not found, using '%s'",
                raw_key,
                price_col,
            )
        else:
            price_col = "Close"

        returns = df[price_col].pct_change().dropna()
        returns.name = output_key
        result[output_key] = returns

        logger.debug(
            "load_market_benchmarks: %s — %d return observations, %s to %s",
            output_key,
            len(returns),
            returns.index[0].date() if len(returns) > 0 else "N/A",
            returns.index[-1].date() if len(returns) > 0 else "N/A",
        )

    logger.info("load_market_benchmarks: loaded %d benchmark series", len(result))
    return result


def main() -> None:
    """Smoke-test benchmarks with synthetic data."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    cols = ["CL", "GC", "NG", "ZC"]
    returns = pd.DataFrame(rng.normal(0, 0.01, (252, 4)), index=dates, columns=cols)

    ew = equal_weight_long(returns)
    logger.info("EW mean return: %.5f", float(ew.mean()))

    cash = cash_benchmark(dates, annual_rate=0.02)
    logger.info("Cash daily rate: %.6f", float(cash.iloc[0]))


if __name__ == "__main__":
    main()
