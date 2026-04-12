"""Volatility regime factor.

Computes a volatility regime signal based on the ratio of short-term to
long-term realized volatility for each commodity:

    vol_regime(t) = realized_vol(short_window)(t) / realized_vol(long_window)(t)

When the ratio > 1, short-term volatility exceeds long-term volatility,
indicating an elevated or heating regime. When < 1, short-term volatility
is below the long-term average, indicating a calm/low-vol regime.

The ratio is then expanding z-scored to produce a stationary factor signal.

Usage:
    python -m commodity_curve_factors.factors.volatility
"""

import logging

import numpy as np
import pandas as pd

from commodity_curve_factors.factors.transforms import expanding_zscore_df
from commodity_curve_factors.utils.constants import TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)


def realized_volatility(
    returns: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Annualized realized volatility from daily returns.

    vol = std(returns, window) * sqrt(252)

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns, columns = commodity symbols.
    window : int
        Rolling window in trading days. Default 20 (~1 month).

    Returns
    -------
    pd.DataFrame
        Same shape as *returns*. Annualized rolling realized volatility.
        NaN for the first ``window - 1`` rows.
    """
    annualization = np.sqrt(TRADING_DAYS_PER_YEAR)
    return returns.rolling(window=window, min_periods=window).std(ddof=1) * annualization


def vol_regime_ratio(
    returns: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 252,
    min_periods: int = 252,
) -> pd.DataFrame:
    """Volatility regime factor: short vol / long vol, z-scored.

    When the ratio > 1, short-term vol exceeds long-term vol (regime is
    heating up). When < 1, short vol is below long-term (calm period).
    The ratio is z-scored with an expanding window.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns, columns = commodity symbols.
    short_window : int
        Window for short-term vol. Default 20 (~1 month).
    long_window : int
        Window for long-term vol. Default 252 (~1 year).
    min_periods : int
        Minimum observations for the expanding z-score. Default 252.

    Returns
    -------
    pd.DataFrame
        Z-scored vol regime ratio per commodity. Same shape as *returns*.
        NaN until both vol windows have filled (i.e. first ``long_window``
        rows) and until the expanding z-score has ``min_periods`` observations.
    """
    short_vol = realized_volatility(returns, window=short_window)
    long_vol = realized_volatility(returns, window=long_window)

    # Avoid division by zero: where long_vol is zero, ratio is undefined
    ratio = short_vol / long_vol.replace(0, np.nan)

    factor = expanding_zscore_df(ratio, min_periods=min_periods)

    logger.info(
        "vol_regime_ratio: %d commodities, %d dates, %.1f%% non-NaN",
        len(factor.columns),
        len(factor),
        factor.notna().mean().mean() * 100,
    )
    return factor


def main() -> None:
    """Smoke-test the volatility regime factor with saved front-month data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from commodity_curve_factors.data.futures_loader import load_front_month_data

    front_month = load_front_month_data()
    if not front_month:
        logger.error("No front-month data — run futures_loader first")
        return

    close_prices = pd.DataFrame(
        {sym: df["Close"] for sym, df in front_month.items() if "Close" in df.columns}
    )
    returns = np.log(close_prices / close_prices.shift(1))

    factor = vol_regime_ratio(returns, short_window=20, long_window=252, min_periods=252)
    logger.info("Factor shape: %s", factor.shape)
    logger.info("Non-NaN per commodity:\n%s", factor.notna().sum())


if __name__ == "__main__":
    main()
