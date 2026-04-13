"""Core performance metrics with IS/OOS splitting."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252
_IS_END = "2017-12-31"
_OOS_START = "2018-01-01"


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    rf : float
        Daily risk-free rate. Default 0.0.

    Returns
    -------
    float
        Annualised Sharpe ratio. Returns 0.0 if std is zero or series is empty.
    """
    excess = returns - rf
    std = float(excess.std())
    if std == 0 or len(returns) == 0:
        return 0.0
    result: float = float(excess.mean()) / std * np.sqrt(_TRADING_DAYS)
    return result


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sortino ratio (downside-only volatility).

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    rf : float
        Daily risk-free rate. Default 0.0.

    Returns
    -------
    float
        Annualised Sortino ratio. Returns 0.0 if no negative excess returns.
    """
    excess = returns - rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    down_std = float(downside.std())
    if down_std == 0:
        return 0.0
    result: float = float(excess.mean()) / down_std * np.sqrt(_TRADING_DAYS)
    return result


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.

    Returns
    -------
    float
        Maximum drawdown as a negative fraction (e.g. -0.25 = -25%).
    """
    cum = np.exp(returns.cumsum())
    dd = cum / cum.cummax() - 1
    result: float = float(dd.min())
    return result


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.

    Returns
    -------
    float
        CAGR as a fraction (e.g. 0.10 = 10% per year).
    """
    n = len(returns)
    if n == 0:
        return 0.0
    years = n / _TRADING_DAYS
    cum = float(np.exp(returns.sum()))
    if cum <= 0 or years <= 0:
        return 0.0
    result: float = cum ** (1.0 / years) - 1.0
    return result


def annual_volatility(returns: pd.Series) -> float:
    """Annualised volatility (standard deviation).

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.

    Returns
    -------
    float
        Annualised volatility as a fraction.
    """
    result: float = float(returns.std()) * np.sqrt(_TRADING_DAYS)
    return result


def calmar_ratio(returns: pd.Series) -> float:
    """Calmar ratio (CAGR / absolute max drawdown).

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.

    Returns
    -------
    float
        Calmar ratio. Returns 0.0 if max drawdown is zero.
    """
    dd = max_drawdown(returns)
    if dd == 0:
        return 0.0
    result: float = cagr(returns) / abs(dd)
    return result


def hit_rate(returns: pd.Series) -> float:
    """Fraction of days with positive returns.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.

    Returns
    -------
    float
        Hit rate in [0, 1]. Returns 0.0 for empty series.
    """
    if len(returns) == 0:
        return 0.0
    result: float = float((returns > 0).mean())
    return result


def compute_all_metrics(returns: pd.Series, rf: float = 0.0) -> dict[str, float]:
    """Compute all standard performance metrics in one call.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    rf : float
        Daily risk-free rate. Default 0.0.

    Returns
    -------
    dict[str, float]
        Keys: ``sharpe``, ``sortino``, ``calmar``, ``max_drawdown``,
        ``cagr``, ``volatility``, ``hit_rate``.
    """
    return {
        "sharpe": sharpe_ratio(returns, rf),
        "sortino": sortino_ratio(returns, rf),
        "calmar": calmar_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "cagr": cagr(returns),
        "volatility": annual_volatility(returns),
        "hit_rate": hit_rate(returns),
    }


def split_is_oos(
    returns: pd.Series,
    is_end: str = _IS_END,
    oos_start: str = _OOS_START,
) -> tuple[pd.Series, pd.Series]:
    """Split a return series into in-sample and out-of-sample windows.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns with a DatetimeIndex.
    is_end : str
        Last date (inclusive) of the in-sample period. Default ``"2017-12-31"``.
    oos_start : str
        First date (inclusive) of the out-of-sample period. Default ``"2018-01-01"``.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        ``(is_returns, oos_returns)`` — each a slice of the input series.
    """
    is_r: pd.Series = returns.loc[:is_end]  # type: ignore[misc]
    oos_r: pd.Series = returns.loc[oos_start:]  # type: ignore[misc]
    return is_r, oos_r
