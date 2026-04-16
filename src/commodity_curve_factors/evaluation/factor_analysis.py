"""Factor-level analysis: information coefficients, decay, correlations."""

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def information_coefficient(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    lag: int = 1,
) -> pd.Series:
    """Cross-sectional Spearman IC between factor values and forward returns.

    Parameters
    ----------
    factor : pd.DataFrame
        Factor values (dates x commodities).
    forward_returns : pd.DataFrame
        Forward returns over the same universe.
    lag : int
        Number of days to shift factor signal before comparing to returns.

    Returns
    -------
    pd.Series
        Daily IC values with DatetimeIndex.
    """
    shifted = factor.shift(lag)
    common_idx = shifted.index.intersection(forward_returns.index)
    shifted = shifted.loc[common_idx]
    fwd = forward_returns.loc[common_idx]

    ics = []
    dates = []
    for dt in common_idx:
        f_row = shifted.loc[dt].dropna()
        r_row = fwd.loc[dt].reindex(f_row.index).dropna()
        valid = f_row.index.intersection(r_row.index)
        if len(valid) < 3:
            continue
        corr, _ = spearmanr(f_row[valid], r_row[valid])
        if np.isfinite(corr):
            ics.append(corr)
            dates.append(dt)

    result = pd.Series(ics, index=pd.DatetimeIndex(dates), name="ic")
    logger.info(
        "information_coefficient: lag=%d, %d dates, mean_ic=%.4f",
        lag,
        len(result),
        result.mean() if len(result) > 0 else 0.0,
    )
    return result


def rolling_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    window: int = 252,
    lag: int = 1,
) -> pd.Series:
    """Rolling mean IC over a trailing window.

    Parameters
    ----------
    factor : pd.DataFrame
        Factor values (dates x commodities).
    forward_returns : pd.DataFrame
        Forward returns.
    window : int
        Rolling window in trading days.
    lag : int
        Factor-return lag in days.

    Returns
    -------
    pd.Series
        Rolling mean IC with DatetimeIndex.
    """
    daily_ic = information_coefficient(factor, forward_returns, lag=lag)
    return daily_ic.rolling(window, min_periods=window // 2).mean()


def ic_decay(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """IC at multiple lags to measure signal persistence.

    Parameters
    ----------
    factor : pd.DataFrame
        Factor values (dates x commodities).
    returns : pd.DataFrame
        Daily returns (used to compute forward returns at each lag).
    lags : list[int] or None
        Lag values to test. Default [1, 5, 10, 20].

    Returns
    -------
    pd.DataFrame
        Columns: lag, mean_ic, std_ic, t_stat, n_obs.
    """
    if lags is None:
        lags = [1, 5, 10, 20]

    rows = []
    for lag in lags:
        fwd = returns.rolling(lag).sum().shift(-lag)
        ic = information_coefficient(factor, fwd, lag=0)
        n = len(ic)
        mean_ic = float(ic.mean()) if n > 0 else 0.0
        std_ic = float(ic.std()) if n > 1 else 0.0
        t = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 and n > 0 else 0.0
        rows.append({"lag": lag, "mean_ic": mean_ic, "std_ic": std_ic, "t_stat": t, "n_obs": n})

    result = pd.DataFrame(rows)
    logger.info("ic_decay: %d lags, max |t|=%.2f", len(lags), result["t_stat"].abs().max())
    return result


def factor_correlations(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cross-sectional average correlation between factor pairs.

    For each date, compute pairwise Spearman correlations between factors,
    then average across dates.

    Parameters
    ----------
    factors : dict[str, pd.DataFrame]
        Factor name -> (dates x commodities) DataFrame.

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix (factor x factor).
    """
    names = sorted(factors.keys())
    n = len(names)
    corr_matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            fi = factors[names[i]]
            fj = factors[names[j]]
            common_idx = fi.index.intersection(fj.index)
            common_cols = fi.columns.intersection(fj.columns)
            fi_aligned = fi.loc[common_idx, common_cols]
            fj_aligned = fj.loc[common_idx, common_cols]

            corrs = []
            for dt in common_idx:
                row_i = fi_aligned.loc[dt].dropna()
                row_j = fj_aligned.loc[dt].reindex(row_i.index).dropna()
                valid = row_i.index.intersection(row_j.index)
                if len(valid) < 3:
                    continue
                c, _ = spearmanr(row_i[valid], row_j[valid])
                if np.isfinite(c):
                    corrs.append(c)
            mean_corr = np.mean(corrs) if corrs else 0.0
            corr_matrix[i, j] = mean_corr
            corr_matrix[j, i] = mean_corr

    result = pd.DataFrame(corr_matrix, index=names, columns=names)
    logger.info("factor_correlations: %d factors, shape=%s", n, result.shape)
    return result


def cumulative_factor_returns(
    factors: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    long_n: int = 3,
    short_n: int = 3,
) -> pd.DataFrame:
    """Long-short cumulative returns for each factor individually.

    For each factor, go long the top-N ranked commodities and short the bottom-N,
    equal-weighted, rebalanced daily.

    Parameters
    ----------
    factors : dict[str, pd.DataFrame]
        Factor name -> (dates x commodities) DataFrame.
    returns : pd.DataFrame
        Daily returns (dates x commodities).
    long_n, short_n : int
        Number of long/short positions.

    Returns
    -------
    pd.DataFrame
        Cumulative log returns per factor (dates x factor_names).
    """
    cum_returns = {}
    for name, factor in sorted(factors.items()):
        common_idx = factor.index.intersection(returns.index)
        common_cols = factor.columns.intersection(returns.columns)
        f = factor.loc[common_idx, common_cols]
        r = returns.loc[common_idx, common_cols]

        daily_ret = []
        dates = []
        for dt in common_idx:
            f_row = f.loc[dt].dropna()
            if len(f_row) < long_n + short_n:
                continue
            ranked = f_row.rank(ascending=True)
            longs = ranked.nlargest(long_n).index
            shorts = ranked.nsmallest(short_n).index
            ret_day = r.loc[dt]
            ls_ret = ret_day[longs].mean() - ret_day[shorts].mean()
            if np.isfinite(ls_ret):
                daily_ret.append(ls_ret)
                dates.append(dt)

        if daily_ret:
            s = pd.Series(daily_ret, index=pd.DatetimeIndex(dates))
            cum_returns[name] = s.cumsum()

    result = pd.DataFrame(cum_returns)
    logger.info(
        "cumulative_factor_returns: %d factors, %d dates", len(cum_returns), len(result)
    )
    return result
