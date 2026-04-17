"""Macro exposure factor via rolling OLS.

For each commodity, estimates the sensitivity (beta) to three macro factors
using a rolling window OLS regression:

    r_commodity = alpha + beta_usd * r_usd + beta_rate * d_rate + beta_infl * d_infl + eps

The factor signal is the expected return from the current macro state:

    signal(t) = beta_usd(t) * r_usd(t) + beta_rate(t) * d_rate(t) + beta_infl(t) * d_infl(t)

The signal is then expanding z-scored to produce a stationary factor.

Macro factor definitions (all first-differenced or log-returns to be stationary):
- ``r_usd``    : log return of the broad USD index (``usd_index``, from FRED DTWEXBGS).
                 A rising USD is typically bearish for commodities priced in USD.
- ``d_rate``   : first difference of the 10Y Treasury yield (``dgs10``, from FRED DGS10).
                 Rising rates are typically bearish for commodities.
- ``d_infl``   : first difference of the 5Y breakeven inflation rate
                 (``t5yie``, from FRED T5YIE).
                 Rising inflation expectations are typically bullish for commodities.

Usage:
    python -m commodity_curve_factors.factors.macro
"""

import logging

import numpy as np
import pandas as pd

from commodity_curve_factors.factors.transforms import expanding_zscore_df

logger = logging.getLogger(__name__)

# Keys expected in macro_data dict (from load_macro_data())
_KEY_USD = "usd_index"
_KEY_RATE = "dgs10"
_KEY_INFL = "t5yie"


def _extract_macro_factors(
    macro_data: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    """Extract and transform macro series into stationary daily factors.

    Parameters
    ----------
    macro_data : dict[str, pd.DataFrame]
        Output of ``load_macro_data()``. Must contain at least one of the
        three macro keys (``usd_index``, ``dgs10``, ``t5yie``).

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ``["r_usd", "d_rate", "d_infl"]``, indexed
        by business date. Missing series become all-NaN columns.
        Returns None if macro_data is empty.
    """
    if not macro_data:
        logger.warning("_extract_macro_factors: macro_data is empty")
        return None

    result: dict[str, pd.Series] = {}

    if _KEY_USD in macro_data:
        usd = macro_data[_KEY_USD]["value"].dropna()
        result["r_usd"] = pd.Series(np.log(usd / usd.shift(1)), index=usd.index)
    else:
        logger.warning("macro_data missing '%s' — r_usd will be NaN", _KEY_USD)
        result["r_usd"] = pd.Series(dtype=float)

    if _KEY_RATE in macro_data:
        rate = macro_data[_KEY_RATE]["value"].dropna()
        result["d_rate"] = rate.diff()
    else:
        logger.warning("macro_data missing '%s' — d_rate will be NaN", _KEY_RATE)
        result["d_rate"] = pd.Series(dtype=float)

    if _KEY_INFL in macro_data:
        infl = macro_data[_KEY_INFL]["value"].dropna()
        result["d_infl"] = infl.diff()
    else:
        logger.warning("macro_data missing '%s' — d_infl will be NaN", _KEY_INFL)
        result["d_infl"] = pd.Series(dtype=float)

    macro_df = pd.DataFrame(result)
    macro_df.index.name = "Date"
    return macro_df


def _rolling_ols_betas(
    y: pd.Series,
    X: pd.DataFrame,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    """Compute rolling OLS betas for y on X using numpy lstsq.

    At each time step t, uses observations in [t-window+1, t] (inclusive)
    to fit:   y = X @ beta   (no intercept — intercept is handled by mean
    centering within the window, which is equivalent for the beta estimates).

    Parameters
    ----------
    y : pd.Series
        Dependent variable (commodity returns), DatetimeIndex.
    X : pd.DataFrame
        Macro factor returns, same DatetimeIndex as y.
    window : int
        Rolling window size (rows).
    min_periods : int
        Minimum non-NaN observations required; return NaN if fewer available.

    Returns
    -------
    pd.DataFrame
        Same index as y, columns = X.columns, values = rolling OLS betas.
        NaN where fewer than min_periods rows are available.
    """
    k = X.shape[1]
    betas = np.full((len(y), k), np.nan)

    y_arr = y.to_numpy(dtype=float)
    X_arr = X.to_numpy(dtype=float)

    for t in range(len(y)):
        start = max(0, t - window + 1)
        y_win = y_arr[start : t + 1]
        X_win = X_arr[start : t + 1]

        # Drop rows where any value is NaN
        valid = ~(np.isnan(y_win) | np.isnan(X_win).any(axis=1))
        n_valid = int(valid.sum())

        if n_valid < min_periods:
            continue

        y_v = y_win[valid]
        X_v = X_win[valid]

        # Add constant column (intercept) — discard it from output
        ones = np.ones((len(y_v), 1))
        X_aug = np.hstack([ones, X_v])

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_aug, y_v, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # coeffs[0] = intercept, coeffs[1:] = betas for X columns
        betas[t] = coeffs[1:]

    return pd.DataFrame(betas, index=y.index, columns=X.columns)


def compute_macro_factor(
    commodity_returns: pd.DataFrame,
    macro_data: dict[str, pd.DataFrame],
    window: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Macro factor exposure via rolling OLS.

    For each commodity, run a rolling OLS regression of commodity returns
    on macro factor returns::

        r_commodity = alpha + beta_usd * r_usd + beta_rate * d_rate + beta_infl * d_infl + eps

    The factor signal is the expected return from the current macro state::

        signal = beta_usd * r_usd(t) + beta_rate * d_rate(t) + beta_infl * d_infl(t)

    Then expanding z-score the signal.

    Parameters
    ----------
    commodity_returns : pd.DataFrame
        Daily log returns, columns = commodity symbols.
    macro_data : dict[str, pd.DataFrame]
        Output of ``load_macro_data()``. Must contain keys for USD index,
        10Y yield, and inflation breakeven.
    window : int
        Rolling window for OLS regression. Default 252.
    min_periods : int
        Minimum observations for the rolling OLS. Default 60.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, columns = commodity symbols, values = z-scored
        macro exposure signal.
    """
    if commodity_returns.empty:
        logger.warning("compute_macro_factor: empty commodity_returns")
        return pd.DataFrame(columns=commodity_returns.columns)

    macro_raw = _extract_macro_factors(macro_data)
    if macro_raw is None or macro_raw.empty:
        logger.warning("compute_macro_factor: no macro data available — returning all NaN")
        return pd.DataFrame(
            np.nan, index=commodity_returns.index, columns=commodity_returns.columns
        )

    # Align commodity returns and macro factors to a common daily index
    common_idx = commodity_returns.index.intersection(macro_raw.index)
    if len(common_idx) == 0:
        logger.warning("compute_macro_factor: no overlapping dates between returns and macro data")
        return pd.DataFrame(
            np.nan, index=commodity_returns.index, columns=commodity_returns.columns
        )

    ret = commodity_returns.loc[common_idx]
    mf = macro_raw.loc[common_idx]

    # Check how many macro factors are actually available
    available_cols = [c for c in mf.columns if mf[c].notna().any()]
    if not available_cols:
        logger.warning("compute_macro_factor: all macro factors are NaN — returning all NaN")
        return pd.DataFrame(
            np.nan, index=commodity_returns.index, columns=commodity_returns.columns
        )

    mf_avail = mf[available_cols]

    # Compute rolling OLS betas and expected returns for each commodity
    signals: dict[str, pd.Series] = {}

    for sym in ret.columns:
        y = ret[sym]

        # Rolling betas
        betas_df = _rolling_ols_betas(y, mf_avail, window=window, min_periods=min_periods)

        # Expected return from macro state: beta @ macro_factor(t)
        # Element-wise multiply betas by current macro factors, then sum across factors.
        expected_ret = (betas_df * mf_avail).sum(axis=1, min_count=1)
        # Where any beta is NaN (pre min_periods), expected_ret is already NaN via min_count.

        signals[sym] = expected_ret

    signal_df = pd.DataFrame(signals)

    # Reindex to the full commodity_returns index (fill with NaN for out-of-sample)
    signal_df = signal_df.reindex(commodity_returns.index)

    # Expanding z-score
    factor: pd.DataFrame = expanding_zscore_df(signal_df, min_periods=min_periods)

    logger.info(
        "compute_macro_factor: %d commodities, %d dates, %.1f%% non-NaN",
        len(factor.columns),
        len(factor),
        factor.notna().mean().mean() * 100,
    )
    return factor


def main() -> None:
    """Smoke-test the macro factor with saved macro and front-month data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from commodity_curve_factors.data.futures_loader import load_front_month_data
    from commodity_curve_factors.data.macro_loader import load_macro_data

    macro_data = load_macro_data()
    front_month = load_front_month_data()

    if not macro_data:
        logger.error("No macro data loaded — run macro_loader first")
        return
    if not front_month:
        logger.error("No front-month data loaded — run futures_loader first")
        return

    # Build log returns
    close_prices = pd.DataFrame(
        {sym: df["Close"] for sym, df in front_month.items() if "Close" in df.columns}
    )
    returns: pd.DataFrame = pd.DataFrame(
        np.log(close_prices / close_prices.shift(1)),
        index=close_prices.index,
        columns=close_prices.columns,
    )

    factor = compute_macro_factor(returns, macro_data, window=252, min_periods=60)
    logger.info("Factor shape: %s", factor.shape)
    logger.info("Non-NaN per commodity:\n%s", factor.notna().sum())


if __name__ == "__main__":
    main()
