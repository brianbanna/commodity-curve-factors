"""Log-linear curve interpolation for commodity futures term structures.

Algorithmic design
------------------
The curve is built by fitting an OLS line to ``(years_to_expiry, log(price))``
for each day's cross-section of active contracts, then evaluating at a fixed
set of standard tenor points.

Handling WTI 2020-04-20 negative prices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
On 2020-04-20 the May-2020 WTI contract settled at -$37.63 while all
back-month contracts remained positive.  The chosen approach is to **exclude**
any contract with ``price <= price_floor`` from the OLS fit rather than to
floor the bad price.  Flooring at a small positive value would produce a
wildly distorted log-linear front — the curve would appear steeply
backwardated in a way that does not reflect the market for any contract that
actually trades.  Excluding the one bad point preserves the shape of the
positive back-month structure so the curve still carries useful signal.

Extrapolation policy
~~~~~~~~~~~~~~~~~~~~
Interpolation within the observed tenor range is always allowed.  Extrapolation
beyond the outermost contract is only allowed when the gap between the target
tenor and the nearest observed tenor is at most ``extrapolation_max_days``
calendar days (default 45).  This prevents GC/SI quarterly contracts from
producing spurious F1M / F2M values that are really unbounded extrapolations.
"""

import datetime
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DAYS_PER_YEAR: float = 365.25


def time_to_expiry_years(
    trade_date: pd.Timestamp,
    expiry_date: pd.Timestamp | datetime.date,
) -> float:
    """Years from ``trade_date`` to ``expiry_date``.

    Parameters
    ----------
    trade_date : pd.Timestamp
        Reference date (today's trade date).
    expiry_date : pd.Timestamp or datetime.date
        Contract last-trade or settlement date.

    Returns
    -------
    float
        Fractional years (can be negative for expired contracts).
        NaN if either argument is NaT / None.
    """
    if pd.isna(trade_date):
        return float("nan")
    try:
        if pd.isna(expiry_date):
            return float("nan")
    except (TypeError, ValueError):
        pass

    expiry_ts = pd.Timestamp(expiry_date)
    delta_days = (expiry_ts - trade_date).days
    return delta_days / DAYS_PER_YEAR


def log_linear_interpolate(
    tenors_years: np.ndarray,
    prices: np.ndarray,
    target_years: np.ndarray,
    *,
    extrapolation_max_days: int = 45,
    min_points: int = 3,
    price_floor: float = 0.01,
) -> np.ndarray:
    """Log-linear interpolation of ``prices`` onto ``target_years``.

    Parameters
    ----------
    tenors_years : np.ndarray
        1-D array of years-to-expiry for each available contract.
    prices : np.ndarray
        1-D array of settlement prices (same length as tenors_years).
    target_years : np.ndarray
        1-D array of target tenors in years (e.g. [1/12, 2/12, …, 12/12]).
    extrapolation_max_days : int
        Maximum gap in days between a target and the nearest observed tenor
        when extrapolating.  Targets outside this gap receive NaN.
    min_points : int
        Minimum number of valid (positive, non-NaN) input points required.
    price_floor : float
        Prices at or below this threshold are treated as missing.

    Returns
    -------
    np.ndarray
        Interpolated prices at ``target_years``.  NaN for insufficient data
        or extrapolation beyond the limit.
    """
    result = np.full(len(target_years), np.nan)

    tenors_years = np.asarray(tenors_years, dtype=float)
    prices = np.asarray(prices, dtype=float)
    target_years = np.asarray(target_years, dtype=float)

    # Step 1: keep only valid (positive, non-NaN) points
    valid = np.isfinite(prices) & (prices > price_floor) & np.isfinite(tenors_years)
    tenors_valid = tenors_years[valid]
    prices_valid = prices[valid]

    # Step 2: require minimum points
    if len(tenors_valid) < min_points:
        return result

    # Step 3: OLS fit on (tenor, log(price))
    log_prices = np.log(prices_valid)
    slope, intercept = np.polyfit(tenors_valid, log_prices, deg=1)

    # Step 4 & 5: evaluate and apply extrapolation guard
    t_min = tenors_valid.min()
    t_max = tenors_valid.max()
    extrap_limit_years = extrapolation_max_days / DAYS_PER_YEAR

    for i, t in enumerate(target_years):
        if np.isnan(t):
            continue
        # Check extrapolation gap
        if t < t_min:
            gap = t_min - t
        elif t > t_max:
            gap = t - t_max
        else:
            gap = 0.0

        if gap > extrap_limit_years:
            continue  # leave as NaN

        result[i] = np.exp(intercept + slope * t)

    return result


def interpolate_curve_day(
    contracts_day: pd.DataFrame,
    standard_tenors_months: list[int],
    *,
    extrapolation_max_days: int = 45,
    min_contracts: int = 3,
    price_floor: float = 0.01,
) -> pd.Series:
    """Build one day's interpolated curve from a cross-section of contracts.

    Parameters
    ----------
    contracts_day : pd.DataFrame
        Rows for a single ``trade_date``.  Must have columns
        ``trade_date``, ``lasttrddate``, ``settlement``.
    standard_tenors_months : list[int]
        Target tenor points in months (e.g. [1, 2, 3, 6, 9, 12]).
    extrapolation_max_days : int
        Forwarded to :func:`log_linear_interpolate`.
    min_contracts : int
        Forwarded as ``min_points`` to :func:`log_linear_interpolate`.
    price_floor : float
        Forwarded to :func:`log_linear_interpolate`.

    Returns
    -------
    pd.Series
        Index = tenor labels (e.g. ``["F1M", "F2M", …, "F12M"]``),
        values = interpolated prices (NaN where insufficient data).
    """
    labels = [f"F{m}M" for m in standard_tenors_months]
    target_years = np.array([m / 12.0 for m in standard_tenors_months])

    if contracts_day.empty:
        return pd.Series(np.nan, index=labels)

    # Use the first trade_date as the reference (all rows share one date)
    trade_date = pd.Timestamp(contracts_day["trade_date"].iloc[0])

    # Compute time-to-expiry for each contract
    tenors = np.array(
        [
            time_to_expiry_years(trade_date, row["lasttrddate"])
            for _, row in contracts_day.iterrows()
        ]
    )
    prices = contracts_day["settlement"].to_numpy(dtype=float)

    interpolated = log_linear_interpolate(
        tenors,
        prices,
        target_years,
        extrapolation_max_days=extrapolation_max_days,
        min_points=min_contracts,
        price_floor=price_floor,
    )

    return pd.Series(interpolated, index=labels)
