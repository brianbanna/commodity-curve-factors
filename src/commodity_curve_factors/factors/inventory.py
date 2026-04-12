"""EIA Inventory Surprise factor.

Computes the surprise as the actual week-over-week inventory change minus the
5-year seasonal expectation, normalized by the expanding standard deviation.
Only 4 energy commodities have EIA inventory data (CL, NG, HO, RB); the
remaining 9 in the universe receive NaN columns and are handled by
NaN-tolerant factor combination in the pipeline.

The factor output is daily (forward-filled from each weekly release to the
next), because the inventory_loader.align_to_daily function already shifts
each weekly observation to its EIA release date (Wednesday or Thursday) and
forward-fills across business days.

Usage:
    python -m commodity_curve_factors.factors.inventory
"""

import logging

import numpy as np
import pandas as pd

from commodity_curve_factors.factors.transforms import expanding_zscore

logger = logging.getLogger(__name__)

# Default commodity → EIA series name mapping for the 4 energy commodities
# that have accessible inventory data.
COMMODITY_MAP: dict[str, str] = {
    "CL": "crude_stocks",
    "NG": "natural_gas_storage",
    "HO": "distillate_stocks",
    "RB": "gasoline_stocks",
}


def compute_seasonal_expectation(
    series: pd.Series,
    years: int = 5,
) -> pd.Series:
    """5-year average change for each calendar week.

    At each date, compute the average week-over-week inventory change
    for the same ISO calendar week over the preceding ``years`` years.
    Uses only past data (no lookahead) — at week W of year Y, the average
    uses years Y-1, Y-2, ..., Y-years.

    Parameters
    ----------
    series : pd.Series
        Weekly inventory levels with a DatetimeIndex.
    years : int
        Number of historical years to include in the seasonal average.
        Default 5.

    Returns
    -------
    pd.Series
        Series aligned to the input index with the expected
        inventory change for that week. NaN where fewer than one
        historical year of the same week is available.
    """
    wow_change = series.diff()

    iso_week = series.index.isocalendar().week.astype(int).to_numpy()
    iso_year = series.index.isocalendar().year.astype(int).to_numpy()

    expectations = pd.Series(np.nan, index=series.index)

    for i in range(len(series)):
        target_week = iso_week[i]
        target_year = iso_year[i]

        # Collect changes from the same ISO week in the preceding `years` years
        historical_years = set(range(target_year - years, target_year))
        mask = (iso_week == target_week) & np.array([y in historical_years for y in iso_year])

        historical_changes = wow_change.iloc[:i][mask[:i]]
        if len(historical_changes) >= 1:
            expectations.iloc[i] = historical_changes.mean()

    return expectations


def compute_inventory_surprise(
    series: pd.Series,
    years: int = 5,
) -> pd.Series:
    """Inventory surprise: (actual_change - expected_change) / expanding_std.

    The surprise is the actual week-over-week change minus the seasonal
    expectation, normalized by the expanding standard deviation of the
    surprise series itself.

    Parameters
    ----------
    series : pd.Series
        Weekly inventory levels with a DatetimeIndex.
    years : int
        Number of historical years for the seasonal expectation. Default 5.

    Returns
    -------
    pd.Series
        Expanding z-score of the inventory surprise. NaN where insufficient
        history or where seasonal expectation is unavailable.
    """
    wow_change = series.diff()
    expected = compute_seasonal_expectation(series, years=years)
    raw_surprise = wow_change - expected

    # Use min_periods=1 so we get values as soon as there is one surprise
    # observation; callers that want a longer warmup can filter themselves.
    z_surprise = expanding_zscore(raw_surprise, min_periods=1)
    return z_surprise


def compute_all_inventory_surprises(
    inventory_data: dict[str, pd.DataFrame],
    commodity_map: dict[str, str],
    all_commodities: list[str],
    years: int = 5,
) -> pd.DataFrame:
    """Build inventory surprise factor for all commodities.

    Parameters
    ----------
    inventory_data : dict
        Output of ``load_inventory_data()``. Keys are series names
        (e.g. "crude_stocks"), values are DataFrames. Each DataFrame
        must have a ``value`` column. The data may be weekly (raw) or
        already aligned to daily business days via ``align_to_daily``.
    commodity_map : dict
        Maps commodity symbol to inventory series name, e.g.
        {"CL": "crude_stocks", "NG": "natural_gas_storage", ...}.
        Only commodities in this map get non-NaN values.
    all_commodities : list[str]
        Full universe of commodity symbols. Commodities NOT in
        ``commodity_map`` get NaN columns (no inventory data available).
    years : int
        Lookback years for seasonal expectation. Default 5.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (daily), columns = all_commodities, values = surprise
        z-scores. NaN for non-energy commodities. Aligned to daily via
        forward-fill (weekly data → daily).
    """
    factor_columns: dict[str, pd.Series] = {}

    for sym in all_commodities:
        series_name = commodity_map.get(sym)
        if series_name is None or series_name not in inventory_data:
            logger.debug("No inventory data for %s — column will be NaN", sym)
            factor_columns[sym] = pd.Series(dtype=float, name=sym)
            continue

        df = inventory_data[series_name]
        if "value" not in df.columns:
            logger.warning(
                "inventory_data[%s] has no 'value' column — skipping %s", series_name, sym
            )
            factor_columns[sym] = pd.Series(dtype=float, name=sym)
            continue

        raw_series = df["value"].dropna()
        if raw_series.empty:
            logger.warning("inventory_data[%s] is empty — skipping %s", series_name, sym)
            factor_columns[sym] = pd.Series(dtype=float, name=sym)
            continue

        # Determine whether the series is already on a daily business-day
        # index (as produced by align_to_daily) or still weekly.
        # We detect weekly data by checking if the median gap between
        # consecutive observations is around 7 days.
        if len(raw_series) > 1:
            median_gap = pd.Series(raw_series.index).diff().median()
            is_weekly = median_gap >= pd.Timedelta(days=5)
        else:
            is_weekly = True

        if is_weekly:
            # Compute surprise on the weekly series directly, then
            # forward-fill to daily business days.
            z_weekly = compute_inventory_surprise(raw_series, years=years)
            bday_range = pd.bdate_range(start=raw_series.index.min(), end=raw_series.index.max())
            z_daily = z_weekly.reindex(bday_range, method="ffill")
            z_daily.name = sym
            factor_columns[sym] = z_daily
        else:
            # Daily data already aligned (e.g. via align_to_daily).
            # Resample back to weekly for surprise computation, then
            # z-score and forward-fill.
            weekly = raw_series.resample("W-FRI").last().dropna()
            z_weekly = compute_inventory_surprise(weekly, years=years)
            bday_range = pd.bdate_range(start=raw_series.index.min(), end=raw_series.index.max())
            z_daily = z_weekly.reindex(bday_range, method="ffill")
            z_daily.name = sym
            factor_columns[sym] = z_daily

        logger.info(
            "Inventory surprise for %s (%s): %d weekly observations → %d daily rows, "
            "%.1f%% non-NaN",
            sym,
            series_name,
            len(raw_series) if is_weekly else len(weekly),
            len(z_daily),
            z_daily.notna().mean() * 100,
        )

    # Align all series to a common daily index (union) and return.
    if not any(len(s) > 0 for s in factor_columns.values()):
        logger.warning("No inventory surprise data computed for any commodity")
        return pd.DataFrame(columns=all_commodities)

    result = pd.DataFrame(factor_columns)
    result = result[all_commodities]
    result.index.name = "Date"

    logger.info(
        "compute_all_inventory_surprises: %d commodities, %d dates, %.1f%% non-NaN overall",
        len(result.columns),
        len(result),
        result.notna().mean().mean() * 100,
    )
    return result


def main() -> None:
    """Smoke-test the inventory surprise factor with saved EIA data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from commodity_curve_factors.data.inventory_loader import load_inventory_data
    from commodity_curve_factors.utils.constants import ALL_COMMODITIES

    inventory_data = load_inventory_data()
    if not inventory_data:
        logger.error("No inventory data loaded — run inventory_loader first")
        return

    factor = compute_all_inventory_surprises(
        inventory_data=inventory_data,
        commodity_map=COMMODITY_MAP,
        all_commodities=ALL_COMMODITIES,
        years=5,
    )
    logger.info("Factor shape: %s", factor.shape)
    logger.info("Non-NaN per commodity:\n%s", factor.notna().sum())


if __name__ == "__main__":
    main()
