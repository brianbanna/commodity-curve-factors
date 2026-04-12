"""CFTC Positioning contrarian factor.

Computes a contrarian signal from managed-money net speculative positions
in each commodity futures market. When positioning is extremely crowded
(high percentile rank), the signal is negative (fade the crowd). When
positioning is extremely short, the signal is positive.

The factor pipeline:
1. Lag COT data to the Friday release date via ``lag_to_release_date``.
2. Pivot to wide format via ``compute_net_speculative``.
3. Compute rolling percentile rank of net speculative positions over the
   trailing ``window`` weeks (~3 years at 156 weeks).
4. Transform to contrarian signal: ``1 - 2 * percentile``.
   Percentile = 1.0 (max long) → signal = -1.0 (max negative/fade).
   Percentile = 0.0 (max short) → signal = +1.0 (max positive/buy).
5. Forward-fill weekly signal to daily business-day frequency.

Usage:
    python -m commodity_curve_factors.factors.positioning
"""

import logging

import pandas as pd

from commodity_curve_factors.data.cftc_loader import lag_to_release_date
from commodity_curve_factors.factors.transforms import percentile_rank

logger = logging.getLogger(__name__)


def compute_positioning_factor(
    cot: pd.DataFrame,
    window: int = 156,
    all_commodities: list[str] | None = None,
) -> pd.DataFrame:
    """CFTC positioning contrarian signal.

    Steps:
    1. Lag COT data to release date (Friday) via ``lag_to_release_date``
    2. Pivot to wide format via ``compute_net_speculative``
    3. Compute percentile rank of net speculative positions over trailing
       ``window`` weeks (~3 years at 52 weeks/year → 156 weeks)
    4. Transform to contrarian signal: ``1 - 2 * percentile``. When
       positioning is extremely long (high percentile), the contrarian
       signal is negative (crowded long → fade). When positioning is
       extremely short, the signal is positive.
    5. Forward-fill weekly data to daily frequency

    Parameters
    ----------
    cot : pd.DataFrame
        Long-format COT data from ``load_cot_data()``. Must have columns
        ``report_date``, ``commodity``, ``mm_net``.
    window : int
        Rolling window for percentile rank, in weeks. Default 156 (~3 years).
    all_commodities : list[str], optional
        Full universe for column alignment. Missing commodities get NaN.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (daily), columns = commodity symbols, values = contrarian
        signal in [-1, 1]. Daily frequency via forward-fill.
    """
    if cot.empty:
        logger.warning("compute_positioning_factor: empty COT input")
        cols = all_commodities if all_commodities is not None else []
        return pd.DataFrame(columns=cols)

    # Step 1: lag to release date
    cot_lagged = lag_to_release_date(cot)

    # Step 2: pivot to wide format using release_date as the index
    # compute_net_speculative uses report_date by default, so we build the
    # wide DataFrame directly from the lagged frame.
    wide = cot_lagged.pivot_table(
        index="release_date",
        columns="commodity",
        values="mm_net",
        aggfunc="last",
    )
    wide = wide.sort_index()
    wide.columns.name = None
    wide.index.name = "release_date"

    # Step 3: rolling percentile rank per column (on weekly data)
    pct_df = wide.apply(percentile_rank, window=window)

    # Step 4: contrarian transform
    signal_weekly = 1.0 - 2.0 * pct_df

    # Step 5: forward-fill to daily business days
    if len(signal_weekly) == 0:
        cols = all_commodities if all_commodities is not None else list(signal_weekly.columns)
        return pd.DataFrame(columns=cols)

    first_date = signal_weekly.index.min()
    last_date = signal_weekly.index.max()
    bday_range = pd.bdate_range(start=first_date, end=last_date)
    signal_daily = signal_weekly.reindex(bday_range, method="ffill")
    signal_daily.index.name = "Date"

    # Align columns to full commodity universe (add NaN columns for missing)
    if all_commodities is not None:
        for sym in all_commodities:
            if sym not in signal_daily.columns:
                signal_daily[sym] = float("nan")
        signal_daily = signal_daily[all_commodities]

    logger.info(
        "compute_positioning_factor: %d commodities, %d weekly obs → %d daily rows, %.1f%% non-NaN",
        len(signal_daily.columns),
        len(signal_weekly),
        len(signal_daily),
        signal_daily.notna().mean().mean() * 100,
    )
    result: pd.DataFrame = signal_daily
    return result


def main() -> None:
    """Smoke-test the positioning factor with saved COT data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from commodity_curve_factors.data.cftc_loader import load_cot_data
    from commodity_curve_factors.utils.constants import ALL_COMMODITIES

    cot = load_cot_data()
    factor = compute_positioning_factor(cot, window=156, all_commodities=ALL_COMMODITIES)
    logger.info("Factor shape: %s", factor.shape)
    logger.info("Non-NaN per commodity:\n%s", factor.notna().sum())


if __name__ == "__main__":
    main()
