"""Roll calendar utilities for futures contracts.

Determines which contracts are "active" on a given trade date based on a
per-commodity ``roll_days_before_expiry`` offset, and identifies the front
contract on each day.

Roll day calculation uses CALENDAR days (not business days). The offset values
in ``curve.yaml`` are small integers (5–15), and the difference between calendar
days and business days at that scale is at most 2–3 days — negligible for
strategy purposes. Calendar days are used for simplicity and auditability.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def active_contracts_on_day(
    contracts: pd.DataFrame,
    trade_date: pd.Timestamp,
    roll_days_before_expiry: int,
) -> pd.DataFrame:
    """Subset of ``contracts`` that are tradeable on ``trade_date``.

    A contract is considered active on ``trade_date`` if:
      - ``startdate <= trade_date``
      - ``trade_date <= lasttrddate - roll_days_before_expiry`` (calendar days)
      - the row for ``trade_date`` exists AND ``settlement`` is not NaN.

    Parameters
    ----------
    contracts : pd.DataFrame
        All rows for one commodity (many contracts × many trade_dates).
        Must have columns: ``startdate``, ``lasttrddate``, ``trade_date``,
        ``settlement``.  ``startdate`` and ``lasttrddate`` may be
        ``datetime.date`` objects or ``datetime64[ns]``.
    trade_date : pd.Timestamp
        The date of interest.
    roll_days_before_expiry : int
        Number of calendar days before ``lasttrddate`` to stop treating a
        contract as active.

    Returns
    -------
    pd.DataFrame
        Rows of ``contracts`` where the contract is active on ``trade_date``.
        One row per active contract (the row for that specific trade_date).
    """
    day_rows = contracts[contracts["trade_date"] == trade_date].copy()

    if day_rows.empty:
        return day_rows

    # Normalise date columns to Timestamps for comparison
    def _to_ts(col: pd.Series) -> pd.Series:
        return pd.to_datetime(col)

    start_ts = _to_ts(day_rows["startdate"])
    last_ts = _to_ts(day_rows["lasttrddate"])
    roll_deadline = last_ts - pd.Timedelta(days=roll_days_before_expiry)

    mask = (start_ts <= trade_date) & (trade_date <= roll_deadline) & day_rows["settlement"].notna()

    return day_rows[mask.values].copy()


def get_front_contract(
    contracts: pd.DataFrame,
    trade_date: pd.Timestamp,
    roll_days_before_expiry: int,
) -> pd.Series | None:
    """Return the single-row Series for the nearest-to-expiry active contract.

    Parameters
    ----------
    contracts : pd.DataFrame
        All rows for one commodity.
    trade_date : pd.Timestamp
        The date of interest.
    roll_days_before_expiry : int
        Calendar-day roll offset.

    Returns
    -------
    pd.Series or None
        The row of the front contract on ``trade_date``, or None if there
        are no active contracts.
    """
    active = active_contracts_on_day(contracts, trade_date, roll_days_before_expiry)

    if active.empty:
        return None

    # Front = smallest lasttrddate among active contracts
    last_ts = pd.to_datetime(active["lasttrddate"])
    front_idx = last_ts.idxmin()
    row: pd.Series = active.loc[front_idx]  # type: ignore[assignment]
    return row


def build_roll_schedule(
    contracts: pd.DataFrame,
    roll_days_before_expiry: int,
) -> pd.DataFrame:
    """Build a roll schedule: one row per trade_date with front-contract info.

    Parameters
    ----------
    contracts : pd.DataFrame
        All rows for one commodity.
    roll_days_before_expiry : int
        Calendar-day roll offset.

    Returns
    -------
    pd.DataFrame
        Columns: ``[trade_date, front_futcode, front_dsmnem, days_to_expiry,
        settlement]``.  Sorted by ``trade_date`` ascending.  Rows where no
        active contract exists have NaN in the contract columns.
    """
    trade_dates = contracts["trade_date"].sort_values().unique()
    records: list[dict[str, Any]] = []

    for td in trade_dates:
        ts = pd.Timestamp(td)
        front = get_front_contract(contracts, ts, roll_days_before_expiry)

        if front is None:
            records.append(
                {
                    "trade_date": ts,
                    "front_futcode": None,
                    "front_dsmnem": None,
                    "days_to_expiry": None,
                    "settlement": None,
                }
            )
        else:
            last_td = pd.Timestamp(front["lasttrddate"])
            days_to_expiry = (last_td - ts).days
            records.append(
                {
                    "trade_date": ts,
                    "front_futcode": int(front["futcode"]),
                    "front_dsmnem": str(front["dsmnem"]),
                    "days_to_expiry": int(days_to_expiry),
                    "settlement": float(front["settlement"]),
                }
            )

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("trade_date").reset_index(drop=True)

    logger.debug(
        "build_roll_schedule: %d trade_dates, %d rows with front contract",
        len(trade_dates),
        result["front_futcode"].notna().sum() if not result.empty else 0,
    )
    return result
