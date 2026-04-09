"""Spot-check downloaded futures data against known market facts.

Checks:
    - WTI ~$147 in July 2008
    - WTI negative on April 20, 2020
    - Gold ~$1900 in Aug 2011
    - Gold >$2000 in 2024
    - No unexplained daily returns >20%
    - No missing commodities
    - No excessive NaN gaps

Usage:
    python -m commodity_curve_factors.data.validate
"""

import logging
import sys

import pandas as pd

from commodity_curve_factors.data.futures_loader import load_front_month_data
from commodity_curve_factors.utils.constants import ALL_COMMODITIES

logger = logging.getLogger(__name__)

LARGE_RETURN_THRESHOLD = 0.20
MAX_CONSECUTIVE_NANS = 10


def _check_price_range(df: pd.DataFrame, start: str, end: str, col: str = "Close") -> float | None:
    """Return the max price in a date range, or None if no data."""
    mask = (df.index >= start) & (df.index <= end)
    subset = df.loc[mask, col]
    if subset.empty:
        return None
    return float(subset.max())


def _check_min_price(df: pd.DataFrame, start: str, end: str, col: str = "Close") -> float | None:
    """Return the min price in a date range, or None if no data."""
    mask = (df.index >= start) & (df.index <= end)
    subset = df.loc[mask, col]
    if subset.empty:
        return None
    return float(subset.min())


def validate_spot_checks(data: dict[str, pd.DataFrame]) -> list[str]:
    """Run known-fact spot checks against downloaded data.

    Returns
    -------
    list[str]
        List of failure messages. Empty list means all checks passed.
    """
    failures = []

    # WTI July 2008: ~$147
    if "CL" in data:
        high = _check_price_range(data["CL"], "2008-07-01", "2008-07-15")
        if high is None:
            failures.append("CL: no data for July 2008")
        elif not (130 <= high <= 160):
            failures.append(f"CL July 2008 high={high:.2f}, expected ~147")
        else:
            logger.info("PASS  CL July 2008 high=%.2f", high)

    # WTI April 20, 2020: negative
    if "CL" in data:
        low = _check_min_price(data["CL"], "2020-04-20", "2020-04-20")
        if low is None:
            failures.append("CL: no data for April 20, 2020")
        elif low >= 0:
            failures.append(f"CL April 20 2020 min={low:.2f}, expected negative")
        else:
            logger.info("PASS  CL April 20 2020 min=%.2f (negative)", low)

    # Gold Aug 2011: ~$1900
    if "GC" in data:
        high = _check_price_range(data["GC"], "2011-08-01", "2011-08-31")
        if high is None:
            failures.append("GC: no data for Aug 2011")
        elif not (1700 <= high <= 2000):
            failures.append(f"GC Aug 2011 high={high:.2f}, expected ~1900")
        else:
            logger.info("PASS  GC Aug 2011 high=%.2f", high)

    # Gold 2024: >$2000
    if "GC" in data:
        high = _check_price_range(data["GC"], "2024-01-01", "2024-12-31")
        if high is None:
            failures.append("GC: no data for 2024")
        elif high < 2000:
            failures.append(f"GC 2024 high={high:.2f}, expected >2000")
        else:
            logger.info("PASS  GC 2024 high=%.2f", high)

    return failures


def validate_completeness(data: dict[str, pd.DataFrame]) -> list[str]:
    """Check that all commodities are present and have reasonable coverage."""
    failures = []

    for sym in ALL_COMMODITIES:
        if sym not in data:
            failures.append(f"{sym}: missing from downloaded data")
            continue

        df = data[sym]
        if len(df) < 4000:
            failures.append(f"{sym}: only {len(df)} rows (expected ~5000 for 2005-2024)")

        # Check for consecutive NaN gaps in Close
        if "Close" in df.columns:
            is_nan = df["Close"].isna()
            if is_nan.any():
                max_gap = is_nan.astype(int).groupby((~is_nan).cumsum()).sum().max()
                if max_gap > MAX_CONSECUTIVE_NANS:
                    failures.append(f"{sym}: {max_gap} consecutive NaN closes")

        logger.info(
            "PASS  %s: %d rows, %s to %s", sym, len(df), df.index[0].date(), df.index[-1].date()
        )

    return failures


def validate_returns(data: dict[str, pd.DataFrame]) -> list[str]:
    """Flag daily returns exceeding the threshold for manual review."""
    warnings = []

    for sym, df in sorted(data.items()):
        if "Close" not in df.columns:
            continue
        returns = df["Close"].pct_change().abs()
        large = returns[returns > LARGE_RETURN_THRESHOLD]
        for date, ret in large.items():
            date_str = str(date)[:10]
            msg = f"{sym} {date_str}: {ret:.1%} daily return"
            warnings.append(msg)

    return warnings


def main() -> None:
    """Run all validation checks."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    data = load_front_month_data()
    if not data:
        print("No data found. Run `make data` first.")
        sys.exit(1)

    all_failures = []

    print("\n=== Spot Checks ===")
    failures = validate_spot_checks(data)
    all_failures.extend(failures)

    print("\n=== Completeness ===")
    failures = validate_completeness(data)
    all_failures.extend(failures)

    print("\n=== Large Daily Returns (>20%) ===")
    warnings = validate_returns(data)
    for w in warnings:
        logger.warning("REVIEW  %s", w)

    print(f"\n{'=' * 50}")
    if all_failures:
        print(f"  FAILURES: {len(all_failures)}")
        for f in all_failures:
            print(f"    FAIL  {f}")
        sys.exit(1)
    else:
        print("  All validation checks passed.")
        print(f"  {len(warnings)} large returns flagged for review (expected).")
    print()


if __name__ == "__main__":
    main()
