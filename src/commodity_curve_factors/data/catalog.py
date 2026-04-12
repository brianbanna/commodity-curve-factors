"""Print a summary catalog of all downloaded data files.

Usage:
    python -m commodity_curve_factors.data.catalog
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from commodity_curve_factors.data.storage import build_catalog
from commodity_curve_factors.utils.paths import DATA_RAW

logger = logging.getLogger(__name__)


def catalog_directory(directory: str, glob_pattern: str = "**/*.parquet") -> pd.DataFrame:
    """Build a catalog DataFrame for all Parquet files under a directory.

    Thin wrapper around :func:`~commodity_curve_factors.data.storage.build_catalog`
    that accepts a string path for backward compatibility.

    Parameters
    ----------
    directory : str
        Root directory to scan.
    glob_pattern : str
        Glob pattern for matching files.

    Returns
    -------
    pd.DataFrame
        Columns: path, rows, start_date, end_date, columns, size_mb.
    """
    result: pd.DataFrame = build_catalog(Path(directory), glob_pattern=glob_pattern)
    return result


def print_catalog() -> None:
    """Print the full data catalog to stdout."""
    cat = build_catalog(DATA_RAW)

    if cat.empty:
        print("No data files found. Run `make data` first.")
        return

    print(f"\n{'=' * 72}")
    print(f"  DATA CATALOG  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 72}")
    print(f"  {'File':<35s} {'Rows':>6s}  {'Start':>12s}  {'End':>12s}  {'MB':>7s}")
    print(f"  {'-' * 35} {'-' * 6}  {'-' * 12}  {'-' * 12}  {'-' * 7}")

    for _, row in cat.iterrows():
        start = (
            str(row["start_date"].date())
            if pd.notna(row["start_date"]) and row["start_date"] is not None
            else "—"
        )
        end = (
            str(row["end_date"].date())
            if pd.notna(row["end_date"]) and row["end_date"] is not None
            else "—"
        )
        rows = str(int(row["rows"])) if pd.notna(row["rows"]) and row["rows"] is not None else "ERR"
        print(f"  {row['path']:<35s} {rows:>6s}  {start:>12s}  {end:>12s}  {row['size_mb']:>7.2f}")

    print(f"\n  Total files: {len(cat)}")
    valid = cat[cat["rows"].notna()]
    total_rows = valid["rows"].sum()
    total_mb = cat["size_mb"].sum()
    print(f"  Total rows:  {int(total_rows):,}")
    print(f"  Total size:  {total_mb:.2f} MB")
    print()


def main() -> None:
    """Entry point for make catalog."""
    logging.basicConfig(level=logging.WARNING)
    print_catalog()


if __name__ == "__main__":
    main()
