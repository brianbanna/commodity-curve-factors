"""Print a summary catalog of all downloaded data files.

Usage:
    python -m commodity_curve_factors.data.catalog
"""
import logging
from datetime import datetime

import pandas as pd

from commodity_curve_factors.utils.paths import DATA_RAW

logger = logging.getLogger(__name__)


def catalog_directory(directory: str, glob_pattern: str = "**/*.parquet") -> pd.DataFrame:
    """Build a catalog DataFrame for all Parquet files under a directory.

    Parameters
    ----------
    directory : str
        Root directory to scan.
    glob_pattern : str
        Glob pattern for matching files.

    Returns
    -------
    pd.DataFrame
        Columns: file, rows, columns, start, end, size_kb.
    """
    from pathlib import Path

    root = Path(directory)
    records = []

    for path in sorted(root.glob(glob_pattern)):
        try:
            df = pd.read_parquet(path)
            start = df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None
            end = df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None
            records.append({
                "file": str(path.relative_to(root)),
                "rows": len(df),
                "columns": len(df.columns),
                "start": start.date() if start else None,
                "end": end.date() if end else None,
                "size_kb": round(path.stat().st_size / 1024, 1),
            })
        except Exception as exc:
            records.append({
                "file": str(path.relative_to(root)),
                "rows": None,
                "columns": None,
                "start": None,
                "end": None,
                "size_kb": round(path.stat().st_size / 1024, 1),
                "error": str(exc),
            })

    return pd.DataFrame(records)


def print_catalog() -> None:
    """Print the full data catalog to stdout."""
    cat = catalog_directory(str(DATA_RAW))

    if cat.empty:
        print("No data files found. Run `make data` first.")
        return

    print(f"\n{'='*72}")
    print(f"  DATA CATALOG  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*72}")
    print(f"  {'File':<35s} {'Rows':>6s}  {'Start':>12s}  {'End':>12s}  {'KB':>7s}")
    print(f"  {'-'*35} {'-'*6}  {'-'*12}  {'-'*12}  {'-'*7}")

    for _, row in cat.iterrows():
        start = str(row["start"]) if row["start"] else "—"
        end = str(row["end"]) if row["end"] else "—"
        rows = str(row["rows"]) if row["rows"] else "ERR"
        print(f"  {row['file']:<35s} {rows:>6s}  {start:>12s}  {end:>12s}  {row['size_kb']:>7.1f}")

    print(f"\n  Total files: {len(cat)}")
    total_rows = cat["rows"].sum()
    total_kb = cat["size_kb"].sum()
    print(f"  Total rows:  {int(total_rows):,}")
    print(f"  Total size:  {total_kb:.0f} KB ({total_kb/1024:.1f} MB)")
    print()


def main() -> None:
    """Entry point for make catalog."""
    logging.basicConfig(level=logging.WARNING)
    print_catalog()


if __name__ == "__main__":
    main()
