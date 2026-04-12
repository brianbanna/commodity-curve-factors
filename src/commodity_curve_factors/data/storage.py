"""Parquet I/O helpers and directory catalog builder.

Provides a thin, convention-enforcing layer over ``pd.read_parquet`` /
``DataFrame.to_parquet``:

- :func:`save_parquet` — sorts DatetimeIndex ascending before writing and
  creates missing parent directories automatically.
- :func:`load_parquet` — raises ``FileNotFoundError`` for absent files and
  logs a WARNING when a loaded DatetimeIndex is not sorted ascending (rather
  than silently re-sorting, which would mask an upstream bug).
- :func:`build_catalog` — scans a directory tree for Parquet files and
  returns a summary DataFrame with shape, date range, and size metadata.

This module does NOT import any data loader (wrds_loader, macro_loader, etc.)
— it is a utility layer only.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to Parquet with a sorted DatetimeIndex guarantee.

    If the DataFrame has a DatetimeIndex, sort it ascending before writing.
    If the index is not a DatetimeIndex, save as-is (no error — some data
    like COT long-format uses a RangeIndex).

    Creates parent directories if they don't exist.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : Path
        Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()

    df.to_parquet(path)
    logger.debug("Saved %d rows to %s", len(df), path)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file and return a DataFrame.

    If the loaded DataFrame has a DatetimeIndex, verify it's sorted
    ascending and log a WARNING if not (don't re-sort — that would mask
    a silent upstream bug).

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df: pd.DataFrame = pd.read_parquet(path)

    if isinstance(df.index, pd.DatetimeIndex) and not df.index.is_monotonic_increasing:
        logger.warning(
            "Loaded Parquet at %s has an unsorted DatetimeIndex — "
            "the upstream writer should call save_parquet to guarantee sort order.",
            path,
        )

    logger.debug("Loaded %d rows from %s", len(df), path)
    return df


def build_catalog(
    root: Path,
    glob_pattern: str = "**/*.parquet",
) -> pd.DataFrame:
    """Scan a directory tree for Parquet files and return a summary catalog.

    Returns
    -------
    pd.DataFrame
        Columns: [path, rows, start_date, end_date, columns, size_mb].
        ``path`` is relative to ``root``. ``start_date`` / ``end_date``
        are the DatetimeIndex min/max (None if the index is not datetime).
        ``size_mb`` is the on-disk file size in megabytes (float, 2 decimal
        places). ``columns`` is the number of columns (int).

    Files that fail to read are included with ``rows=None`` and an
    ``error`` column containing the exception message.

    Parameters
    ----------
    root : Path
        Directory to scan.
    glob_pattern : str
        Glob pattern relative to ``root``. Default ``"**/*.parquet"``.
    """
    root = Path(root)
    records = []
    _CATALOG_COLS = ["path", "rows", "start_date", "end_date", "columns", "size_mb"]

    for file_path in sorted(root.glob(glob_pattern)):
        rel = str(file_path.relative_to(root))
        size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)

        try:
            df = pd.read_parquet(file_path)
            if isinstance(df.index, pd.DatetimeIndex):
                start_date = df.index.min()
                end_date = df.index.max()
            else:
                start_date = None
                end_date = None

            records.append(
                {
                    "path": rel,
                    "rows": len(df),
                    "start_date": start_date,
                    "end_date": end_date,
                    "columns": len(df.columns),
                    "size_mb": size_mb,
                }
            )
        except Exception as exc:
            logger.warning("Could not read Parquet at %s: %s", file_path, exc)
            records.append(
                {
                    "path": rel,
                    "rows": None,
                    "start_date": None,
                    "end_date": None,
                    "columns": None,
                    "size_mb": size_mb,
                    "error": str(exc),
                }
            )

    if not records:
        return pd.DataFrame(columns=_CATALOG_COLS)

    return pd.DataFrame(records)
