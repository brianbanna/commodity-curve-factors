"""Offline Parquet reader for WRDS Datastream futures contracts.

Reads pre-downloaded per-commodity contract files from
``data/raw/futures/contracts/{SYMBOL}/all_contracts.parquet``.  No network
calls, no ``wrds`` package dependency — purely local I/O with schema
validation and dtype normalization.

The live download that produces these files lives in
``scripts/probes/wrds_download_all.py`` and is run once by the researcher
outside of CI.

Usage:
    python -m commodity_curve_factors.data.wrds_loader
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from commodity_curve_factors.utils.config import load_config
from commodity_curve_factors.utils.paths import DATA_RAW

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constant — single source of truth for column names
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS: set[str] = {
    "futcode",
    "dsmnem",
    "contrdate",
    "startdate",
    "lasttrddate",
    "sttlmntdate",
    "isocurrcode",
    "ldb",
    "trade_date",
    "open_price",
    "high_price",
    "low_price",
    "settlement",
    "volume",
    "openinterest",
}

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _default_contracts_root() -> Path:
    """Return the default directory that holds per-symbol contract subdirs."""
    return Path(DATA_RAW) / "futures" / "contracts"


def _contract_path(root: Path, symbol: str) -> Path:
    """Return the expected Parquet path for *symbol* under *root*."""
    return root / symbol / "all_contracts.parquet"


def _validate_schema(df: pd.DataFrame, source: Path) -> None:
    """Raise ValueError if *df* is missing any of EXPECTED_COLUMNS.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame loaded from *source*.
    source : Path
        Path used in the error message for easier debugging.

    Raises
    ------
    ValueError
        Lists the names of every missing column and the source path.
        Never prints the full DataFrame.
    """
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        sorted_missing = sorted(missing)
        raise ValueError(
            f"Parquet at {source} is missing required columns: "
            f"{sorted_missing}. "
            f"Present columns: {sorted(df.columns.tolist())}."
        )


def _normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast futcode → int64 and trade_date → datetime64[ns].

    Returns a new DataFrame; does not mutate the input.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as loaded from Parquet (before dtype casts).

    Returns
    -------
    pd.DataFrame
        Copy with corrected dtypes.
    """
    df = df.copy()
    df["futcode"] = df["futcode"].astype("int64")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_contracts(
    symbol: str,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    """Load all contracts for one commodity from disk.

    Parameters
    ----------
    symbol : str
        Commodity ticker, e.g. ``"CL"``, ``"NG"``, ``"GC"``. Must match a
        key in ``configs/universe.yaml``.
    root : Path, optional
        Override for the contracts directory.  Defaults to
        ``paths.DATA_RAW / "futures" / "contracts"``.  Primarily used by
        unit tests which point at a ``tmp_path`` containing the fixture.

    Returns
    -------
    pd.DataFrame
        One row per (contract, trade_date).  Columns match
        :data:`EXPECTED_COLUMNS`.  ``futcode`` is int64, ``trade_date`` is
        ``datetime64[ns]``.  Rows sorted by (futcode, trade_date).

    Raises
    ------
    FileNotFoundError
        If the expected Parquet file does not exist.
    ValueError
        If the loaded DataFrame is missing any of :data:`EXPECTED_COLUMNS`.
    """
    if root is None:
        root = _default_contracts_root()

    path = _contract_path(root, symbol)

    if not path.exists():
        raise FileNotFoundError(
            f"No contracts file found for symbol {symbol!r} at {path}. "
            "Run scripts/probes/wrds_download_all.py to populate the data tree."
        )

    logger.info("Loading contracts for %s from %s", symbol, path)
    df: pd.DataFrame = pd.read_parquet(path)

    _validate_schema(df, path)
    df = _normalize_dtypes(df)
    df = df.sort_values(["futcode", "trade_date"]).reset_index(drop=True)

    logger.info(
        "Loaded %s: %d rows, %d contracts, trade_date %s to %s",
        symbol,
        len(df),
        df["futcode"].nunique(),
        df["trade_date"].min().date(),
        df["trade_date"].max().date(),
    )
    return df


def load_all_contracts(
    *,
    root: Path | None = None,
    symbols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load contracts for every commodity in ``configs/universe.yaml``.

    Parameters
    ----------
    root : Path, optional
        See :func:`load_contracts`.
    symbols : list[str], optional
        Restrict to a subset of commodities.  Defaults to all 13 from
        ``universe.yaml``.  Missing commodities (no Parquet file on disk)
        are logged as warnings and skipped — not raised — to match how the
        bulk download summary handles partial runs.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by commodity symbol.
    """
    universe: dict[str, Any] = load_config("universe")
    all_symbols: list[str] = list(universe["commodities"].keys())

    if symbols is not None:
        all_symbols = [s for s in symbols if s in all_symbols or s not in all_symbols]
        # keep caller-specified order; accept symbols not in universe.yaml so
        # tests with synthetic symbols work — they will simply produce a warning
        all_symbols = symbols

    result: dict[str, pd.DataFrame] = {}
    for sym in all_symbols:
        try:
            result[sym] = load_contracts(sym, root=root)
        except FileNotFoundError:
            logger.warning(
                "Contracts file not found for %s — skipping (run wrds_download_all.py)",
                sym,
            )

    logger.info(
        "load_all_contracts complete: %d/%d symbols loaded",
        len(result),
        len(all_symbols),
    )
    return result


def filter_to_date_range(
    df: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter a contracts DataFrame to ``start <= trade_date <= end``.

    Either bound may be ``None`` (open-ended).  Returns a new DataFrame;
    does not mutate the input.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``trade_date`` column of ``datetime64[ns]`` dtype.
    start : str or pd.Timestamp or None
        Lower bound (inclusive).  Parsed via ``pd.Timestamp`` if a string.
    end : str or pd.Timestamp or None
        Upper bound (inclusive).  Parsed via ``pd.Timestamp`` if a string.

    Returns
    -------
    pd.DataFrame
        Filtered copy of *df*.
    """
    mask = pd.Series(True, index=df.index)

    if start is not None:
        mask &= df["trade_date"] >= pd.Timestamp(start)
    if end is not None:
        mask &= df["trade_date"] <= pd.Timestamp(end)

    return df.loc[mask].copy()


def get_contract_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """One row per contract with its static metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_contracts` — one row per (contract, trade_date).

    Returns
    -------
    pd.DataFrame
        Columns: ``[futcode, dsmnem, contrdate, startdate, lasttrddate,
        sttlmntdate, isocurrcode, ldb, first_trade_date, last_trade_date,
        n_observations]``.  Sorted by ``lasttrddate`` ascending (earliest-
        expiring contract first).
    """
    static_cols = [
        "futcode",
        "dsmnem",
        "contrdate",
        "startdate",
        "lasttrddate",
        "sttlmntdate",
        "isocurrcode",
        "ldb",
    ]

    # Aggregate per-contract stats
    agg = (
        df.groupby("futcode", sort=False)
        .agg(
            first_trade_date=("trade_date", "min"),
            last_trade_date=("trade_date", "max"),
            n_observations=("trade_date", "count"),
        )
        .reset_index()
    )

    # Pull static fields — take first occurrence per contract (they are
    # identical across rows for the same futcode)
    static = df[static_cols].drop_duplicates(subset=["futcode"]).reset_index(drop=True)

    meta = static.merge(agg, on="futcode", how="left")
    meta = meta.sort_values("lasttrddate").reset_index(drop=True)

    return meta


# ---------------------------------------------------------------------------
# CLI entry point (for interactive inspection, not CI)
# ---------------------------------------------------------------------------


def main() -> None:
    """Print a summary of all contract files on disk."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=== WRDS Contracts Loader Summary ===")
    data = load_all_contracts()

    if not data:
        logger.warning(
            "No contract files found under %s. Run scripts/probes/wrds_download_all.py first.",
            _default_contracts_root(),
        )
        return

    for sym, df in sorted(data.items()):
        meta = get_contract_metadata(df)
        logger.info(
            "  %-4s  %6d rows  %3d contracts  %s to %s",
            sym,
            len(df),
            len(meta),
            df["trade_date"].min().date(),
            df["trade_date"].max().date(),
        )


if __name__ == "__main__":
    main()
