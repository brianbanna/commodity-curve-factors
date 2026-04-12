"""Download weekly Commitments of Traders (COT) disaggregated reports from cftc.gov.

The CFTC publishes one zip per calendar year of disaggregated futures-only
Commitments of Traders history at::

    https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip

Each zip contains a single comma-delimited text file named ``f_year.txt``
(quoted headers, ``"."`` used as the missing-value marker) with ~190 columns
covering every contract that has ever reported under the disaggregated
regime (June 2006 onward).

For this project we extract managed-money long/short positioning for the 13
commodities listed in ``configs/universe.yaml`` under the ``cftc_code``
field.  Each row in the source file corresponds to one
(contract, weekly report date) pair.  We keep:

=================================  =========================================
Source column                      Renamed to
=================================  =========================================
``Market_and_Exchange_Names``      (used only to filter — not kept)
``CFTC_Contract_Market_Code``      ``commodity`` (after mapping to our ticker)
``Report_Date_as_YYYY-MM-DD``      ``report_date``
``M_Money_Positions_Long_All``     ``mm_long``
``M_Money_Positions_Short_All``    ``mm_short``
``Open_Interest_All``              ``open_interest``
=================================  =========================================

plus a derived ``mm_net = mm_long - mm_short``.

COT reports are an "as of Tuesday close" snapshot released the following
Friday 15:30 ET.  Downstream factor code must key signals by the
``release_date`` column (added by :func:`lag_to_release_date`), *not*
``report_date``, to avoid lookahead bias.

Usage:
    python -m commodity_curve_factors.data.cftc_loader
"""

import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from commodity_curve_factors.utils.config import load_config
from commodity_curve_factors.utils.paths import DATA_CACHE, DATA_RAW

logger = logging.getLogger(__name__)

_CFTC_HISTORY_BASE = "https://www.cftc.gov/files/dea/history"
_REQUEST_TIMEOUT_SECONDS = 60

# Exact column names in the real ``f_year.txt`` file (verified against the
# 2023 disaggregated futures-only download).  Keeping a single source of
# truth so the parser fails loudly on schema drift.
_COL_NAME = "Market_and_Exchange_Names"
_COL_CODE = "CFTC_Contract_Market_Code"
_COL_DATE = "Report_Date_as_YYYY-MM-DD"
# Pre-2017 files used a different date column; we rename on load.
_COL_DATE_LEGACY = "As_of_Date_In_Form_YYMMDD"
_COL_OI = "Open_Interest_All"
_COL_MM_LONG = "M_Money_Positions_Long_All"
_COL_MM_SHORT = "M_Money_Positions_Short_All"

_REQUIRED_COLUMNS: tuple[str, ...] = (
    _COL_NAME,
    _COL_CODE,
    _COL_DATE,
    _COL_OI,
    _COL_MM_LONG,
    _COL_MM_SHORT,
)

_WEEKDAY_INDEX: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def download_cot_zip(year: int, *, use_cache: bool = True) -> pd.DataFrame | None:
    """Download one year of disaggregated futures-only COT data.

    Parameters
    ----------
    year : int
        Calendar year of the COT history file (e.g. ``2023``).  The
        disaggregated report starts in June 2006; earlier years are not
        available and will return ``None``.
    use_cache : bool
        If True (default), a previously downloaded zip at
        ``DATA_CACHE / "cftc_{year}.zip"`` is reused.  Set to False to
        force a re-download.

    Returns
    -------
    pd.DataFrame or None
        Raw DataFrame with the column names as published by the CFTC
        (see :data:`_REQUIRED_COLUMNS` for the subset we rely on).
        Returns ``None`` on any network failure, non-200 response,
        missing zip member, or parse failure — the caller is responsible
        for logging the skip and continuing.
    """
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_CACHE / f"cftc_{year}.zip"

    if use_cache and cache_path.exists():
        logger.info("Using cached COT zip for %d at %s", year, cache_path)
        zip_bytes = cache_path.read_bytes()
    else:
        url = f"{_CFTC_HISTORY_BASE}/fut_disagg_txt_{year}.zip"
        logger.info("Downloading COT zip for %d from %s", year, url)
        try:
            response = requests.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            # Never include ``exc`` itself in the log message: the default
            # stringification on ConnectionError / Timeout embeds the full
            # URL, which is safe here (no secrets) but we match the
            # sanitization pattern used in ``inventory_loader`` for
            # consistency across the data/ package.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            logger.warning(
                "CFTC download failed for %d (%s, status=%s)",
                year,
                type(exc).__name__,
                status,
            )
            return None

        if response.status_code != 200:
            logger.warning("CFTC download failed for %d: HTTP %d", year, response.status_code)
            return None

        zip_bytes = response.content
        try:
            cache_path.write_bytes(zip_bytes)
        except OSError:
            logger.exception("Failed to cache COT zip for %d at %s", year, cache_path)

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            txt_members = [name for name in zf.namelist() if name.endswith(".txt")]
            if not txt_members:
                logger.error("CFTC zip for %d contains no .txt files: %s", year, zf.namelist())
                return None
            if len(txt_members) > 1:
                logger.warning(
                    "CFTC zip for %d contains multiple .txt files, using first: %s",
                    year,
                    txt_members,
                )
            member = txt_members[0]
            with zf.open(member) as fh:
                df = pd.read_csv(fh, low_memory=False)
    except (zipfile.BadZipFile, ValueError, pd.errors.ParserError) as exc:
        logger.warning("Failed to parse CFTC zip for %d (%s)", year, type(exc).__name__)
        return None

    logger.info("CFTC %d: %d raw rows, %d columns", year, len(df), df.shape[1])
    return df


def parse_cot_csv(
    raw: pd.DataFrame,
    commodity_codes: dict[str, str],
) -> pd.DataFrame:
    """Extract rows for our 13 commodities from a raw COT DataFrame.

    Parameters
    ----------
    raw : pd.DataFrame
        Output of :func:`download_cot_zip` (or a fixture with the same
        columns).
    commodity_codes : dict[str, str]
        Mapping from our internal symbol (e.g. ``"CL"``) to the CFTC
        contract market code string as it appears in the COT file (e.g.
        ``"067651"``).  Read from ``configs/universe.yaml`` in the caller.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns
        ``[commodity, report_date, mm_long, mm_short, mm_net,
        open_interest]``, sorted by ``(commodity, report_date)``.
        Rows whose ``CFTC_Contract_Market_Code`` is not in
        ``commodity_codes`` are dropped.  ``mm_long``, ``mm_short`` and
        ``open_interest`` are float64 (``"."`` values in the source become
        ``NaN``).  ``mm_net = mm_long - mm_short``.

    Raises
    ------
    ValueError
        If any of the required columns (:data:`_REQUIRED_COLUMNS`) is
        missing from ``raw``.  The error message lists every missing
        column so schema drift is easy to diagnose.
    """
    # Pre-2017 CFTC files use a different date column name and YYMMDD format.
    # Normalize to the modern schema so downstream code sees a single name.
    if _COL_DATE_LEGACY in raw.columns and _COL_DATE not in raw.columns:
        raw = raw.copy()
        raw[_COL_DATE] = pd.to_datetime(
            raw[_COL_DATE_LEGACY].astype(str), format="%y%m%d"
        ).dt.strftime("%Y-%m-%d")
        logger.debug("Renamed legacy date column %s → %s", _COL_DATE_LEGACY, _COL_DATE)

    missing = [col for col in _REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(
            f"COT DataFrame is missing required columns: {missing}. "
            f"Present columns: {sorted(raw.columns.tolist())[:20]}..."
        )

    # Invert code->symbol for cheap lookup.
    code_to_symbol: dict[str, str] = {
        str(code).strip(): symbol for symbol, code in commodity_codes.items()
    }

    # Select only the columns we need, then filter to our universe.
    subset = raw[list(_REQUIRED_COLUMNS)].copy()
    subset[_COL_CODE] = subset[_COL_CODE].astype(str).str.strip()
    subset = subset[subset[_COL_CODE].isin(code_to_symbol)]

    if subset.empty:
        logger.warning(
            "parse_cot_csv: no rows matched any of the %d universe codes",
            len(code_to_symbol),
        )
        return pd.DataFrame(
            columns=[
                "commodity",
                "report_date",
                "mm_long",
                "mm_short",
                "mm_net",
                "open_interest",
            ]
        )

    # CFTC uses "." as a missing-value marker; ``to_numeric(errors="coerce")``
    # turns that (and any other non-numeric) into NaN.
    mm_long = pd.to_numeric(subset[_COL_MM_LONG], errors="coerce")
    mm_short = pd.to_numeric(subset[_COL_MM_SHORT], errors="coerce")
    open_interest = pd.to_numeric(subset[_COL_OI], errors="coerce")

    out = pd.DataFrame(
        {
            "commodity": subset[_COL_CODE].map(code_to_symbol),
            "report_date": pd.to_datetime(subset[_COL_DATE], format="%Y-%m-%d"),
            "mm_long": mm_long.astype(float),
            "mm_short": mm_short.astype(float),
            "mm_net": (mm_long - mm_short).astype(float),
            "open_interest": open_interest.astype(float),
        }
    )

    out = out.sort_values(["commodity", "report_date"]).reset_index(drop=True)
    return out


def download_cot_history(
    start_year: int,
    end_year: int,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download and parse COT data for every year in ``[start_year, end_year]``.

    Iterates :func:`download_cot_zip` over the inclusive year range and
    calls :func:`parse_cot_csv` on each successful download.  Years that
    return ``None`` (404, network failure, parse failure) are logged at
    WARNING level and skipped — a single bad year never aborts the run.

    Parameters
    ----------
    start_year, end_year : int
        Inclusive year bounds.
    use_cache : bool
        Passed through to :func:`download_cot_zip`.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with the schema documented on
        :func:`parse_cot_csv`, sorted by ``(commodity, report_date)``.
        Empty DataFrame (same schema) if every year failed.
    """
    universe: dict[str, Any] = load_config("universe")
    commodity_codes: dict[str, str] = {
        sym: str(meta["cftc_code"])
        for sym, meta in universe["commodities"].items()
        if meta.get("cftc_code") is not None
    }

    if not commodity_codes:
        logger.error("No cftc_code entries in universe.yaml — cannot parse COT history")
        return pd.DataFrame(
            columns=[
                "commodity",
                "report_date",
                "mm_long",
                "mm_short",
                "mm_net",
                "open_interest",
            ]
        )

    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        raw = download_cot_zip(year, use_cache=use_cache)
        if raw is None:
            logger.warning("Skipping COT year %d — no data available", year)
            continue
        parsed = parse_cot_csv(raw, commodity_codes)
        frames.append(parsed)
        logger.info("COT %d: %d rows after filtering to universe", year, len(parsed))

    if not frames:
        logger.warning("download_cot_history: no years succeeded for %d-%d", start_year, end_year)
        return pd.DataFrame(
            columns=[
                "commodity",
                "report_date",
                "mm_long",
                "mm_short",
                "mm_net",
                "open_interest",
            ]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["commodity", "report_date"]).reset_index(drop=True)
    logger.info(
        "COT history %d-%d: %d total rows, %d commodities",
        start_year,
        end_year,
        len(combined),
        combined["commodity"].nunique(),
    )
    return combined


def compute_net_speculative(cot: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format COT data to a wide ``report_date x commodity`` grid.

    Parameters
    ----------
    cot : pd.DataFrame
        Long-format DataFrame as returned by :func:`parse_cot_csv` or
        :func:`download_cot_history`.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with a ``DatetimeIndex`` of report dates and
        one column per commodity symbol, values = managed-money net
        positioning (``mm_net``, contracts).  Used as the raw input to the
        Phase 3 positioning factor.
    """
    dupes = cot.duplicated(subset=["report_date", "commodity"], keep=False)
    if dupes.any():
        n_dupes = int(dupes.sum())
        logger.warning(
            "compute_net_speculative: %d duplicate (report_date, commodity) rows; "
            "keeping last. Check universe.yaml for duplicate cftc_code mappings.",
            n_dupes,
        )

    wide = cot.pivot_table(
        index="report_date",
        columns="commodity",
        values="mm_net",
        aggfunc="last",
    )
    wide = wide.sort_index()
    wide.columns.name = None
    wide.index.name = "report_date"
    return wide


def _next_release_date(obs_date: pd.Timestamp, release_dow: int) -> pd.Timestamp:
    """Return the first date strictly after ``obs_date`` with weekday ``release_dow``.

    "Strictly after" (>= obs_date + 1 day) enforces that a report date
    that happens to fall on the release day rolls to the *next* week —
    the COT data cannot be known the same day it is dated.
    """
    if pd.isna(obs_date):
        return pd.NaT
    next_day = obs_date + pd.Timedelta(days=1)
    days_ahead = (release_dow - next_day.weekday()) % 7
    return next_day + pd.Timedelta(days=days_ahead)


def lag_to_release_date(
    cot: pd.DataFrame,
    *,
    release_weekday: str = "friday",
) -> pd.DataFrame:
    """Add a ``release_date`` column for no-lookahead signal alignment.

    Each COT report has an as-of date (Tuesday close) but is not released
    until the following Friday 15:30 ET in the current era.  This helper
    adds a ``release_date`` column holding the first ``release_weekday``
    strictly after ``report_date``.  Downstream factor code must key
    signals on ``release_date`` (not ``report_date``) to avoid lookahead.

    The rule is strict weekday-based; exchange holidays are not applied.
    That matches how the EIA/FRED loaders in this package handle release
    lag — the one-day blur is tolerable for weekly data and avoids
    pulling in a calendar dependency.

    Parameters
    ----------
    cot : pd.DataFrame
        Long-format COT DataFrame with a ``report_date`` column of
        ``datetime64[ns]`` dtype (i.e. output of :func:`parse_cot_csv`).
    release_weekday : str
        Lowercase weekday name, one of ``"monday"`` .. ``"sunday"``.
        Defaults to ``"friday"`` (the current CFTC release day).

    Returns
    -------
    pd.DataFrame
        Copy of *cot* with a new ``release_date`` column of
        ``datetime64[ns]`` dtype, sorted by ``release_date`` ascending.
    """
    key = release_weekday.lower()
    if key not in _WEEKDAY_INDEX:
        raise ValueError(
            f"release_weekday must be one of {sorted(_WEEKDAY_INDEX)}, got {release_weekday!r}"
        )
    release_dow = _WEEKDAY_INDEX[key]

    out = cot.copy()
    if out.empty:
        out["release_date"] = pd.Series(dtype="datetime64[ns]")
        return out

    out["release_date"] = pd.DatetimeIndex(
        [_next_release_date(ts, release_dow) for ts in out["report_date"]]
    )
    out = out.sort_values("release_date", kind="stable", na_position="last").reset_index(drop=True)
    return out


def save_cot_data(cot: pd.DataFrame, *, out_dir: Path | None = None) -> None:
    """Save the long-format COT DataFrame to Parquet.

    Parameters
    ----------
    cot : pd.DataFrame
        Long-format DataFrame as returned by :func:`download_cot_history`.
    out_dir : Path, optional
        Override for the output directory.  Defaults to
        ``paths.DATA_RAW / "cftc"``.  Primarily used by unit tests
        pointing at a ``tmp_path``.
    """
    if out_dir is None:
        out_dir = DATA_RAW / "cftc"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "cot_history.parquet"
    cot.to_parquet(path)

    if len(cot) > 0:
        logger.info(
            "Saved COT history → %s (%d rows, %s to %s)",
            path,
            len(cot),
            cot["report_date"].min().date(),
            cot["report_date"].max().date(),
        )
    else:
        logger.info("Saved empty COT history → %s", path)


def load_cot_data(*, in_dir: Path | None = None) -> pd.DataFrame:
    """Load the saved long-format COT DataFrame from Parquet.

    Parameters
    ----------
    in_dir : Path, optional
        Override for the input directory.  Defaults to
        ``paths.DATA_RAW / "cftc"``.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame.  Column dtypes match what was saved.

    Raises
    ------
    FileNotFoundError
        If ``{in_dir}/cot_history.parquet`` does not exist.  Mirrors
        :func:`wrds_loader.load_contracts` fail-loud semantics.
    """
    if in_dir is None:
        in_dir = DATA_RAW / "cftc"

    path = in_dir / "cot_history.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No COT history file found at {path}. "
            "Run python -m commodity_curve_factors.data.cftc_loader to populate."
        )

    df: pd.DataFrame = pd.read_parquet(path)
    logger.info("Loaded COT history from %s: %d rows", path, len(df))
    return df


def main() -> None:
    """Download 2010-2024 COT history, save to Parquet, log summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=== CFTC COT History Download ===")
    cot = download_cot_history(2010, 2024, use_cache=True)

    if cot.empty:
        logger.error("No COT data downloaded — check network or CFTC availability")
        return

    save_cot_data(cot)

    logger.info("=== Download Summary ===")
    counts = cot.groupby("commodity").size().sort_index()
    date_range = cot.groupby("commodity")["report_date"].agg(["min", "max"]).sort_index()
    for sym in counts.index:
        logger.info(
            "  %-4s  %5d rows  %s → %s",
            sym,
            int(counts.loc[sym]),
            date_range.loc[sym, "min"].date(),
            date_range.loc[sym, "max"].date(),
        )

    # Sanity: warn if mm_net is NaN or zero across a commodity — likely
    # a mis-mapped cftc_code in universe.yaml.
    nan_rates = cot.groupby("commodity")["mm_net"].apply(lambda s: float(np.isnan(s).mean()))
    suspicious = nan_rates[nan_rates > 0.5]
    if len(suspicious) > 0:
        logger.warning(
            "Commodities with >50%% NaN in mm_net (check cftc_code mapping): %s",
            suspicious.to_dict(),
        )


if __name__ == "__main__":
    main()
