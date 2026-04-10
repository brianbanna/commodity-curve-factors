"""Download weekly petroleum and natural gas inventory data from the EIA API.

Four series are fetched, one per energy commodity:

===================  =================================  ==================
Key                  EIA v2 series ID                   Commodity (futures)
===================  =================================  ==================
crude_stocks         WCESTUS1                           CL (WTI crude oil)
natural_gas_storage  NW2_EPG0_SWO_R48_BCF               NG (Henry Hub gas)
gasoline_stocks      WGFSTUS1                           RB (RBOB gasoline)
distillate_stocks    WDISTUS1                           HO (heating oil)
===================  =================================  ==================

The raw weekly data is stored verbatim as Parquet; the "inventory surprise"
factor ``(actual_change - seasonal_expected_change) / historical_std`` is
computed in Phase 3 of the pipeline. The helper :func:`align_to_daily`
reindexes a weekly series onto business days without lookahead: each weekly
observation becomes visible only from the next release-day occurrence
*after* its period date.

Usage:
    python -m commodity_curve_factors.data.inventory_loader
"""

import logging
import os
from typing import Any

import pandas as pd
import requests

from commodity_curve_factors.utils.config import load_config
from commodity_curve_factors.utils.paths import DATA_CACHE, DATA_RAW

logger = logging.getLogger(__name__)

# API endpoint path (after /v2/) for each series. Not part of the YAML config
# because it is a property of the EIA API layout, not a business parameter.
EIA_ROUTES: dict[str, str] = {
    "crude_stocks": "petroleum/stoc/wstk",
    "natural_gas_storage": "natural-gas/stor/wkly",
    "gasoline_stocks": "petroleum/stoc/wstk",
    "distillate_stocks": "petroleum/stoc/wstk",
}

_EIA_API_BASE = "https://api.eia.gov/v2"
_REQUEST_TIMEOUT_SECONDS = 30

# USDA NASS QuickStats ---------------------------------------------------------

USDA_NASS_BASE_URL: str = "https://quickstats.nass.usda.gov/api/api_GET/"

# Maps commodity symbol to the (commodity_desc, short_desc) pair for NASS
# filtering. ``short_desc`` is the specific stocks measure — using exact
# match avoids ambiguity between e.g. total stocks and on-farm-only stocks.
USDA_STOCK_SERIES: dict[str, dict[str, str]] = {
    "ZC": {
        "commodity_desc": "CORN",
        "short_desc": "CORN, GRAIN - STOCKS, MEASURED IN BU",
    },
    "ZS": {
        "commodity_desc": "SOYBEANS",
        "short_desc": "SOYBEANS - STOCKS, MEASURED IN BU",
    },
    "ZW": {
        "commodity_desc": "WHEAT",
        "short_desc": "WHEAT - STOCKS, MEASURED IN BU",
    },
}

_WEEKDAY_INDEX: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def download_eia_series(
    series_id: str,
    route: str,
    start: str,
    end: str,
    api_key: str,
) -> pd.DataFrame | None:
    """Download a single weekly series from the EIA v2 API.

    Parameters
    ----------
    series_id : str
        EIA v2 series identifier (e.g. ``"WCESTUS1"``). This is the value
        passed straight to the ``facets[series][]`` query parameter; legacy
        dotted ids like ``"PET.WCESTUS1.W"`` are not supported.
    route : str
        URL path segment between ``/v2/`` and ``/data/`` (e.g.
        ``"petroleum/stoc/wstk"``). Look up from :data:`EIA_ROUTES`.
    start, end : str
        Date range in ``YYYY-MM-DD`` format.
    api_key : str
        EIA API key. Never logged.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with a ``DatetimeIndex`` (period end dates) and a single
        ``value`` column of ``float`` dtype, sorted ascending. Returns
        ``None`` on network/API failure. An empty successful response is
        also returned as ``None`` with a warning so the orchestrator can
        skip cleanly.
    """
    logger.info("Downloading EIA series %s from %s", series_id, route)

    url = f"{_EIA_API_BASE}/{route}/data/"
    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": series_id,
        "start": start,
        "end": end,
    }

    try:
        response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        # Never include ``exc`` itself in the log message: the default
        # stringification on ConnectionError / Timeout embeds the full URL,
        # which contains ``api_key=...`` in the query string.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        logger.warning(
            "EIA download failed for %s (%s, status=%s)",
            series_id,
            type(exc).__name__,
            status,
        )
        return None

    if response.status_code != 200:
        logger.warning("EIA download failed for %s: HTTP %d", series_id, response.status_code)
        return None

    try:
        payload: dict[str, Any] = response.json()
    except ValueError as exc:
        logger.warning("EIA response for %s was not valid JSON: %s", series_id, exc)
        return None

    # EIA v2 sometimes returns HTTP 200 with an error body (e.g. invalid API
    # key) and no ``response.data`` field. Surface that directly.
    if "error" in payload:
        logger.warning("EIA API error for %s: %s", series_id, payload["error"])
        return None

    records = payload.get("response", {}).get("data")
    if records is None:
        logger.warning("EIA response for %s missing 'response.data' key", series_id)
        return None

    if len(records) == 0:
        logger.warning("EIA returned empty data for %s (%s to %s)", series_id, start, end)
        return None

    df = pd.DataFrame.from_records(records)
    if "period" not in df.columns or "value" not in df.columns:
        logger.warning(
            "EIA response for %s missing required columns; got %s",
            series_id,
            list(df.columns),
        )
        return None

    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df.set_index("period")[["value"]].sort_index()
    df.index.name = "Date"
    df["value"] = df["value"].astype(float)

    # Cache raw parsed response for reuse.
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_CACHE / f"eia_{series_id}.parquet"
    try:
        df.to_parquet(cache_path)
    except Exception:
        logger.exception("Failed to cache EIA %s at %s", series_id, cache_path)

    logger.info(
        "EIA %s: %d rows, %s to %s",
        series_id,
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
    )
    return df


def _next_release_date(obs_date: pd.Timestamp, release_dow: int) -> pd.Timestamp:
    """Return the first date strictly after ``obs_date`` whose weekday is ``release_dow``.

    "Strictly after" (>= obs_date + 1 day) enforces that a period date which
    happens to fall on the release day rolls to the *next* week — the data
    cannot be known the same day it is dated.
    """
    next_day = obs_date + pd.Timedelta(days=1)
    days_ahead = (release_dow - next_day.weekday()) % 7
    return next_day + pd.Timedelta(days=days_ahead)


def align_to_daily(weekly_df: pd.DataFrame, release_day: str) -> pd.DataFrame:
    """Reindex a weekly series onto business days with no-lookahead release logic.

    Each weekly observation is shifted forward to its release date (the next
    occurrence of ``release_day`` strictly after the period date) and then
    forward-filled across business days until the next release. Dates before
    the first release date exist in the output index with ``NaN`` values,
    so the result can be joined cleanly against a full daily trading
    calendar.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        DataFrame with a ``DatetimeIndex`` (weekly period end dates) and a
        single ``value`` column.
    release_day : str
        Lowercase weekday name, one of ``"monday"`` .. ``"friday"``.

    Returns
    -------
    pd.DataFrame
        Business-day DataFrame with the same ``value`` column, forward-filled
        from each observation's release date. Empty input returns an empty
        DataFrame with the correct schema.
    """
    key = release_day.lower()
    if key not in _WEEKDAY_INDEX:
        raise ValueError(
            f"release_day must be one of {sorted(_WEEKDAY_INDEX)}, got {release_day!r}"
        )
    release_dow = _WEEKDAY_INDEX[key]

    if weekly_df.empty:
        return pd.DataFrame({"value": pd.Series(dtype=float)})

    shifted = weekly_df.copy()
    shifted.index = pd.DatetimeIndex(
        [_next_release_date(ts, release_dow) for ts in weekly_df.index]
    )
    shifted = shifted.sort_index()
    # Multiple observations landing on the same release date should not happen
    # with EIA weekly data, but guard against it by keeping the latest.
    shifted = shifted[~shifted.index.duplicated(keep="last")]

    # Start the business-day index from the first raw weekly period date,
    # not the first release date, so that pre-release dates appear in the
    # output with NaN values (matches the docstring contract and lets
    # downstream factor code join against a full trading calendar).
    business_days = pd.bdate_range(start=weekly_df.index.min(), end=shifted.index.max())
    daily = shifted.reindex(business_days).ffill()
    daily.index.name = "Date"
    return daily


def _parse_usda_value(raw: Any) -> float:
    """Parse a USDA NASS ``Value`` string to a float.

    NASS encodes numeric stock quantities as comma-separated strings
    (e.g. ``"1,760,000,000"``). Undisclosed values are reported as
    ``"(D)"`` and should become ``NaN``. Empty strings and ``None``
    likewise map to ``NaN``. Any other non-parseable value also yields
    ``NaN`` rather than raising, so a single malformed row does not
    abort the whole download.
    """
    if raw is None:
        return float("nan")
    text = str(raw).strip()
    if text == "" or text == "(D)":
        return float("nan")
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return float("nan")


# NASS Grain Stocks reference periods are always "first of" a quarter-end
# month. Map the free-text ``reference_period_desc`` values we have seen in
# practice onto the month number of the as-of date.
_USDA_REF_PERIOD_MONTH: dict[str, int] = {
    "FIRST OF MAR": 3,
    "FIRST OF JUN": 6,
    "FIRST OF SEP": 9,
    "FIRST OF DEC": 12,
    # Short variants occasionally returned by older API versions:
    "MAR 1": 3,
    "JUN 1": 6,
    "SEP 1": 9,
    "DEC 1": 12,
}


def _parse_usda_period_date(year: Any, reference_period_desc: Any) -> pd.Timestamp | None:
    """Derive the as-of date of a NASS stocks observation.

    Combines ``year`` with the month implied by ``reference_period_desc``
    (e.g. ``"FIRST OF SEP"`` → September 1 of that year). Returns ``None``
    when either input is missing or the reference period is unrecognised.
    """
    if year is None or reference_period_desc is None:
        return None
    try:
        year_int = int(year)
    except (TypeError, ValueError):
        return None
    month = _USDA_REF_PERIOD_MONTH.get(str(reference_period_desc).strip().upper())
    if month is None:
        return None
    return pd.Timestamp(year=year_int, month=month, day=1)


def download_usda_stocks(
    commodity: str,
    start: str,
    end: str,
    api_key: str,
) -> pd.DataFrame | None:
    """Download a USDA NASS QuickStats stocks series for one crop commodity.

    Parameters
    ----------
    commodity : str
        Commodity symbol, one of the keys of :data:`USDA_STOCK_SERIES`
        (``"ZC"``, ``"ZS"``, ``"ZW"``).
    start, end : str
        Date range in ``YYYY-MM-DD`` format. NASS filters by year, so the
        API request uses ``year__GE`` / ``year__LE`` derived from these
        dates; the returned DataFrame is then trimmed to the exact date
        window on the release (``load_time``) axis.
    api_key : str
        USDA NASS QuickStats API key. Never logged.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with a ``DatetimeIndex`` of release dates (the
        ``load_time`` field, i.e. the day USDA published the report)
        and columns ``value`` (float bushels) and ``period_date``
        (the quarterly as-of date derived from ``year`` +
        ``reference_period_desc``). The index may contain duplicate
        timestamps because a single NASS Grain Stocks report typically
        publishes several quarterly observations on the same
        ``load_time``; rows are deduplicated on the composite key
        ``(load_time, period_date)`` with ``keep="last"``. Sorted
        ascending by ``(load_time, period_date)``. Returns ``None`` on
        network/API failure, on an unknown commodity symbol, on an
        error response body, or when no rows match the ``short_desc``
        filter.

    Notes
    -----
    We filter strictly on ``short_desc == USDA_STOCK_SERIES[commodity]["short_desc"]``
    rather than matching only the leading commodity name. NASS publishes
    several disaggregated stocks series per crop (on-farm vs. off-farm,
    for instance); only the "total" series — exact string match — should
    flow into the inventory surprise factor.

    The ``load_time`` (release date) is the no-lookahead anchor: data
    should only be visible to a strategy from that date onward. The
    ``period_date`` column is the measurement timestamp (e.g. September 1
    for a Q3 Grain Stocks observation) — downstream factor code can use
    it to compute quarterly inventory changes without lookahead because
    both dates are preserved.
    """
    series = USDA_STOCK_SERIES.get(commodity)
    if series is None:
        logger.warning("No USDA stock series registered for %s", commodity)
        return None

    logger.info("Downloading USDA NASS stocks for %s", commodity)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    params = {
        "key": api_key,
        "commodity_desc": series["commodity_desc"],
        "statisticcat_desc": "STOCKS",
        "year__GE": str(start_ts.year),
        "year__LE": str(end_ts.year),
        "agg_level_desc": "NATIONAL",
        "format": "JSON",
    }

    try:
        response = requests.get(
            USDA_NASS_BASE_URL, params=params, timeout=_REQUEST_TIMEOUT_SECONDS
        )
    except requests.RequestException as exc:
        # As with EIA, never include ``exc`` in the log message: default
        # stringification may embed the full URL which contains ``key=...``.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        logger.warning(
            "USDA download failed for %s (%s, status=%s)",
            commodity,
            type(exc).__name__,
            status,
        )
        return None

    if response.status_code != 200:
        logger.warning(
            "USDA download failed for %s: HTTP %d", commodity, response.status_code
        )
        return None

    try:
        payload: dict[str, Any] = response.json()
    except ValueError as exc:
        logger.warning("USDA response for %s was not valid JSON: %s", commodity, exc)
        return None

    # NASS sometimes returns HTTP 200 with an ``error`` body (e.g. invalid
    # key or bad parameter combination). Surface that directly.
    if "error" in payload:
        logger.warning("USDA API error for %s: %s", commodity, payload["error"])
        return None

    records = payload.get("data")
    if records is None:
        logger.warning("USDA response for %s missing 'data' key", commodity)
        return None

    target_short_desc = series["short_desc"]
    filtered = [r for r in records if r.get("short_desc") == target_short_desc]

    if len(filtered) == 0:
        logger.warning(
            "USDA returned no rows matching short_desc=%r for %s (%s to %s)",
            target_short_desc,
            commodity,
            start,
            end,
        )
        return None

    rows: list[dict[str, Any]] = []
    for record in filtered:
        load_time_raw = record.get("load_time")
        if not load_time_raw:
            continue
        try:
            load_time = pd.to_datetime(load_time_raw)
        except (ValueError, TypeError):
            continue
        period_date = _parse_usda_period_date(
            record.get("year"), record.get("reference_period_desc")
        )
        rows.append(
            {
                "load_time": load_time,
                "period_date": period_date,
                "value": _parse_usda_value(record.get("Value")),
            }
        )

    if len(rows) == 0:
        logger.warning("USDA response for %s had no parseable rows", commodity)
        return None

    df = pd.DataFrame.from_records(rows)
    # Sort by (load_time, period_date) so that dedup with keep="last" picks
    # the most recent entry for each (release, as-of) pair. Missing
    # period_date values sort to the end — we still keep them so unknown
    # reference periods never silently vanish.
    df = df.sort_values(["load_time", "period_date"], na_position="last")
    df = df.drop_duplicates(subset=["load_time", "period_date"], keep="last")
    df["value"] = df["value"].astype(float)
    df["period_date"] = pd.to_datetime(df["period_date"])

    df = df.set_index("load_time")
    df.index.name = "Date"

    # NASS filtered by year, so trim to the exact requested window on the
    # release-date axis (the no-lookahead anchor).
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

    if len(df) == 0:
        logger.warning(
            "USDA %s: all rows fell outside window %s to %s", commodity, start, end
        )
        return None

    # Cache raw parsed response for reuse.
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_CACHE / f"usda_{commodity}.parquet"
    try:
        df.to_parquet(cache_path)
    except Exception:
        logger.exception("Failed to cache USDA %s at %s", commodity, cache_path)

    logger.info(
        "USDA %s: %d rows, %s to %s",
        commodity,
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
    )
    return df


def download_all_eia(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Download every EIA series listed in ``configs/inventory.yaml``.

    Date range is read from ``configs/universe.yaml``. The API key is read
    from the ``EIA_API_KEY`` environment variable; if missing, an empty dict
    is returned and the caller is warned.

    Parameters
    ----------
    use_cache : bool
        If True, series whose final Parquet file already exists in
        ``data/raw/inventory/`` are loaded from disk instead of
        re-downloaded.

    Returns
    -------
    dict
        ``{series_name: df}`` for each successfully downloaded series.
        Series that fail are omitted; the orchestrator does not abort on a
        single failure.
    """
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        logger.error("EIA_API_KEY not set — cannot download inventory data")
        return {}

    inventory_config = load_config("inventory")
    series_config = inventory_config["eia"]["series"]

    universe = load_config("universe")
    start = universe["date_range"]["start"]
    end = universe["date_range"]["end"]

    inventory_dir = DATA_RAW / "inventory"
    all_data: dict[str, pd.DataFrame] = {}

    for name, meta in series_config.items():
        cached_path = inventory_dir / f"{name}.parquet"
        if use_cache and cached_path.exists():
            logger.info("Using cached Parquet for %s", name)
            all_data[name] = pd.read_parquet(cached_path)
            continue

        route = EIA_ROUTES.get(name)
        if route is None:
            logger.warning("No EIA route registered for %s — skipping", name)
            continue

        df = download_eia_series(
            series_id=meta["id"],
            route=route,
            start=start,
            end=end,
            api_key=api_key,
        )
        if df is not None:
            all_data[name] = df

    logger.info("EIA download complete: %d series", len(all_data))
    return all_data


def download_all_usda(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Download every USDA NASS stocks series listed in ``configs/inventory.yaml``.

    Crops are read from ``inventory.yaml`` under ``usda.crops``. Date range
    is read from ``configs/universe.yaml``. The API key is read from the
    ``USDA_NASS_API_KEY`` environment variable; if missing, an empty dict
    is returned and the caller is warned.

    Parameters
    ----------
    use_cache : bool
        If True, crops whose final Parquet file already exists in
        ``data/raw/inventory/`` (under the ``usda_`` prefix) are loaded
        from disk instead of re-downloaded.

    Returns
    -------
    dict
        ``{commodity_symbol: df}`` for each successfully downloaded crop.
        Crops that fail are omitted; the orchestrator does not abort on
        a single failure.
    """
    api_key = os.environ.get("USDA_NASS_API_KEY")
    if not api_key:
        logger.error("USDA_NASS_API_KEY not set — cannot download USDA inventory data")
        return {}

    inventory_config = load_config("inventory")
    crops = inventory_config["usda"]["crops"]

    universe = load_config("universe")
    start = universe["date_range"]["start"]
    end = universe["date_range"]["end"]

    inventory_dir = DATA_RAW / "inventory"
    all_data: dict[str, pd.DataFrame] = {}

    for commodity in crops:
        cached_path = inventory_dir / f"usda_{commodity}.parquet"
        if use_cache and cached_path.exists():
            logger.info("Using cached Parquet for USDA %s", commodity)
            all_data[commodity] = pd.read_parquet(cached_path)
            continue

        if commodity not in USDA_STOCK_SERIES:
            logger.warning("No USDA series registered for %s — skipping", commodity)
            continue

        df = download_usda_stocks(
            commodity=commodity,
            start=start,
            end=end,
            api_key=api_key,
        )
        if df is not None:
            all_data[commodity] = df

    logger.info("USDA download complete: %d crops", len(all_data))
    return all_data


def save_inventory_data(data: dict[str, pd.DataFrame], prefix: str = "") -> None:
    """Save each inventory series as Parquet in ``data/raw/inventory/``.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Mapping from series name (or commodity symbol) to DataFrame.
    prefix : str, optional
        Optional filename prefix. For example, ``prefix="usda_"`` saves
        ``"ZC"`` as ``usda_ZC.parquet``. Default is no prefix, which
        keeps behaviour backwards-compatible with the EIA call sites.
    """
    out_dir = DATA_RAW / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, df in data.items():
        path = out_dir / f"{prefix}{name}.parquet"
        df.to_parquet(path)
        if len(df) > 0:
            logger.info(
                "Saved %s → %s (%d rows, %s to %s)",
                name,
                path,
                len(df),
                df.index[0].date(),
                df.index[-1].date(),
            )
        else:
            logger.info("Saved %s → %s (0 rows)", name, path)


def load_inventory_data() -> dict[str, pd.DataFrame]:
    """Load previously saved inventory Parquet files from ``data/raw/inventory/``.

    Returns
    -------
    dict
        Keys are file stems (e.g. ``"crude_stocks"``). Empty dict if the
        directory does not exist.
    """
    inventory_dir = DATA_RAW / "inventory"
    if not inventory_dir.exists():
        logger.warning("No inventory data directory at %s", inventory_dir)
        return {}

    data: dict[str, pd.DataFrame] = {}
    for path in sorted(inventory_dir.glob("*.parquet")):
        name = path.stem
        data[name] = pd.read_parquet(path)
        logger.debug("Loaded %s (%d rows)", name, len(data[name]))

    return data


def main() -> None:
    """Download all EIA and USDA inventory series and save as Parquet."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        logger.debug("python-dotenv not installed; skipping .env loading")

    logger.info("=== EIA Inventory Data Download ===")
    eia_data = download_all_eia(use_cache=True)

    if eia_data:
        save_inventory_data(eia_data)
    else:
        logger.error("No EIA inventory data downloaded — check network and EIA_API_KEY")

    logger.info("=== USDA NASS Inventory Data Download ===")
    usda_data = download_all_usda(use_cache=True)

    if usda_data:
        save_inventory_data(usda_data, prefix="usda_")
    else:
        logger.error(
            "No USDA inventory data downloaded — check network and USDA_NASS_API_KEY"
        )

    if not eia_data and not usda_data:
        return

    logger.info("=== Download Summary ===")
    for name, df in sorted(eia_data.items()):
        if len(df) > 0:
            logger.info(
                "  %-22s  %5d rows  %s → %s",
                name,
                len(df),
                df.index[0].date(),
                df.index[-1].date(),
            )
        else:
            logger.info("  %-22s  %5d rows", name, 0)
    for commodity, df in sorted(usda_data.items()):
        label = f"usda_{commodity}"
        if len(df) > 0:
            logger.info(
                "  %-22s  %5d rows  %s → %s",
                label,
                len(df),
                df.index[0].date(),
                df.index[-1].date(),
            )
        else:
            logger.info("  %-22s  %5d rows", label, 0)


if __name__ == "__main__":
    main()
