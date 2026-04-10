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

_WEEKDAY_INDEX: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _extract_v2_series_id(raw_id: str) -> str:
    """Extract the v2 series ID from a legacy ``PET.WCESTUS1.W``-style string.

    The EIA v1 API used dotted identifiers like ``PET.WCESTUS1.W``, but the v2
    API's ``facets[series][]`` filter expects just the middle part
    (``WCESTUS1``). If the incoming id already has no dots it is returned as
    is, which allows the YAML config to use either style.
    """
    if "." not in raw_id:
        return raw_id
    parts = raw_id.split(".")
    # Strip any empty trailing frequency suffix and take the longest segment.
    segments = [p for p in parts if p]
    if len(segments) >= 2:
        return segments[1]
    return segments[0]


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
        EIA v2 series identifier (e.g. ``"WCESTUS1"``). Legacy dotted ids
        like ``"PET.WCESTUS1.W"`` are also accepted and normalised.
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
    v2_id = _extract_v2_series_id(series_id)
    logger.info("Downloading EIA series %s from %s", v2_id, route)

    url = f"{_EIA_API_BASE}/{route}/data/"
    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": v2_id,
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
            v2_id,
            type(exc).__name__,
            status,
        )
        return None

    if response.status_code != 200:
        logger.warning("EIA download failed for %s: HTTP %d", v2_id, response.status_code)
        return None

    try:
        payload: dict[str, Any] = response.json()
    except ValueError as exc:
        logger.warning("EIA response for %s was not valid JSON: %s", v2_id, exc)
        return None

    # EIA v2 sometimes returns HTTP 200 with an error body (e.g. invalid API
    # key) and no ``response.data`` field. Surface that directly.
    if "error" in payload:
        logger.warning("EIA API error for %s: %s", v2_id, payload["error"])
        return None

    records = payload.get("response", {}).get("data")
    if records is None:
        logger.warning("EIA response for %s missing 'response.data' key", v2_id)
        return None

    if len(records) == 0:
        logger.warning("EIA returned empty data for %s (%s to %s)", v2_id, start, end)
        return None

    df = pd.DataFrame.from_records(records)
    if "period" not in df.columns or "value" not in df.columns:
        logger.warning(
            "EIA response for %s missing required columns; got %s",
            v2_id,
            list(df.columns),
        )
        return None

    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df.set_index("period")[["value"]].sort_index()
    df.index.name = "Date"
    df["value"] = df["value"].astype(float)

    # Cache raw parsed response for reuse. Key the cache file by v2 id so
    # callers using either id style land on the same file.
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_CACHE / f"eia_{v2_id}.parquet"
    try:
        df.to_parquet(cache_path)
    except Exception:
        logger.exception("Failed to cache EIA %s at %s", v2_id, cache_path)

    logger.info(
        "EIA %s: %d rows, %s to %s",
        v2_id,
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
    the first release are left as ``NaN``.

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

    business_days = pd.bdate_range(start=shifted.index.min(), end=shifted.index.max())
    daily = shifted.reindex(business_days).ffill()
    daily.index.name = "Date"
    return daily


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


def save_inventory_data(data: dict[str, pd.DataFrame]) -> None:
    """Save each inventory series as Parquet in ``data/raw/inventory/``."""
    out_dir = DATA_RAW / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, df in data.items():
        path = out_dir / f"{name}.parquet"
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
    """Download all EIA inventory series and save as Parquet."""
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
    data = download_all_eia(use_cache=True)

    if not data:
        logger.error("No inventory data downloaded — check network and EIA_API_KEY")
        return

    save_inventory_data(data)

    logger.info("=== Download Summary ===")
    for name, df in sorted(data.items()):
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


if __name__ == "__main__":
    main()
