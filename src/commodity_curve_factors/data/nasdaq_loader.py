"""Download back-month futures contracts from Nasdaq Data Link (CHRIS dataset).

The CHRIS (wiki continuous) dataset provides continuous back-month contracts
keyed by tenor (1 = front, 2 = second nearest, ..., up to 12). This module
extends the front-month data collected via yfinance (task 2.1) with the deeper
curve needed to compute slope, curvature, and carry signals.

CHRIS is community-maintained and has gaps — not every commodity has all 12
back-month contracts available. Missing contracts are logged and skipped.

Usage:
    python -m commodity_curve_factors.data.nasdaq_loader
"""

import logging
import os
import time

import nasdaqdatalink  # type: ignore[import-untyped]
import pandas as pd

from commodity_curve_factors.utils.config import load_config
from commodity_curve_factors.utils.paths import DATA_CACHE, DATA_RAW

logger = logging.getLogger(__name__)

THROTTLE_SECONDS = 1.0


def build_chris_symbol(nasdaq_prefix: str, contract_num: int) -> str:
    """Build a CHRIS dataset symbol for a given commodity and contract tenor.

    Parameters
    ----------
    nasdaq_prefix : str
        Prefix like ``"CHRIS/CME_CL"`` or ``"CHRIS/ICE_KC"``.
    contract_num : int
        Contract tenor (1 = front, 2 = second, ..., up to 12).

    Returns
    -------
    str
        Full CHRIS symbol, e.g. ``"CHRIS/CME_CL3"``.
    """
    return f"{nasdaq_prefix}{contract_num}"


def _safe_symbol(symbol: str) -> str:
    """Convert a CHRIS symbol into a filename-safe string."""
    return symbol.replace("/", "_")


def download_chris_contract(
    symbol: str,
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """Download one CHRIS continuous contract from Nasdaq Data Link.

    Parameters
    ----------
    symbol : str
        Full CHRIS symbol (e.g. ``"CHRIS/CME_CL3"``).
    start, end : str
        Date range in YYYY-MM-DD format.
    use_cache : bool
        If True, load from local Parquet cache when available.

    Returns
    -------
    pd.DataFrame or None
        OHLCV DataFrame with DatetimeIndex, or None on failure.
    """
    cache_dir = DATA_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"nasdaq_{_safe_symbol(symbol)}.parquet"

    if use_cache and cache_path.exists():
        logger.info("Using cached Nasdaq data for %s", symbol)
        return pd.read_parquet(cache_path)

    logger.info("Downloading %s from Nasdaq Data Link", symbol)

    try:
        df: pd.DataFrame = nasdaqdatalink.get(symbol, start_date=start, end_date=end)
    except Exception:
        logger.warning("Nasdaq Data Link download failed for %s", symbol, exc_info=True)
        return None

    if df is None or df.empty:
        logger.warning("No data returned for %s", symbol)
        return None

    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Date"
    df = df.sort_index()

    try:
        df.to_parquet(cache_path)
    except Exception:
        logger.warning("Failed to cache %s to %s", symbol, cache_path, exc_info=True)

    logger.info(
        "%s: %d rows, %s to %s",
        symbol,
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
    )
    return df


def download_back_months(
    symbol: str,
    nasdaq_prefix: str,
    max_tenor: int,
    start: str,
    end: str,
    use_cache: bool = True,
) -> dict[int, pd.DataFrame]:
    """Download contracts 1..max_tenor for one commodity.

    Parameters
    ----------
    symbol : str
        Commodity code (e.g. ``"CL"``). Used only for logging.
    nasdaq_prefix : str
        CHRIS prefix, e.g. ``"CHRIS/CME_CL"``.
    max_tenor : int
        Highest contract tenor to request (inclusive).
    start, end : str
        Date range in YYYY-MM-DD format.
    use_cache : bool
        Whether to use the Parquet cache for raw CHRIS responses.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping from contract tenor to OHLCV DataFrame. Contracts that fail
        to download are skipped (not included in the dict).
    """
    contracts: dict[int, pd.DataFrame] = {}
    for n in range(1, max_tenor + 1):
        chris_symbol = build_chris_symbol(nasdaq_prefix, n)
        df = download_chris_contract(chris_symbol, start, end, use_cache=use_cache)
        if df is not None:
            contracts[n] = df
        else:
            logger.warning("Skipping %s contract %d (%s)", symbol, n, chris_symbol)
        time.sleep(THROTTLE_SECONDS)
    logger.info("%s: downloaded %d/%d back-month contracts", symbol, len(contracts), max_tenor)
    return contracts


def download_all_back_months(use_cache: bool = True) -> dict[str, dict[int, pd.DataFrame]]:
    """Download back-month contracts for every commodity in the universe.

    Parameters
    ----------
    use_cache : bool
        If True, skip commodities whose full set of output Parquet files
        (contracts 1..max_tenor) already exists under ``data/raw/futures/chris/``.
        Matches the semantics used by :mod:`futures_loader`.

    Returns
    -------
    dict[str, dict[int, pd.DataFrame]]
        Nested dict of ``{symbol: {contract_num: df}}``.
    """
    universe = load_config("universe")
    start = universe["date_range"]["start"]
    end = universe["date_range"]["end"]
    commodities = universe["commodities"]

    chris_dir = DATA_RAW / "futures" / "chris"
    all_data: dict[str, dict[int, pd.DataFrame]] = {}

    for symbol, spec in commodities.items():
        nasdaq_prefix = spec.get("nasdaq_prefix")
        max_tenor = spec.get("max_tenor")
        if nasdaq_prefix is None or max_tenor is None:
            logger.warning("Skipping %s — missing nasdaq_prefix or max_tenor", symbol)
            continue

        # If all expected output files already exist, load them instead of redownloading.
        expected_paths = [
            chris_dir / f"{symbol}_c{n}.parquet" for n in range(1, max_tenor + 1)
        ]
        if use_cache and all(p.exists() for p in expected_paths):
            logger.info(
                "Using cached Parquet files for %s back months (1..%d)",
                symbol,
                max_tenor,
            )
            all_data[symbol] = {
                n: pd.read_parquet(p) for n, p in enumerate(expected_paths, start=1)
            }
            continue

        contracts = download_back_months(
            symbol=symbol,
            nasdaq_prefix=nasdaq_prefix,
            max_tenor=max_tenor,
            start=start,
            end=end,
            use_cache=use_cache,
        )
        if contracts:
            all_data[symbol] = contracts

    logger.info(
        "Back-month download complete: %d/%d commodities had any data",
        len(all_data),
        len(commodities),
    )
    return all_data


def save_back_month_data(data: dict[str, dict[int, pd.DataFrame]]) -> None:
    """Save back-month contract data as Parquet files under data/raw/futures/chris/."""
    out_dir = DATA_RAW / "futures" / "chris"
    out_dir.mkdir(parents=True, exist_ok=True)

    for symbol, contracts in data.items():
        for n, df in contracts.items():
            path = out_dir / f"{symbol}_c{n}.parquet"
            df.to_parquet(path)
            logger.info("Saved %s c%d -> %s (%d rows)", symbol, n, path, len(df))


def load_back_month_data() -> dict[str, dict[int, pd.DataFrame]]:
    """Load previously saved back-month Parquet files.

    Returns
    -------
    dict[str, dict[int, pd.DataFrame]]
        Same structure as :func:`download_all_back_months`.
    """
    chris_dir = DATA_RAW / "futures" / "chris"
    if not chris_dir.exists():
        logger.warning("No CHRIS data directory at %s", chris_dir)
        return {}

    data: dict[str, dict[int, pd.DataFrame]] = {}
    # Only match files whose suffix after "_c" starts with a digit, e.g. "CL_c3".
    # This guards against stray files like "HG_carry_c3.parquet" or "CL_cfoo.parquet".
    for path in sorted(chris_dir.glob("*_c[0-9]*.parquet")):
        stem = path.stem  # e.g. "CL_c3"
        symbol, tenor_part = stem.rsplit("_c", 1)
        if not tenor_part.isdigit():
            logger.debug("Skipping unrecognized filename %s", path.name)
            continue
        n = int(tenor_part)
        data.setdefault(symbol, {})[n] = pd.read_parquet(path)
        logger.debug("Loaded %s c%d (%d rows)", symbol, n, len(data[symbol][n]))

    return data


def main() -> None:
    """Download all back-month futures data and save as Parquet."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load env vars (.env) for the Nasdaq Data Link API key.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv not installed — relying on process environment only")

    api_key = os.environ.get("NASDAQ_DATA_LINK_API_KEY")
    if not api_key:
        logger.error(
            "NASDAQ_DATA_LINK_API_KEY not set — add it to .env or export it "
            "before running this module"
        )
        return

    nasdaqdatalink.ApiConfig.api_key = api_key

    logger.info("=== Back-Month Futures Data Download (Nasdaq Data Link / CHRIS) ===")
    data = download_all_back_months(use_cache=True)

    if not data:
        logger.error("No data downloaded — check API key, network, and CHRIS availability")
        return

    save_back_month_data(data)

    # Summary
    logger.info("=== Download Summary ===")
    for symbol in sorted(data):
        contracts = data[symbol]
        tenors = sorted(contracts.keys())
        logger.info("  %-4s  %2d contracts  tenors=%s", symbol, len(tenors), tenors)


if __name__ == "__main__":
    main()
