"""Download continuous futures price data for all commodities.

Primary source: yfinance (front-month continuous contracts).
Back-month contracts (2nd through 12th) will come from Nasdaq Data Link (task 2.2).

yfinance provides free daily OHLCV for front-month continuous futures using the
'{SYMBOL}=F' convention (e.g. 'CL=F' for WTI crude front month). Data available
back to ~2000 for most commodities.

Usage:
    python -m commodity_curve_factors.data.futures_loader
"""

import logging
import time

import pandas as pd
import yfinance as yf

from commodity_curve_factors.utils.config import load_config
from commodity_curve_factors.utils.paths import DATA_RAW

logger = logging.getLogger(__name__)

THROTTLE_SECONDS = 1.0


def _get_yfinance_symbol(symbol: str) -> str | None:
    """Look up the yfinance ticker for a commodity from universe.yaml."""
    universe = load_config("universe")
    spec = universe["commodities"].get(symbol)
    if spec is None:
        return None
    result: str | None = spec.get("yfinance_symbol")
    return result


def download_front_month(
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """Download front-month continuous futures from yfinance.

    Parameters
    ----------
    symbol : str
        Commodity code (e.g. 'CL', 'GC').
    start, end : str
        Date range in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume],
        or None if download failed.
    """
    yf_ticker = _get_yfinance_symbol(symbol)
    if yf_ticker is None:
        logger.error("No yfinance_symbol in universe.yaml for %s", symbol)
        return None

    logger.info("Downloading %s (%s) from yfinance", symbol, yf_ticker)

    try:
        df: pd.DataFrame = yf.download(yf_ticker, start=start, end=end, progress=False)
    except Exception:
        logger.exception("yfinance download failed for %s", symbol)
        return None

    if df.empty:
        logger.warning("No data returned for %s (%s)", symbol, yf_ticker)
        return None

    # yfinance returns MultiIndex columns when downloading a single ticker:
    # ('Close', 'CL=F'), etc. Flatten to simple column names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Date"
    df = df.sort_index()

    # Keep standard OHLCV columns
    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep_cols]

    # Drop rows where Close is NaN
    df = df.dropna(subset=["Close"])

    logger.info(
        "%s front month: %d rows, %s to %s",
        symbol,
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
    )
    return df


def download_all_front_month(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Download front-month data for all commodities in the universe.

    Parameters
    ----------
    use_cache : bool
        If True, skip commodities that already have saved Parquet files.

    Returns
    -------
    dict
        Keys are commodity codes (e.g. 'CL'), values are OHLCV DataFrames.
    """
    universe = load_config("universe")
    start = universe["date_range"]["start"]
    end = universe["date_range"]["end"]
    commodities = universe["commodities"]

    all_data: dict[str, pd.DataFrame] = {}

    for symbol in commodities:
        # Check cache
        parquet_path = DATA_RAW / "futures" / f"{symbol}_front.parquet"
        if use_cache and parquet_path.exists():
            logger.info("Using cached Parquet for %s front month", symbol)
            all_data[symbol] = pd.read_parquet(parquet_path)
            continue

        df = download_front_month(symbol, start, end)
        if df is not None:
            all_data[symbol] = df

        time.sleep(THROTTLE_SECONDS)

    logger.info("Front-month download complete: %d/%d commodities", len(all_data), len(commodities))
    return all_data


def save_front_month_data(data: dict[str, pd.DataFrame]) -> None:
    """Save front-month data as Parquet files to data/raw/futures/."""
    out_dir = DATA_RAW / "futures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for symbol, df in data.items():
        path = out_dir / f"{symbol}_front.parquet"
        df.to_parquet(path)
        logger.info("Saved %s → %s (%d rows)", symbol, path, len(df))


def load_front_month_data() -> dict[str, pd.DataFrame]:
    """Load previously saved front-month Parquet files.

    Returns
    -------
    dict
        Keys are commodity codes (e.g. 'CL'), values are OHLCV DataFrames.
    """
    futures_dir = DATA_RAW / "futures"
    if not futures_dir.exists():
        logger.warning("No futures data directory at %s", futures_dir)
        return {}

    data = {}
    for path in sorted(futures_dir.glob("*_front.parquet")):
        symbol = path.stem.replace("_front", "")
        data[symbol] = pd.read_parquet(path)
        logger.debug("Loaded %s (%d rows)", symbol, len(data[symbol]))

    return data


def main() -> None:
    """Download all front-month futures data and save as Parquet."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=== Front-Month Futures Data Download ===")
    data = download_all_front_month(use_cache=True)

    if not data:
        logger.error("No data downloaded — check network connectivity")
        return

    save_front_month_data(data)

    # Summary
    logger.info("=== Download Summary ===")
    for symbol, df in sorted(data.items()):
        logger.info(
            "  %-4s  %5d rows  %s → %s",
            symbol,
            len(df),
            df.index[0].date(),
            df.index[-1].date(),
        )


if __name__ == "__main__":
    main()
