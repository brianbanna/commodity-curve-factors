"""Download macro time series: FRED rates/USD/breakevens, VIX, and benchmarks.

Series downloaded:
  - FRED (via fredapi): broad USD index, 10Y Treasury, 5Y breakeven inflation,
    3M T-bill (used as risk-free rate proxy).
  - VIX: daily close from yfinance ticker ``^VIX``.
  - Benchmarks: S&P 500 index (``^GSPC``) and US aggregate bond ETF (``AGG``).

Used downstream for macro factor exposure regressions, VIX-based regime
classification, and strategy-vs-benchmark comparisons.

Usage:
    python -m commodity_curve_factors.data.macro_loader
"""

import logging
import os

import pandas as pd
import yfinance as yf

from commodity_curve_factors.utils.config import load_config
from commodity_curve_factors.utils.paths import DATA_CACHE, DATA_RAW

logger = logging.getLogger(__name__)

MACRO_SERIES: dict[str, str] = {
    "usd_index": "DTWEXBGS",
    "dgs10": "DGS10",
    "t5yie": "T5YIE",
    "dgs3mo": "DGS3MO",
}

# yfinance tickers used for non-FRED series. ``^GSPC`` (S&P 500 index) is used
# instead of the SPY ETF because the index has longer history with no ETF fees
# baked into the price. The internal dict key stays ``"spy"`` as a short,
# readable alias for "S&P 500 equivalent" used elsewhere in the codebase.
_BENCHMARK_TICKERS: dict[str, str] = {
    "spy": "^GSPC",
    "agg": "AGG",
}

_VIX_TICKER = "^VIX"


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns to plain column names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_fred_series(
    series_id: str,
    start: str,
    end: str,
    api_key: str,
) -> pd.DataFrame | None:
    """Download a single FRED series via fredapi.

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g. ``"DGS10"``).
    start, end : str
        Date range in YYYY-MM-DD format.
    api_key : str
        FRED API key.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with DatetimeIndex and a single ``value`` column, or None
        on failure.
    """
    logger.info("Downloading FRED series %s", series_id)

    try:
        from fredapi import Fred
    except ImportError:
        logger.exception("fredapi not installed")
        return None

    try:
        fred = Fred(api_key=api_key)
        series = fred.get_series(
            series_id,
            observation_start=start,
            observation_end=end,
        )
    except Exception as exc:
        logger.warning("FRED download failed for %s: %s", series_id, exc)
        return None

    if series is None or len(series) == 0:
        logger.warning("FRED returned empty series for %s", series_id)
        return None

    df = pd.DataFrame({"value": series})
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Date"
    df = df.sort_index().dropna(subset=["value"])

    # Cache raw response for reuse if needed later.
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_CACHE / f"fred_{series_id}.parquet"
    try:
        df.to_parquet(cache_path)
    except Exception:
        logger.exception("Failed to cache FRED %s at %s", series_id, cache_path)

    logger.info(
        "FRED %s: %d rows, %s to %s",
        series_id,
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
    )
    return df


def _download_yf_series(
    ticker: str,
    cache_key: str,
    keep_cols: list[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """Download a single yfinance ticker with caching and column normalization.

    Shared backend for :func:`download_vix` and :func:`download_benchmarks`.
    Handles the cache read/write, MultiIndex column flattening, DatetimeIndex
    normalization, OHLCV column filtering, and ``Close`` dropna.

    Parameters
    ----------
    ticker : str
        yfinance ticker symbol (e.g. ``"^VIX"``, ``"^GSPC"``, ``"AGG"``).
    cache_key : str
        Short name used in the cache filename (``yf_<cache_key>.parquet``)
        and in log messages.
    keep_cols : list[str]
        OHLCV column names to retain in the output. Columns not present in
        the yfinance response are silently dropped.
    start, end : str
        Date range in YYYY-MM-DD format.
    use_cache : bool
        If True and a cached Parquet exists, return it without hitting the
        network.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with DatetimeIndex and the requested columns, or None on
        failure.
    """
    cache_path = DATA_CACHE / f"yf_{cache_key}.parquet"
    if use_cache and cache_path.exists():
        logger.info("Using cached yfinance data for %s", cache_key)
        return pd.read_parquet(cache_path)

    logger.info("Downloading %s (%s) from yfinance", cache_key, ticker)

    try:
        df: pd.DataFrame = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )
    except Exception:
        logger.exception("yfinance download failed for %s (%s)", cache_key, ticker)
        return None

    if df is None or df.empty:
        logger.warning("No data returned for %s (%s)", cache_key, ticker)
        return None

    df = _flatten_yf_columns(df)
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Date"
    df = df.sort_index()

    available = [c for c in keep_cols if c in df.columns]
    df = df[available]
    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])

    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(cache_path)
    except Exception:
        logger.exception("Failed to cache %s at %s", cache_key, cache_path)

    logger.info(
        "%s: %d rows, %s to %s",
        cache_key,
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
    )
    return df


def download_vix(start: str, end: str, use_cache: bool = True) -> pd.DataFrame | None:
    """Download daily VIX OHLCV from yfinance (``^VIX``)."""
    return _download_yf_series(
        ticker=_VIX_TICKER,
        cache_key="vix",
        keep_cols=["Open", "High", "Low", "Close", "Volume"],
        start=start,
        end=end,
        use_cache=use_cache,
    )


def download_benchmarks(
    start: str,
    end: str,
    keys: list[str] | None = None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download S&P 500 and US aggregate bond benchmarks from yfinance.

    The S&P 500 is downloaded via the index ticker ``^GSPC`` (not the SPY ETF)
    because the index has longer history and no ETF fees baked in. The
    returned dict still uses ``"spy"`` as the key for downstream naming
    consistency.

    Parameters
    ----------
    start, end : str
        Date range in YYYY-MM-DD format.
    keys : list[str] or None
        If provided, only download this subset of benchmark keys (e.g.
        ``["spy"]``). Unknown keys are ignored. Default ``None`` downloads
        every ticker in ``_BENCHMARK_TICKERS``.
    use_cache : bool
        If True, return cached Parquet files from ``data/cache/`` when they
        exist.

    Returns
    -------
    dict
        ``{"spy": df_gspc, "agg": df_agg}``. Missing downloads are omitted.
    """
    if keys is None:
        tickers_to_fetch = _BENCHMARK_TICKERS
    else:
        tickers_to_fetch = {k: _BENCHMARK_TICKERS[k] for k in keys if k in _BENCHMARK_TICKERS}

    result: dict[str, pd.DataFrame] = {}
    for key, ticker in tickers_to_fetch.items():
        df = _download_yf_series(
            ticker=ticker,
            cache_key=key,
            keep_cols=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
            start=start,
            end=end,
            use_cache=use_cache,
        )
        if df is not None:
            result[key] = df

    return result


def download_all_macro(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Download all macro series (FRED + VIX + benchmarks).

    Date range is read from ``configs/universe.yaml``. The FRED API key is
    read from the ``FRED_API_KEY`` environment variable; if missing, the
    FRED block is skipped with a warning but VIX and benchmarks are still
    downloaded.

    Parameters
    ----------
    use_cache : bool
        If True, skip series whose final Parquet file already exists in
        ``data/raw/macro/``.

    Returns
    -------
    dict
        Flat dict keyed by series name
        (``usd_index``, ``dgs10``, ``t5yie``, ``dgs3mo``, ``vix``, ``spy``,
        ``agg``). Series that failed to download are omitted.
    """
    universe = load_config("universe")
    start = universe["date_range"]["start"]
    end = universe["date_range"]["end"]

    macro_dir = DATA_RAW / "macro"
    all_data: dict[str, pd.DataFrame] = {}

    def _cached(name: str) -> pd.DataFrame | None:
        path = macro_dir / f"{name}.parquet"
        if use_cache and path.exists():
            logger.info("Using cached Parquet for %s", name)
            return pd.read_parquet(path)
        return None

    # FRED series
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.warning("FRED_API_KEY not set — skipping FRED series")
    else:
        for name, series_id in MACRO_SERIES.items():
            cached = _cached(name)
            if cached is not None:
                all_data[name] = cached
                continue
            df = download_fred_series(series_id, start, end, api_key)
            if df is not None:
                all_data[name] = df

    # VIX
    cached_vix = _cached("vix")
    if cached_vix is not None:
        all_data["vix"] = cached_vix
    else:
        vix = download_vix(start, end, use_cache=use_cache)
        if vix is not None:
            all_data["vix"] = vix

    # Benchmarks (spy, agg)
    bench_to_fetch: dict[str, str] = {}
    for key, ticker in _BENCHMARK_TICKERS.items():
        cached = _cached(key)
        if cached is not None:
            all_data[key] = cached
        else:
            bench_to_fetch[key] = ticker

    if bench_to_fetch:
        # Only hit the network for benchmarks not already in the cache.
        benches = download_benchmarks(
            start, end, keys=list(bench_to_fetch), use_cache=use_cache
        )
        for key in bench_to_fetch:
            if key in benches:
                all_data[key] = benches[key]

    logger.info("Macro download complete: %d series", len(all_data))
    return all_data


def save_macro_data(data: dict[str, pd.DataFrame]) -> None:
    """Save each macro series as Parquet in ``data/raw/macro/``."""
    out_dir = DATA_RAW / "macro"
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


def load_macro_data() -> dict[str, pd.DataFrame]:
    """Load previously saved macro Parquet files from ``data/raw/macro/``.

    Returns
    -------
    dict
        Keys are file stems (e.g. ``"dgs10"``, ``"vix"``, ``"spy"``). Empty
        dict if the directory does not exist.
    """
    macro_dir = DATA_RAW / "macro"
    if not macro_dir.exists():
        logger.warning("No macro data directory at %s", macro_dir)
        return {}

    data: dict[str, pd.DataFrame] = {}
    for path in sorted(macro_dir.glob("*.parquet")):
        name = path.stem
        data[name] = pd.read_parquet(path)
        logger.debug("Loaded %s (%d rows)", name, len(data[name]))

    return data


def main() -> None:
    """Download all macro series and save as Parquet."""
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

    logger.info("=== Macro Data Download ===")
    data = download_all_macro(use_cache=True)

    if not data:
        logger.error("No macro data downloaded — check network and FRED_API_KEY")
        return

    save_macro_data(data)

    logger.info("=== Download Summary ===")
    for name, df in sorted(data.items()):
        if len(df) > 0:
            logger.info(
                "  %-10s  %5d rows  %s → %s",
                name,
                len(df),
                df.index[0].date(),
                df.index[-1].date(),
            )
        else:
            logger.info("  %-10s  %5d rows", name, 0)


if __name__ == "__main__":
    main()
