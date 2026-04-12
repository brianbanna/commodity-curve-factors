"""Curve builder: orchestrate per-commodity and universe-wide curve construction.

Reads WRDS contract data (via ``wrds_loader``) and produces a daily
DatetimeIndex × tenor DataFrame for each commodity.  All parameters
(standard tenors, roll rules, interpolation settings) are read from
``configs/curve.yaml`` — nothing is hard-coded here.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from commodity_curve_factors.curves.interpolation import interpolate_curve_day
from commodity_curve_factors.curves.roll_calendar import _active_contracts_from_group
from commodity_curve_factors.utils.config import load_config
from commodity_curve_factors.utils.paths import DATA_PROCESSED

logger = logging.getLogger(__name__)


def build_curve(
    contracts: pd.DataFrame,
    commodity: str,
    curve_config: dict[str, Any],
) -> pd.DataFrame:
    """Build a daily interpolated curve DataFrame for one commodity.

    For each unique ``trade_date`` in ``contracts``:
      1. Identify active contracts using the per-commodity roll rule from
         ``curve_config["roll_rules"]`` (falls back to
         ``curve_config["roll_rules"]["default"]``).
      2. Interpolate the curve at each standard tenor via
         :func:`~commodity_curve_factors.curves.interpolation.interpolate_curve_day`.
      3. Collect results into a DataFrame.

    Parameters
    ----------
    contracts : pd.DataFrame
        Output of ``wrds_loader.load_contracts`` for one symbol.
    commodity : str
        Commodity ticker (e.g. ``"CL"``).  Used to look up roll rules.
    curve_config : dict
        Loaded from ``configs/curve.yaml``.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (``trade_date``), columns = tenor labels
        (``F1M``, ``F2M``, …, ``F12M``), values = interpolated prices.
        Rows sorted by date ascending.
    """
    roll_rules: dict[str, Any] = curve_config["roll_rules"]
    commodity_rule = roll_rules.get(commodity, roll_rules["default"])
    roll_days: int = commodity_rule["roll_days_before_expiry"]

    standard_tenors: list[int] = curve_config["standard_tenors"]
    interp_cfg: dict[str, Any] = curve_config.get("interpolation", {})
    min_contracts: int = interp_cfg.get("min_contracts", 3)
    extrap_max_days: int = interp_cfg.get("extrapolation_max_days", 45)

    tenor_labels = [f"F{m}M" for m in standard_tenors]

    rows: list[pd.Series] = []
    dates: list[pd.Timestamp] = []
    insufficient_days: int = 0

    for ts_key, day_df in contracts.groupby("trade_date", sort=True):
        ts = pd.Timestamp(ts_key)  # type: ignore[arg-type]
        active = _active_contracts_from_group(day_df, ts, roll_days)
        if len(active) < min_contracts:
            rows.append(pd.Series({label: np.nan for label in tenor_labels}, name=ts))
            dates.append(ts)
            insufficient_days += 1
            continue
        curve_row = interpolate_curve_day(
            active,
            standard_tenors,
            extrapolation_max_days=extrap_max_days,
            min_contracts=min_contracts,
        )
        if curve_row.isna().all():
            insufficient_days += 1
        rows.append(curve_row)
        dates.append(ts)

    if not rows:
        return pd.DataFrame(columns=tenor_labels, index=pd.DatetimeIndex([], name="trade_date"))

    result = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="trade_date"))

    if insufficient_days > 0:
        logger.info(
            "build_curve %s: %d/%d days had insufficient contracts for any tenor",
            commodity,
            insufficient_days,
            len(rows),
        )

    logger.info(
        "build_curve %s: %d rows, date range %s to %s",
        commodity,
        len(result),
        result.index.min().date(),
        result.index.max().date(),
    )
    return result


def build_all_curves(
    contracts_by_commodity: dict[str, pd.DataFrame],
    curve_config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Build curves for every commodity in ``contracts_by_commodity``.

    Parameters
    ----------
    contracts_by_commodity : dict[str, pd.DataFrame]
        Keyed by commodity symbol; values are outputs of
        ``wrds_loader.load_contracts``.
    curve_config : dict
        Loaded from ``configs/curve.yaml``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by commodity symbol; values are daily curve DataFrames.
    """
    curves: dict[str, pd.DataFrame] = {}

    for symbol, contracts in contracts_by_commodity.items():
        logger.info("Building curve for %s (%d rows)", symbol, len(contracts))
        curve = build_curve(contracts, symbol, curve_config)
        curves[symbol] = curve

        # Summary: NaN fraction per tenor
        nan_fracs = curve.isna().mean()
        for col, frac in nan_fracs.items():
            if frac > 0:
                logger.info("  %s %s: %.1f%% NaN", symbol, col, frac * 100)

    logger.info("build_all_curves complete: %d commodities", len(curves))
    return curves


def save_curves(
    curves: dict[str, pd.DataFrame],
    *,
    out_dir: Path | None = None,
) -> None:
    """Save each curve to ``out_dir/<symbol>.parquet``.

    Parameters
    ----------
    curves : dict[str, pd.DataFrame]
        Keyed by commodity symbol.
    out_dir : Path or None
        Output directory.  Defaults to ``DATA_PROCESSED / "curves"``.
    """
    if out_dir is None:
        out_dir = DATA_PROCESSED / "curves"

    out_dir.mkdir(parents=True, exist_ok=True)

    for symbol, df in curves.items():
        path = out_dir / f"{symbol}.parquet"
        df.to_parquet(path)
        logger.info("Saved curve %s → %s (%d rows)", symbol, path, len(df))


def load_curves(
    *,
    in_dir: Path | None = None,
    symbols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load saved curve Parquet files.

    Parameters
    ----------
    in_dir : Path or None
        Directory to read from.  Defaults to ``DATA_PROCESSED / "curves"``.
    symbols : list[str] or None
        Commodity symbols to load.  Defaults to all keys in
        ``configs/universe.yaml``.  Missing files are logged and skipped.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by commodity symbol.
    """
    if in_dir is None:
        in_dir = DATA_PROCESSED / "curves"

    if symbols is None:
        universe = load_config("universe")
        symbols = list(universe["commodities"].keys())

    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = in_dir / f"{sym}.parquet"
        if not path.exists():
            logger.warning("Curve file not found for %s at %s — skipping", sym, path)
            continue
        df = pd.read_parquet(path)
        result[sym] = df
        logger.info("Loaded curve %s: %d rows", sym, len(df))

    logger.info("load_curves: %d/%d symbols loaded", len(result), len(symbols))
    return result
