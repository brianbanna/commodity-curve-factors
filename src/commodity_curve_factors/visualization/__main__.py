"""Visualization runner: generate all 15 PNGs and website data assets.

Produces:
* All publication-quality PNGs in ``results/figures/``
* ``website/js/chart_data_inline.js`` with Plotly-ready JSON for 3 interactive charts
* Copies PNGs to ``website/assets/figures/``

Run with::

    python -m commodity_curve_factors.visualization
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from commodity_curve_factors.curves.convenience_yield import (
    compute_convenience_yield,
    estimate_storage_cost,
    monthly_convenience_yield,
)
from commodity_curve_factors.signals.curve_regime import classify_regime
from commodity_curve_factors.utils.paths import DATA_PROCESSED, PROJECT_ROOT, RESULTS_FIGURES
from commodity_curve_factors.visualization import (
    curves,
    factors,
    performance,
    risk,
    tsi,
)
from commodity_curve_factors.visualization.style import setup

logger = logging.getLogger(__name__)

_BACKTEST_DIR = DATA_PROCESSED / "backtest"
_CURVES_DIR = DATA_PROCESSED / "curves"
_RF_PATH = DATA_PROCESSED.parent / "raw" / "macro" / "DGS3MO.parquet"

_WEBSITE_JS_DIR = PROJECT_ROOT / "website" / "js"
_WEBSITE_FIG_DIR = PROJECT_ROOT / "website" / "assets" / "figures"

# Regime integer codes consistent with curves.py _REGIME_INT
_REGIME_ORDER: list[str] = [
    "deep_contango",
    "mild_contango",
    "balanced",
    "mild_backwardation",
    "crisis_backwardation",
]
_REGIME_INT: dict[str, int] = {r: i for i, r in enumerate(_REGIME_ORDER)}


# ---------------------------------------------------------------------------
# Chart data builders
# ---------------------------------------------------------------------------


def _build_chart_data() -> dict:
    """Build CHART_DATA for Plotly equity curves.

    Returns
    -------
    dict
        Mapping of strategy/benchmark name → {dates, cumulative_return}.
    """
    result: dict = {}

    # Load benchmarks (daily returns → cumulative wealth index)
    bm_path = _BACKTEST_DIR / "benchmarks.parquet"
    if bm_path.exists():
        bm_df = pd.read_parquet(bm_path)
        bm_df.index = pd.to_datetime(bm_df.index)
        for col in bm_df.columns:
            series = bm_df[col].dropna()
            cum = (1 + series.fillna(0)).cumprod()
            result[col] = {
                "dates": [d.strftime("%Y-%m-%d") for d in cum.index],
                "cumulative_return": [round(v, 6) for v in cum.values.tolist()],
            }

    # Load strategy parquets (skip cost_sensitivity and benchmarks)
    for path in sorted(_BACKTEST_DIR.glob("*.parquet")):
        name = path.stem
        if name in ("benchmarks", "cost_sensitivity"):
            continue
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            cum = df["cumulative"]
            result[name] = {
                "dates": [d.strftime("%Y-%m-%d") for d in cum.index],
                "cumulative_return": [round(v, 6) for v in cum.values.tolist()],
            }
        except Exception:
            logger.warning("Could not load %s for CHART_DATA", name)

    return result


def _build_cost_data() -> dict:
    """Build COST_DATA for the cost sensitivity slider.

    Returns
    -------
    dict
        Arrays: bps, sharpe, cagr, maxdd.
    """
    path = _BACKTEST_DIR / "cost_sensitivity.parquet"
    if not path.exists():
        logger.warning("cost_sensitivity.parquet not found — COST_DATA will be empty")
        return {}
    df = pd.read_parquet(path)
    return {
        "bps": df["cost_bps"].tolist(),
        "sharpe": [round(v, 4) for v in df["sharpe"].tolist()],
        "cagr": [round(v, 4) for v in df["cagr"].tolist()],
        "maxdd": [round(v, 4) for v in df["max_drawdown"].tolist()],
    }


def _build_cy_data() -> dict:
    """Build CY_DATA for the interactive convenience yield explorer.

    Loads all curve parquets, computes daily CY via the same pipeline as
    ``curves.py``, classifies regimes, and downsamples to weekly (W-FRI).

    Returns
    -------
    dict
        {dates, commodities: {SYM: {cy, regime}}, regime_labels}
    """
    # Load curves
    available = sorted(_CURVES_DIR.glob("*.parquet"))
    if not available:
        logger.warning("No curve parquets found — CY_DATA will be empty")
        return {}

    curves_dict: dict[str, pd.DataFrame] = {}
    for path in available:
        sym = path.stem
        curves_dict[sym] = pd.read_parquet(path)

    # Load risk-free rate
    try:
        rf_df = pd.read_parquet(_RF_PATH)
        rf = rf_df["value"]
    except Exception:
        logger.warning("Could not load risk-free rate; using zeros")
        rf = pd.Series(dtype=float)

    storage = estimate_storage_cost(curves_dict)
    daily_cy = compute_convenience_yield(curves_dict, rf, storage)
    monthly_cy = monthly_convenience_yield(daily_cy)
    regimes = classify_regime(monthly_cy)

    # Downsample daily CY to weekly (W-FRI)
    weekly_cy = daily_cy.resample("W-FRI").last()
    # Align regimes (monthly) to weekly dates via forward-fill
    regimes_weekly = regimes.reindex(weekly_cy.index, method="ffill")

    # Build unified date index (strings)
    dates = [d.strftime("%Y-%m-%d") for d in weekly_cy.index]

    commodities: dict[str, dict] = {}
    for sym in weekly_cy.columns:
        cy_vals = weekly_cy[sym].tolist()
        cy_rounded = [round(v, 6) if pd.notna(v) else None for v in cy_vals]

        if sym in regimes_weekly.columns:
            reg_series = regimes_weekly[sym]
            regime_codes = [
                int(_REGIME_INT[str(v)]) if pd.notna(v) and str(v) in _REGIME_INT else None
                for v in reg_series
            ]
        else:
            regime_codes = [None] * len(dates)

        commodities[sym] = {"cy": cy_rounded, "regime": regime_codes}

    return {
        "dates": dates,
        "commodities": commodities,
        "regime_labels": _REGIME_ORDER,
    }


# ---------------------------------------------------------------------------
# chart_data_inline.js writer
# ---------------------------------------------------------------------------


def write_chart_data_inline() -> Path:
    """Generate ``website/js/chart_data_inline.js`` with 3 JS variables.

    Variables: ``CHART_DATA``, ``COST_DATA``, ``CY_DATA``.

    Returns
    -------
    Path
        Path to the written JS file.
    """
    logger.info("Building CHART_DATA …")
    chart_data = _build_chart_data()

    logger.info("Building COST_DATA …")
    cost_data = _build_cost_data()

    logger.info("Building CY_DATA …")
    cy_data = _build_cy_data()

    _WEBSITE_JS_DIR.mkdir(parents=True, exist_ok=True)
    out = _WEBSITE_JS_DIR / "chart_data_inline.js"

    with open(out, "w", encoding="utf-8") as fh:
        fh.write("// Auto-generated by commodity_curve_factors.visualization — do not edit.\n")
        fh.write(f"const CHART_DATA = {json.dumps(chart_data, allow_nan=False)};\n\n")
        fh.write(f"const COST_DATA = {json.dumps(cost_data, allow_nan=False)};\n\n")
        fh.write(f"const CY_DATA = {json.dumps(cy_data, allow_nan=False)};\n")

    logger.info("Wrote %s (%.1f KB)", out, out.stat().st_size / 1024)
    return out


# ---------------------------------------------------------------------------
# Figure copy
# ---------------------------------------------------------------------------


def copy_figures_to_website() -> int:
    """Copy all PNGs from ``results/figures/`` to ``website/assets/figures/``.

    Returns
    -------
    int
        Number of files copied.
    """
    _WEBSITE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for png in RESULTS_FIGURES.glob("*.png"):
        dest = _WEBSITE_FIG_DIR / png.name
        shutil.copy2(png, dest)
        copied += 1
    logger.info("Copied %d PNGs to %s", copied, _WEBSITE_FIG_DIR)
    return copied


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full visualization pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    setup()

    logger.info("=== Generating performance charts ===")
    performance.generate_all()

    logger.info("=== Generating factor charts ===")
    factors.generate_all()

    logger.info("=== Generating curve charts ===")
    curves.generate_all()

    logger.info("=== Generating TSI chart ===")
    tsi.generate_all()

    logger.info("=== Generating risk charts ===")
    risk.generate_all()

    logger.info("=== Writing chart_data_inline.js ===")
    write_chart_data_inline()

    logger.info("=== Copying figures to website ===")
    copy_figures_to_website()

    logger.info("Visualization pipeline complete.")


if __name__ == "__main__":
    main()
