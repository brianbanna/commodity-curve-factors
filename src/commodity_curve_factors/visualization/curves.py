"""Curve-level visualization charts for the commodity curve factors research.

Generates 2 publication-quality figures:

* 07 — Convenience yield time series for CL, NG, GC (3 subplots, regime shading)
* 08 — Curve regime heatmap (19 commodities × time, quarterly)

Call ``generate_all()`` to produce all charts in one shot.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from commodity_curve_factors.curves.convenience_yield import (
    compute_convenience_yield,
    estimate_storage_cost,
    monthly_convenience_yield,
)
from commodity_curve_factors.signals.curve_regime import classify_regime
from commodity_curve_factors.utils.paths import DATA_PROCESSED, DATA_RAW
from commodity_curve_factors.visualization.style import (
    ACCENT,
    BG_COLOR,
    FG_COLOR,
    PAPER,
    UP_COLOR,
    savefig,
    setup,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CURVES_DIR = DATA_PROCESSED / "curves"
_RF_PATH = DATA_RAW / "macro" / "DGS3MO.parquet"

# All 19 commodity symbols for the full heatmap
_ALL_SYMS: list[str] = [
    "CL", "NG", "HO", "RB",  # energy
    "GC", "SI", "HG", "PA", "PL",  # metals
    "ZC", "ZS", "ZW", "KC", "SB", "CC",  # ags
    "LC", "LH", "FC",  # livestock
    "LB",  # lumber
]

# Regime display order and colors
_REGIME_ORDER: list[str] = [
    "deep_contango",
    "mild_contango",
    "balanced",
    "mild_backwardation",
    "crisis_backwardation",
]

_REGIME_COLORS: dict[str, str] = {
    "deep_contango": "#b87c6c",       # copper
    "mild_contango": "#c5b58c",       # tan
    "balanced": "#2a2a2c",            # dark
    "mild_backwardation": "#6a9070",  # muted jade
    "crisis_backwardation": "#8ca891",  # sage
}

_REGIME_LABELS: dict[str, str] = {
    "deep_contango": "Deep Contango",
    "mild_contango": "Mild Contango",
    "balanced": "Balanced",
    "mild_backwardation": "Mild Backwardation",
    "crisis_backwardation": "Crisis Backwardation",
}

# Regime integer mapping for heatmap
_REGIME_INT: dict[str, int] = {r: i for i, r in enumerate(_REGIME_ORDER)}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_curves() -> dict[str, pd.DataFrame]:
    """Load all curve parquet files keyed by symbol.

    Returns
    -------
    dict[str, pd.DataFrame]
        Symbol -> DataFrame with columns F1M, F2M, F3M, F6M, F9M, F12M.
    """
    available = sorted(_CURVES_DIR.glob("*.parquet"))
    curves: dict[str, pd.DataFrame] = {}
    for path in available:
        sym = path.stem
        curves[sym] = pd.read_parquet(path)
    logger.debug("Loaded %d curve files", len(curves))
    return curves


def _load_risk_free() -> pd.Series:
    """Load the 3-month T-bill rate as a Series.

    Returns
    -------
    pd.Series
        Annualised risk-free rate (percentage).
    """
    df = pd.read_parquet(_RF_PATH)
    return df["value"]


def _build_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the convenience yield and regime pipeline.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (monthly_cy, regimes) — both shaped (months × commodities).
    """
    logger.info("Building convenience yield pipeline…")
    curves = _load_curves()
    rf = _load_risk_free()
    storage = estimate_storage_cost(curves)
    daily_cy = compute_convenience_yield(curves, rf, storage)
    monthly_cy = monthly_convenience_yield(daily_cy)
    regimes = classify_regime(monthly_cy)
    return monthly_cy, regimes


# ---------------------------------------------------------------------------
# Chart 07 — Convenience Yield time series (CL, NG, GC)
# ---------------------------------------------------------------------------


def plot_convenience_yield() -> Path:
    """Plot Chart 07: convenience yield time series with regime shading.

    Three vertically-stacked subplots (CL, NG, GC) share the x-axis.
    Background panels are shaded by the 5 curve regimes.

    Returns
    -------
    Path
        Saved PNG path.
    """
    monthly_cy, regimes = _build_pipeline()

    SPOTLIGHT = ["CL", "NG", "GC"]
    COLORS = {"CL": ACCENT, "NG": UP_COLOR, "GC": "#6894be"}
    SYM_LABELS = {"CL": "WTI Crude (CL)", "NG": "Natural Gas (NG)", "GC": "Gold (GC)"}

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 6), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    fig.patch.set_facecolor(BG_COLOR)

    # Build regime legend patches (shared across all subplots)
    legend_patches = [
        Patch(facecolor=_REGIME_COLORS[r], label=_REGIME_LABELS[r], linewidth=0)
        for r in _REGIME_ORDER
    ]

    for ax, sym in zip(axes, SPOTLIGHT):
        ax.set_facecolor(PAPER)

        cy_series = monthly_cy[sym].dropna() if sym in monthly_cy.columns else pd.Series(dtype=float)
        regime_series = regimes[sym] if sym in regimes.columns else pd.Series(dtype=object)

        # Draw regime background shading
        if not regime_series.dropna().empty:
            for date, regime in regime_series.dropna().items():
                color = _REGIME_COLORS.get(str(regime), PAPER)
                # Shade a 1-month span centred on the month-end date
                start = date - pd.offsets.MonthBegin(1)
                ax.axvspan(start, date, color=color, alpha=0.45, linewidth=0)

        # Plot the CY line
        if not cy_series.empty:
            ax.plot(
                cy_series.index,
                cy_series.values,
                color=COLORS[sym],
                linewidth=1.2,
                label=SYM_LABELS[sym],
            )
            ax.axhline(0, color=FG_COLOR, linewidth=0.5, alpha=0.3, linestyle="--")

        ax.set_ylabel("CY (ann.)", color=FG_COLOR, fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)

        # Inline symbol label at right margin
        ax.text(
            0.995, 0.88, SYM_LABELS[sym],
            transform=ax.transAxes,
            ha="right", va="top",
            color=COLORS[sym], fontsize=9,
        )

    # Regime legend on the bottom subplot
    axes[-1].legend(
        handles=legend_patches,
        loc="lower left",
        ncol=5,
        fontsize=7.5,
        framealpha=0.6,
        handlelength=1.0,
        handleheight=0.8,
    )

    fig.suptitle(
        "Commodity Convenience Yield — Regime Classification",
        color=FG_COLOR, fontsize=12, y=1.01,
    )

    return savefig(fig, "07_convenience_yield")


# ---------------------------------------------------------------------------
# Chart 08 — Curve regime heatmap (19 commodities × quarterly time)
# ---------------------------------------------------------------------------


def plot_curve_regime_heatmap() -> Path:
    """Plot Chart 08: curve regime heatmap across all commodities and time.

    Regimes are resampled to quarterly frequency (mode per quarter) for
    readability. A ListedColormap encodes the 5 regimes; a legend replaces
    the colorbar.

    Returns
    -------
    Path
        Saved PNG path.
    """
    _, regimes = _build_pipeline()

    # Restrict to commodities that appear in _ALL_SYMS and have data
    available_syms = [s for s in _ALL_SYMS if s in regimes.columns]

    # Map regimes to integers (explicit map avoids FutureWarning from replace)
    int_regimes = regimes[available_syms].apply(lambda col: col.map(_REGIME_INT))

    # Resample to quarterly using mode
    def _quarterly_mode(series: pd.Series) -> pd.Series:
        """Return modal (most-frequent) value per quarter."""
        return series.resample("QE").apply(
            lambda x: x.mode().iloc[0] if not x.dropna().empty else np.nan
        )

    quarterly = int_regimes.apply(_quarterly_mode)

    # Drop rows where all values are NaN
    quarterly = quarterly.dropna(how="all")

    # Build the colormap
    cmap_colors = [_REGIME_COLORS[r] for r in _REGIME_ORDER]
    cmap = ListedColormap(cmap_colors)

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PAPER)

    # Keep one label per year for readability
    tick_positions = []
    tick_labels_shown = []
    seen_years: set[int] = set()
    for i, d in enumerate(quarterly.index):
        yr = d.year
        if yr not in seen_years:
            seen_years.add(yr)
            tick_positions.append(i)
            tick_labels_shown.append(str(yr))

    sns.heatmap(
        quarterly.T.astype(float),
        ax=ax,
        cmap=cmap,
        vmin=-0.5,
        vmax=4.5,
        linewidths=0.3,
        linecolor="#0a0b0d",
        cbar=False,
        xticklabels=False,
        yticklabels=True,
    )

    # Apply year tick labels manually
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_shown, rotation=45, ha="right", fontsize=8, color=FG_COLOR)
    ax.tick_params(axis="y", labelsize=8, colors=FG_COLOR)
    ax.set_xlabel("Quarter", color=FG_COLOR, fontsize=9)
    ax.set_ylabel("Commodity", color=FG_COLOR, fontsize=9)

    # Regime legend
    legend_patches = [
        Patch(facecolor=_REGIME_COLORS[r], label=_REGIME_LABELS[r], linewidth=0)
        for r in _REGIME_ORDER
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        framealpha=0.6,
        fontsize=8,
        handlelength=1.2,
        handleheight=1.0,
    )

    fig.suptitle(
        "Curve Regime Classification — All Commodities",
        color=FG_COLOR, fontsize=12, y=1.01,
    )

    return savefig(fig, "08_curve_regime_heatmap")


# ---------------------------------------------------------------------------
# generate_all
# ---------------------------------------------------------------------------


def generate_all() -> list[Path]:
    """Generate all curve visualization charts (07, 08).

    Returns
    -------
    list[Path]
        Paths to the saved PNG files.
    """
    setup()
    paths: list[Path] = []
    logger.info("Generating Chart 07 — Convenience Yield time series")
    paths.append(plot_convenience_yield())
    logger.info("Generating Chart 08 — Curve Regime Heatmap")
    paths.append(plot_curve_regime_heatmap())
    logger.info("Curve charts complete: %s", [p.name for p in paths])
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_all()
