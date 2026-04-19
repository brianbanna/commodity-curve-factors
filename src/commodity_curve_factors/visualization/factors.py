"""Factor-level visualization charts for the commodity curve factors research.

Generates 2 publication-quality figures:

* 05 — Factor IC decay (grouped bars at lags 0, 1, 5, 10, 20)
* 06 — Factor correlation heatmap (10×10, lower triangle)

Call ``generate_all()`` to produce all charts in one shot.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from commodity_curve_factors.data.futures_loader import load_front_month_data
from commodity_curve_factors.evaluation.factor_analysis import ic_decay
from commodity_curve_factors.utils.paths import DATA_PROCESSED
from commodity_curve_factors.visualization.style import (
    ACCENT,
    BG_COLOR,
    DOWN_COLOR,
    FG_COLOR,
    MID_COLOR,
    PAPER,
    UP_COLOR,
    savefig,
    setup,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FACTORS_DIR = DATA_PROCESSED / "factors"

FACTOR_LABELS: dict[str, str] = {
    "carry": "Carry",
    "slope": "Slope",
    "curvature": "Curvature",
    "curve_momentum": "Curve Mom",
    "tsmom": "TSMOM",
    "xsmom": "XSMOM",
    "inventory": "Inventory",
    "positioning": "Positioning",
    "macro": "Macro",
    "volatility": "Volatility",
}

_FACTOR_NAMES: list[str] = list(FACTOR_LABELS.keys())

# Colours per lag: lag0 uses DOWN_COLOR (semi-transparent via alpha kwarg)
_LAG_COLORS: dict[int, str] = {
    0: DOWN_COLOR,
    1: ACCENT,
    5: UP_COLOR,
    10: MID_COLOR,
    20: "#eae6de50",
}

_LAGS: list[int] = [0, 1, 5, 10, 20]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_returns() -> pd.DataFrame:
    """Build a daily-returns DataFrame (dates × commodities) from front-month closes.

    Returns
    -------
    pd.DataFrame
        Percentage returns, index named ``trade_date``.
    """
    prices_dict = load_front_month_data()
    closes = pd.DataFrame({ticker: df["Close"] for ticker, df in prices_dict.items()})
    closes.index.name = "trade_date"
    returns = closes.pct_change(fill_method=None)
    return returns


def _load_factor(name: str) -> pd.DataFrame:
    """Load a single factor parquet file.

    Parameters
    ----------
    name:
        Factor name (key in ``FACTOR_LABELS``).

    Returns
    -------
    pd.DataFrame
        Factor values (dates × commodities).
    """
    path: Path = _FACTORS_DIR / f"{name}.parquet"
    if not path.exists():
        logger.warning("Factor file not found: %s", path)
        return pd.DataFrame()
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Chart 05 — Factor IC decay
# ---------------------------------------------------------------------------


def plot_factor_ic_decay() -> Path:
    """Grouped bar chart of factor IC at lags [0, 1, 5, 10, 20].

    The key research finding is that lag-0 IC is 3–4× lag-1 IC for curve
    factors (carry, slope, curvature, curve_momentum), indicating contamination
    when IC is computed without an execution lag.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    logger.info("Computing IC decay for %d factors at lags %s", len(_FACTOR_NAMES), _LAGS)

    returns = _load_returns()

    # Collect IC by factor and lag
    ic_table: dict[str, dict[int, float]] = {}
    for name in _FACTOR_NAMES:
        factor_df = _load_factor(name)
        if factor_df.empty:
            logger.warning("Skipping factor '%s' — empty DataFrame", name)
            ic_table[name] = {lag: 0.0 for lag in _LAGS}
            continue
        try:
            decay_df = ic_decay(factor_df, returns, lags=_LAGS)
            lag_to_ic = dict(zip(decay_df["lag"], decay_df["mean_ic"]))
            ic_table[name] = {lag: lag_to_ic.get(lag, 0.0) for lag in _LAGS}
        except Exception:
            logger.exception("ic_decay failed for factor '%s'", name)
            ic_table[name] = {lag: 0.0 for lag in _LAGS}

    # Build figure
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG_COLOR)
    ax.set_facecolor(PAPER)

    n_factors = len(_FACTOR_NAMES)
    n_lags = len(_LAGS)
    group_width = 0.8
    bar_width = group_width / n_lags
    x = np.arange(n_factors)

    for i, lag in enumerate(_LAGS):
        offsets = x - group_width / 2 + (i + 0.5) * bar_width
        values = [ic_table[name][lag] for name in _FACTOR_NAMES]
        color = _LAG_COLORS[lag]
        alpha = 0.4 if lag == 0 else 0.85
        ax.bar(
            offsets,
            values,
            width=bar_width * 0.92,
            color=color,
            alpha=alpha,
            label=f"Lag {lag}" if lag > 0 else "Lag 0 (contaminated)",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [FACTOR_LABELS[name] for name in _FACTOR_NAMES],
        rotation=30,
        ha="right",
        fontsize=9,
    )
    ax.axhline(0, color=FG_COLOR, linewidth=0.5, alpha=0.3)
    ax.set_ylabel("Mean IC (Spearman)", fontsize=10)
    ax.set_title("Factor IC Decay — Lag 0 Contamination vs. Lagged IC", fontsize=13)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    return savefig(fig, "05_factor_ic_decay")


# ---------------------------------------------------------------------------
# Chart 06 — Factor correlation heatmap
# ---------------------------------------------------------------------------


def plot_factor_correlation() -> Path:
    """10×10 factor correlation heatmap using cross-sectional mean time series.

    Each factor is collapsed to its cross-sectional mean across commodities
    per day, then pairwise Pearson correlations are computed. The lower
    triangle is shown using a masked seaborn heatmap.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    logger.info("Building factor correlation matrix for %d factors", len(_FACTOR_NAMES))

    series: dict[str, pd.Series] = {}
    for name in _FACTOR_NAMES:
        factor_df = _load_factor(name)
        if factor_df.empty:
            logger.warning("Skipping factor '%s' — empty DataFrame", name)
            continue
        series[name] = factor_df.mean(axis=1)

    if not series:
        logger.error("No factor series loaded; cannot build correlation matrix")
        raise RuntimeError("No valid factor data found for correlation chart.")

    combined = pd.DataFrame(series)
    combined.columns = [FACTOR_LABELS[c] for c in combined.columns]

    corr = combined.corr()

    # Lower-triangle mask (True = hide)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
    ax.set_facecolor(PAPER)

    sns.heatmap(
        corr,
        mask=mask,
        cmap="editorial_cool",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8, "color": FG_COLOR},
        linewidths=0.4,
        linecolor=BG_COLOR,
        ax=ax,
        cbar_kws={"shrink": 0.75, "pad": 0.02},
    )

    ax.set_title("Factor Correlation Matrix (Cross-Sectional Mean)", fontsize=13)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    # Style the colorbar
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(colors=FG_COLOR, labelsize=8)

    fig.tight_layout()
    return savefig(fig, "06_factor_correlation")


# ---------------------------------------------------------------------------
# generate_all
# ---------------------------------------------------------------------------


def generate_all() -> list[Path]:
    """Generate all factor visualization charts.

    Each chart is produced with independent exception handling so a failure in
    one does not prevent the others from being saved.

    Returns
    -------
    list[Path]
        Paths to successfully saved figures.
    """
    setup()
    paths: list[Path] = []

    for fn, label in [
        (plot_factor_ic_decay, "IC decay (Chart 05)"),
        (plot_factor_correlation, "factor correlation (Chart 06)"),
    ]:
        try:
            path = fn()
            paths.append(path)
            logger.info("Generated %s → %s", label, path.name)
        except Exception:
            logger.exception("Failed to generate %s", label)

    return paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    generated = generate_all()
    for p in generated:
        print(p)
