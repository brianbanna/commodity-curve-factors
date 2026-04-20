"""Performance charts for the commodity curve factors backtest.

Generates 7 publication-quality figures:

* 01 — Cumulative returns (all strategies + benchmarks)
* 02 — Drawdown from peak
* 03 — Monthly returns heatmap (equal-weight long)
* 04 — Rolling 252-day Sharpe ratio
* 11 — Cost sensitivity (Sharpe vs transaction cost)
* 12 — IS vs OOS Sharpe grouped bars
* 15 — Performance summary table figure

Call ``generate_all()`` to produce all charts in one shot.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from commodity_curve_factors.evaluation.attribution import rolling_sharpe
from commodity_curve_factors.utils.paths import DATA_PROCESSED, RESULTS_TABLES
from commodity_curve_factors.visualization.style import (
    ACCENT,
    BG_COLOR,
    BENCHMARK_COLORS,
    DOWN_COLOR,
    FG_COLOR,
    MID_COLOR,
    PAPER,
    STRATEGY_COLORS,
    STRATEGY_LABELS,
    add_crisis_shading,
    add_is_oos_divider,
    savefig,
)

logger = logging.getLogger(__name__)

_BACKTEST_DIR = DATA_PROCESSED / "backtest"

# ---------------------------------------------------------------------------
# Strategy list used across multiple charts (ordered for legend readability)
# ---------------------------------------------------------------------------
_STRATEGY_FILES: dict[str, str] = {
    "tsmom": "tsmom.parquet",
    "tsi": "tsi.parquet",
    "xs_carry": "xs_carry.parquet",
    "multi_factor_ew": "multi_factor_ew.parquet",
    "regime_conditioned": "regime_conditioned.parquet",
}

# Map strategy name → colour (falls back to a muted bone shade)
_FALLBACK_COLORS: dict[str, str] = {
    "xs_carry": "#eae6deb3",
    "multi_factor_ew": "#eae6de99",
    "regime_conditioned": "#eae6de80",
}


def _get_color(name: str) -> str:
    """Return the chart colour for *name* (strategy or benchmark key)."""
    if name in STRATEGY_COLORS:
        return STRATEGY_COLORS[name]
    if name in BENCHMARK_COLORS:
        return BENCHMARK_COLORS[name]
    return _FALLBACK_COLORS.get(name, "#eae6de66")


def _get_label(name: str) -> str:
    """Return the display label for *name*, capitalising unknown keys."""
    if name in STRATEGY_LABELS:
        return STRATEGY_LABELS[name]
    return name.replace("_", " ").title()


def _load_strategy(filename: str) -> pd.DataFrame:
    """Load a strategy parquet from the backtest directory."""
    path = _BACKTEST_DIR / filename
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def _load_benchmarks() -> pd.DataFrame:
    """Load benchmark parquet (columns: equal_weight_long, cash, SPY, AGG)."""
    df = pd.read_parquet(_BACKTEST_DIR / "benchmarks.parquet")
    df.index = pd.to_datetime(df.index)
    return df


def _cumulative_from_returns(returns: pd.Series) -> pd.Series:
    """Compute cumulative wealth index from daily return series (base = 1.0)."""
    return (1 + returns.fillna(0)).cumprod()


# ---------------------------------------------------------------------------
# Chart 01 — Cumulative returns
# ---------------------------------------------------------------------------


def plot_cumulative_returns() -> Path:
    """Chart 01: Cumulative returns for all strategies + benchmarks.

    Hero line is TSMOM (ACCENT, lw=2.0); TSI in DOWN_COLOR (lw=1.6);
    other strategies in bone with low opacity. Equal-weight long benchmark
    is overlaid for comparison. Crisis shading and IS/OOS divider included.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    benchmarks = _load_benchmarks()
    ew_returns = benchmarks["equal_weight_long"].dropna()
    ew_cumulative = _cumulative_from_returns(ew_returns)
    ax.plot(
        ew_cumulative.index,
        ew_cumulative.values,
        color=BENCHMARK_COLORS["equal_weight_long"],
        linewidth=1.4,
        label=_get_label("equal_weight_long"),
        zorder=3,
    )

    # Other strategies (lower prominence)
    secondary = ["xs_carry", "multi_factor_ew", "regime_conditioned"]
    for name in secondary:
        try:
            df = _load_strategy(_STRATEGY_FILES[name])
            cumulative = df["cumulative"]
            ax.plot(
                cumulative.index,
                cumulative.values,
                color=_get_color(name),
                linewidth=0.9,
                alpha=0.55,
                label=_get_label(name),
                zorder=2,
            )
        except Exception:
            logger.warning("Could not load strategy %s for chart 01", name)

    # TSI — highlighted secondary
    try:
        tsi = _load_strategy("tsi.parquet")
        ax.plot(
            tsi.index,
            tsi["cumulative"].values,
            color=DOWN_COLOR,
            linewidth=1.6,
            label=_get_label("tsi"),
            zorder=4,
        )
    except Exception:
        logger.warning("Could not load tsi for chart 01")

    # TSMOM — hero line
    try:
        tsmom = _load_strategy("tsmom.parquet")
        ax.plot(
            tsmom.index,
            tsmom["cumulative"].values,
            color=ACCENT,
            linewidth=2.0,
            label=_get_label("tsmom"),
            zorder=5,
        )
    except Exception:
        logger.warning("Could not load tsmom for chart 01")

    add_crisis_shading(ax)
    add_is_oos_divider(ax)

    ax.axhline(1.0, color=FG_COLOR, linewidth=0.4, alpha=0.3, linestyle=":")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}×"))
    ax.set_title("Cumulative Returns — All Strategies vs Equal-Weight Long")
    ax.set_xlabel("")
    ax.set_ylabel("Wealth Index")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    return savefig(fig, "01_cumulative_returns")


# ---------------------------------------------------------------------------
# Chart 02 — Drawdown
# ---------------------------------------------------------------------------


def plot_drawdown() -> Path:
    """Chart 02: Drawdown from peak for key strategies + EW Long benchmark.

    Filled area with alpha=0.25 per strategy for readability.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    benchmarks = _load_benchmarks()
    ew_returns = benchmarks["equal_weight_long"].dropna()
    ew_cum = _cumulative_from_returns(ew_returns)
    ew_dd = ew_cum / ew_cum.cummax() - 1

    color_ew = BENCHMARK_COLORS["equal_weight_long"]
    ax.fill_between(ew_dd.index, ew_dd.values, 0, color=color_ew, alpha=0.25)
    ax.plot(ew_dd.index, ew_dd.values, color=color_ew, linewidth=1.2,
            label=_get_label("equal_weight_long"))

    strategy_keys = ["tsmom", "tsi", "xs_carry", "multi_factor_ew"]
    for name in strategy_keys:
        try:
            df = _load_strategy(_STRATEGY_FILES[name])
            dd = df["drawdown"]
            color = _get_color(name)
            ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.25)
            ax.plot(dd.index, dd.values, color=color, linewidth=1.0,
                    label=_get_label(name))
        except Exception:
            logger.warning("Could not load strategy %s for chart 02", name)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Drawdown from Peak")
    ax.set_xlabel("")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    return savefig(fig, "02_drawdown")


# ---------------------------------------------------------------------------
# Chart 03 — Monthly returns heatmap
# ---------------------------------------------------------------------------


def plot_monthly_heatmap() -> Path:
    """Chart 03: Monthly returns heatmap for equal-weight long benchmark.

    Layout: years on the y-axis, months on the x-axis. Uses
    ``editorial_diverging`` colormap registered by the style module.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    benchmarks = _load_benchmarks()
    ew_returns = benchmarks["equal_weight_long"].dropna()

    monthly = (
        (1 + ew_returns)
        .resample("ME")
        .prod()
        .subtract(1)
    )
    monthly.index = monthly.index.to_period("M")
    pivot = monthly.rename_axis("Period").reset_index()
    pivot["Year"] = pivot["Period"].dt.year
    pivot["Month"] = pivot["Period"].dt.month
    heatmap_df = pivot.pivot(index="Year", columns="Month", values="equal_weight_long")
    heatmap_df.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    heatmap_df = heatmap_df.sort_index(ascending=False)

    vmax = max(abs(heatmap_df.values[~np.isnan(heatmap_df.values)])) if heatmap_df.size else 0.12
    vmax = max(vmax, 0.04)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        heatmap_df * 100,
        ax=ax,
        cmap="editorial_diverging",
        center=0,
        vmin=-vmax * 100,
        vmax=vmax * 100,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 7, "color": FG_COLOR},
        linewidths=0.3,
        linecolor=BG_COLOR,
        cbar_kws={"label": "Monthly Return (%)", "shrink": 0.6},
    )
    ax.set_title("Monthly Returns — Equal-Weight Long Benchmark (%)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()

    return savefig(fig, "03_monthly_heatmap")


# ---------------------------------------------------------------------------
# Chart 04 — Rolling Sharpe
# ---------------------------------------------------------------------------


def plot_rolling_sharpe() -> Path:
    """Chart 04: 252-day rolling Sharpe for TSI, TSMOM, and EW Long.

    Includes IS/OOS divider.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    benchmarks = _load_benchmarks()
    ew_returns = benchmarks["equal_weight_long"].dropna()
    ew_rs = rolling_sharpe(ew_returns, window=252)
    ax.plot(ew_rs.index, ew_rs.values, color=BENCHMARK_COLORS["equal_weight_long"],
            linewidth=1.2, label=_get_label("equal_weight_long"))

    pairs = [("tsi", "tsi.parquet"), ("tsmom", "tsmom.parquet")]
    for name, fname in pairs:
        try:
            df = _load_strategy(fname)
            rs = rolling_sharpe(df["net_return"], window=252)
            ax.plot(rs.index, rs.values, color=_get_color(name),
                    linewidth=1.4, label=_get_label(name))
        except Exception:
            logger.warning("Could not compute rolling Sharpe for %s", name)

    ax.axhline(0, color=FG_COLOR, linewidth=0.5, alpha=0.35, linestyle=":")
    add_is_oos_divider(ax)

    ax.set_title("Rolling 252-Day Sharpe Ratio")
    ax.set_xlabel("")
    ax.set_ylabel("Sharpe (annualised)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    return savefig(fig, "04_rolling_sharpe")


# ---------------------------------------------------------------------------
# Chart 11 — Cost sensitivity
# ---------------------------------------------------------------------------


def plot_cost_sensitivity() -> Path:
    """Chart 11: Sharpe ratio vs transaction cost (bps round-trip).

    Reads ``data/processed/backtest/cost_sensitivity.parquet``.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    df = pd.read_parquet(_BACKTEST_DIR / "cost_sensitivity.parquet")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["cost_bps"], df["sharpe"], color=ACCENT, linewidth=2.0, marker="o",
            markersize=5, markerfacecolor=PAPER, markeredgecolor=ACCENT, markeredgewidth=1.5)
    ax.axhline(0, color=FG_COLOR, linewidth=0.5, alpha=0.35, linestyle=":")

    # Annotate each point with its Sharpe value
    for _, row in df.iterrows():
        ax.annotate(
            f"{row['sharpe']:.2f}",
            xy=(row["cost_bps"], row["sharpe"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=FG_COLOR,
            alpha=0.7,
        )

    ax.set_title("Cost Sensitivity — Sharpe Ratio vs Transaction Cost")
    ax.set_xlabel("Round-Trip Transaction Cost (bps)")
    ax.set_ylabel("Annualised Sharpe Ratio")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d bps"))
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    return savefig(fig, "11_cost_sensitivity")


# ---------------------------------------------------------------------------
# Chart 12 — IS vs OOS Sharpe comparison
# ---------------------------------------------------------------------------


def plot_is_oos_comparison() -> Path:
    """Chart 12: Grouped bar chart — IS Sharpe vs OOS Sharpe for all strategies.

    IS bars in ACCENT; OOS bars in DOWN_COLOR. Benchmarks shown separately
    with reduced opacity.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    summary = pd.read_parquet(RESULTS_TABLES / "performance_summary.parquet")

    # Separate active strategies from benchmarks
    is_benchmark = summary["strategy"].str.startswith("BM_")
    active = summary[~is_benchmark].copy()
    benchmarks = summary[is_benchmark].copy()

    def _short_label(s: str) -> str:
        label_map = {
            "xs_carry": "XS Carry",
            "multi_factor_ew": "Multi EW",
            "multi_factor_ic": "Multi IC",
            "regime_conditioned": "Regime",
            "sector_neutral": "Sector Ntrl",
            "tsmom": "TSMOM",
            "calendar_spread": "Cal Spd",
            "tsi": "TSI",
        }
        return label_map.get(s, s.replace("_", " ").title())

    def _short_bm_label(s: str) -> str:
        bm_map = {
            "BM_equal_weight_long": "EW Long",
            "BM_cash": "Cash",
            "BM_SPY": "SPY",
            "BM_AGG": "AGG",
        }
        return bm_map.get(s, s.replace("BM_", ""))

    all_labels = [_short_label(s) for s in active["strategy"]] + [
        _short_bm_label(s) for s in benchmarks["strategy"]
    ]
    all_is = list(active["is_sharpe"]) + list(benchmarks["is_sharpe"])
    all_oos = list(active["oos_sharpe"]) + list(benchmarks["oos_sharpe"])

    n = len(all_labels)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))

    # Active strategies at full opacity, benchmarks at reduced opacity
    n_active = len(active)
    is_colors = [ACCENT] * n_active + [ACCENT + "80"] * len(benchmarks)
    oos_colors = [DOWN_COLOR] * n_active + [DOWN_COLOR + "80"] * len(benchmarks)

    for i, (is_val, oos_val, is_c, oos_c) in enumerate(
        zip(all_is, all_oos, is_colors, oos_colors)
    ):
        ax.bar(x[i] - width / 2, is_val, width, color=is_c, label="In-Sample" if i == 0 else "")
        ax.bar(x[i] + width / 2, oos_val, width, color=oos_c,
               label="Out-of-Sample" if i == 0 else "")

    ax.axhline(0, color=FG_COLOR, linewidth=0.5, alpha=0.35, linestyle=":")

    # Divider between active strategies and benchmarks
    if n_active > 0 and len(benchmarks) > 0:
        ax.axvline(n_active - 0.5, color=FG_COLOR, linewidth=0.6, alpha=0.3, linestyle="--")
        ax.text(
            n_active - 0.5 + 0.05,
            ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 0.5,
            "Benchmarks →",
            color=FG_COLOR,
            fontsize=7,
            alpha=0.5,
            va="top",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=8)
    ax.set_title("In-Sample vs Out-of-Sample Sharpe Ratio")
    ax.set_ylabel("Annualised Sharpe Ratio")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()

    return savefig(fig, "12_is_oos_comparison")


# ---------------------------------------------------------------------------
# Chart 15 — Performance table figure
# ---------------------------------------------------------------------------


def plot_performance_table() -> Path:
    """Chart 15: Performance summary rendered as a matplotlib table figure.

    Displays IS/OOS Sharpe, CAGR, max-drawdown, and turnover for each
    strategy alongside the benchmarks.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    summary = pd.read_parquet(RESULTS_TABLES / "performance_summary.parquet")

    display_cols = {
        "strategy": "Strategy",
        "is_sharpe": "IS Sharpe",
        "oos_sharpe": "OOS Sharpe",
        "full_sharpe": "Full Sharpe",
        "is_cagr": "IS CAGR",
        "oos_cagr": "OOS CAGR",
        "is_max_dd": "IS Max DD",
        "oos_max_dd": "OOS Max DD",
        "turnover": "Turnover",
    }
    table_df = summary[[c for c in display_cols if c in summary.columns]].copy()
    table_df = table_df.rename(columns=display_cols)

    def _fmt_strategy(s: str) -> str:
        if s.startswith("BM_"):
            s = s[3:]
        return s.replace("_", " ").title()

    table_df["Strategy"] = table_df["Strategy"].apply(_fmt_strategy)

    def _pct(v: float) -> str:
        return f"{v * 100:.1f}%" if not np.isnan(v) else "—"

    def _sharpe(v: float) -> str:
        return f"{v:.2f}" if not np.isnan(v) else "—"

    for col in ["IS Sharpe", "OOS Sharpe", "Full Sharpe"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].apply(_sharpe)

    for col in ["IS CAGR", "OOS CAGR", "IS Max DD", "OOS Max DD"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].apply(lambda v: _pct(float(v)))

    if "Turnover" in table_df.columns:
        table_df["Turnover"] = table_df["Turnover"].apply(
            lambda v: f"{float(v):.1%}" if not (isinstance(v, float) and np.isnan(v)) else "—"
        )

    n_rows, n_cols = table_df.shape
    fig_h = max(4.0, 0.45 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")

    col_widths = [0.20] + [0.10] * (n_cols - 1)

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns.tolist(),
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    # Style header row
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor(ACCENT + "30")
        cell.set_text_props(color=FG_COLOR, fontweight="bold")
        cell.set_edgecolor(PAPER)

    # Style data rows
    is_benchmark = [s.startswith("Bm ") or s in ("Spy", "Agg", "Cash", "Ew Long",
                    "Equal Weight Long") for s in table_df["Strategy"].tolist()]
    for i in range(n_rows):
        row_bg = PAPER if i % 2 == 0 else BG_COLOR
        for j in range(n_cols):
            cell = tbl[i + 1, j]
            cell.set_facecolor(row_bg)
            cell.set_text_props(color=FG_COLOR if not is_benchmark[i] else MID_COLOR)
            cell.set_edgecolor(PAPER)

    ax.set_title("Performance Summary", pad=12, fontsize=13, color=FG_COLOR)
    fig.tight_layout()

    return savefig(fig, "15_performance_table")


# ---------------------------------------------------------------------------
# generate_all
# ---------------------------------------------------------------------------


def generate_all() -> list[Path]:
    """Generate all 7 performance charts.

    Catches exceptions per function so a single failure does not prevent
    the remaining charts from being produced.

    Returns
    -------
    list[Path]
        Paths of successfully saved figures.
    """
    tasks = [
        ("plot_cumulative_returns", plot_cumulative_returns),
        ("plot_drawdown", plot_drawdown),
        ("plot_monthly_heatmap", plot_monthly_heatmap),
        ("plot_rolling_sharpe", plot_rolling_sharpe),
        ("plot_cost_sensitivity", plot_cost_sensitivity),
        ("plot_is_oos_comparison", plot_is_oos_comparison),
        ("plot_performance_table", plot_performance_table),
    ]

    paths: list[Path] = []
    for name, fn in tasks:
        try:
            path = fn()
            paths.append(path)
            logger.info("Generated chart: %s → %s", name, path)
        except Exception:
            logger.exception("Failed to generate chart: %s", name)

    return paths
