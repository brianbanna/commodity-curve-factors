"""Risk charts for the commodity curve factors backtest.

Generates 3 publication-quality figures:

* 10 — Stress test cumulative returns (grouped bars per crisis period)
* 13 — Return attribution by sector for EW Long benchmark
* 14 — Bootstrap Sharpe 95% CI for key strategies (horizontal bars)

Call ``generate_all()`` to produce all charts in one shot.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from commodity_curve_factors.data.futures_loader import load_front_month_data
from commodity_curve_factors.evaluation.bootstrap import bootstrap_sharpe_ci
from commodity_curve_factors.utils.constants import SECTORS
from commodity_curve_factors.utils.paths import DATA_PROCESSED, RESULTS_TABLES
from commodity_curve_factors.visualization.style import (
    ACCENT,
    BENCHMARK_COLORS,
    DOWN_COLOR,
    FG_COLOR,
    MID_COLOR,
    STRATEGY_COLORS,
    STRATEGY_LABELS,
    UP_COLOR,
    savefig,
)

logger = logging.getLogger(__name__)

_BACKTEST_DIR = DATA_PROCESSED / "backtest"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FALLBACK_COLORS: dict[str, str] = {
    "xs_carry": "#eae6deb3",
    "multi_factor_ew": "#eae6de99",
    "multi_factor_ic": "#eae6de80",
    "regime_conditioned": "#eae6de66",
    "sector_neutral": "#eae6de50",
    "calendar_spread": "#eae6de40",
}

_SECTOR_COLORS: list[str] = [ACCENT, UP_COLOR, MID_COLOR, DOWN_COLOR, "#6894be"]


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


# ---------------------------------------------------------------------------
# Chart 10 — Stress test results
# ---------------------------------------------------------------------------

# Map raw period keys to display labels
_PERIOD_LABELS: dict[str, str] = {
    "oil_crash_2008": "Oil Crash 2008",
    "oil_glut_2014": "Oil Glut 2014",
    "covid_negative_wti": "COVID / Neg WTI",
    "energy_spike_2022": "Energy Spike 2022",
}

# Strategies to highlight in the stress test chart
_STRESS_STRATEGIES: list[str] = ["tsmom", "tsi", "equal_weight_long"]

# Bar colours per strategy slot (ordered to match _STRESS_STRATEGIES)
_STRESS_COLORS: list[str] = [
    STRATEGY_COLORS["tsmom"],
    STRATEGY_COLORS["tsi"],
    BENCHMARK_COLORS["equal_weight_long"],
]


def plot_stress_test() -> Path:
    """Chart 10: Stress-test cumulative returns per crisis period (grouped bars).

    Loads ``results/tables/stress_tests.parquet`` and plots one grouped bar
    cluster per crisis period, with one bar per selected strategy.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    df = pd.read_parquet(RESULTS_TABLES / "stress_tests.parquet")

    periods = df["period"].unique().tolist()

    # Build a pivot: rows = period, cols = strategy
    pivot = df.pivot_table(
        index="period", columns="strategy", values="cumulative_return", aggfunc="first"
    )

    # Filter to strategies that exist in the data
    available = [s for s in _STRESS_STRATEGIES if s in pivot.columns]
    pivot = pivot[available]

    n_periods = len(periods)
    n_strategies = len(available)
    x = np.arange(n_periods)
    total_width = 0.72
    bar_width = total_width / n_strategies

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, strategy in enumerate(available):
        offsets = x + (i - (n_strategies - 1) / 2) * bar_width
        values = [pivot.loc[p, strategy] * 100 if p in pivot.index else np.nan for p in periods]
        color = _STRESS_COLORS[_STRESS_STRATEGIES.index(strategy)]
        bars = ax.bar(
            offsets,
            values,
            bar_width * 0.88,
            color=color,
            label=_get_label(strategy),
            zorder=3,
        )
        # Value labels on top of each bar
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                y_pos = bar.get_height() + (0.4 if val >= 0 else -1.8)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_pos,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=FG_COLOR,
                    alpha=0.8,
                )

    ax.axhline(0, color=FG_COLOR, linewidth=0.5, alpha=0.35, linestyle=":")
    period_display = [_PERIOD_LABELS.get(p, p.replace("_", " ").title()) for p in periods]
    ax.set_xticks(x)
    ax.set_xticklabels(period_display, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_title("Stress Test — Cumulative Return by Crisis Period")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()

    return savefig(fig, "10_stress_test")


# ---------------------------------------------------------------------------
# Chart 13 — Sector attribution
# ---------------------------------------------------------------------------


def plot_sector_attribution() -> Path:
    """Chart 13: Annualised return attribution by sector for EW Long benchmark.

    Loads front-month futures, computes equal-weight daily returns per
    commodity, groups by sector, and plots the annualised percentage
    contribution of each sector.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    raw: dict[str, pd.DataFrame] = load_front_month_data()

    # Compute daily log returns per commodity
    return_series: dict[str, pd.Series] = {}
    for ticker, ohlcv in raw.items():
        close = ohlcv["Close"] if "Close" in ohlcv.columns else ohlcv.iloc[:, 3]
        ret = np.log(close / close.shift(1)).dropna()
        return_series[ticker] = ret

    # Determine N (total number of commodities present in our data)
    all_tickers = list(return_series.keys())
    n_total = len(all_tickers)
    if n_total == 0:
        raise ValueError("No front-month data loaded — cannot compute sector attribution.")

    ew_weight = 1.0 / n_total

    # Aggregate contribution per sector
    sector_contribs: dict[str, float] = {}
    for sector, tickers in SECTORS.items():
        contrib_sum = pd.Series(dtype=float)
        for ticker in tickers:
            if ticker in return_series:
                weighted = return_series[ticker] * ew_weight
                if contrib_sum.empty:
                    contrib_sum = weighted
                else:
                    contrib_sum = contrib_sum.add(weighted, fill_value=0.0)
        if contrib_sum.empty:
            sector_contribs[sector] = 0.0
        else:
            # Annualised sector contribution as percentage
            sector_contribs[sector] = contrib_sum.mean() * 252 * 100

    sectors = list(sector_contribs.keys())
    values = [sector_contribs[s] for s in sectors]
    colors = [_SECTOR_COLORS[i % len(_SECTOR_COLORS)] for i in range(len(sectors))]
    display_labels = [s.title() for s in sectors]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(sectors, values, color=colors, width=0.6, zorder=3)

    # Value labels on top of each bar
    for bar, val, label in zip(bars, values, display_labels):
        y_pos = val + (0.01 if val >= 0 else -0.04)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color=FG_COLOR,
            alpha=0.85,
        )

    ax.axhline(0, color=FG_COLOR, linewidth=0.5, alpha=0.35, linestyle=":")
    ax.set_xticks(range(len(sectors)))
    ax.set_xticklabels(display_labels, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    ax.set_title("Return Attribution by Sector — Equal-Weight Long Benchmark")
    ax.set_ylabel("Annualised Contribution (%)")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()

    return savefig(fig, "13_sector_attribution")


# ---------------------------------------------------------------------------
# Chart 14 — Bootstrap Sharpe CI
# ---------------------------------------------------------------------------

# Ordered list of strategies/benchmarks to include in CI chart
_CI_STRATEGIES: list[tuple[str, str | None]] = [
    ("tsmom", "tsmom.parquet"),
    ("tsi", "tsi.parquet"),
    ("xs_carry", "xs_carry.parquet"),
    ("multi_factor_ew", "multi_factor_ew.parquet"),
    ("equal_weight_long", None),  # loaded from benchmarks.parquet
]


def plot_bootstrap_ci() -> Path:
    """Chart 14: Bootstrap 95% CI for annualised Sharpe of key strategies.

    Horizontal bars show the confidence interval; a marker indicates the
    point estimate. A dashed vertical line marks Sharpe = 0.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    benchmarks = pd.read_parquet(_BACKTEST_DIR / "benchmarks.parquet")
    benchmarks.index = pd.to_datetime(benchmarks.index)

    results: list[tuple[str, float, float, float, str]] = []
    for name, filename in _CI_STRATEGIES:
        try:
            if filename is not None:
                df = pd.read_parquet(_BACKTEST_DIR / filename)
                df.index = pd.to_datetime(df.index)
                returns = df["net_return"].dropna()
            else:
                # EW Long benchmark
                returns = benchmarks["equal_weight_long"].dropna()

            point, lo, hi = bootstrap_sharpe_ci(returns)
            color = _get_color(name)
            results.append((name, point, lo, hi, color))
            logger.debug("Bootstrap CI for %s: %.3f [%.3f, %.3f]", name, point, lo, hi)
        except Exception:
            logger.warning("Could not compute bootstrap CI for %s", name, exc_info=True)

    if not results:
        raise RuntimeError("No bootstrap CI results computed — check backtest files.")

    n = len(results)
    y_pos = np.arange(n)
    labels = [_get_label(name) for name, *_ in results]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (name, point, lo, hi, color) in enumerate(results):
        ci_width = hi - lo
        ax.barh(
            i,
            ci_width,
            left=lo,
            height=0.55,
            color=color,
            alpha=0.45,
            zorder=2,
        )
        # Point estimate marker
        ax.scatter(
            point,
            i,
            color=color,
            s=55,
            zorder=4,
            linewidths=0,
        )
        # Annotate point estimate
        ax.text(
            hi + 0.04,
            i,
            f"{point:.2f}",
            va="center",
            ha="left",
            fontsize=8,
            color=FG_COLOR,
            alpha=0.8,
        )

    ax.axvline(0, color=DOWN_COLOR, linewidth=1.0, linestyle="--", alpha=0.7, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Bootstrap 95% CI — Annualised Sharpe Ratio")
    ax.set_xlabel("Annualised Sharpe Ratio")
    ax.grid(True, axis="x", alpha=0.4)
    fig.tight_layout()

    return savefig(fig, "14_bootstrap_ci")


# ---------------------------------------------------------------------------
# generate_all
# ---------------------------------------------------------------------------


def generate_all() -> list[Path]:
    """Generate all 3 risk charts.

    Catches exceptions per function so a single failure does not prevent
    the remaining charts from being produced.

    Returns
    -------
    list[Path]
        Paths of successfully saved figures.
    """
    tasks = [
        ("plot_stress_test", plot_stress_test),
        ("plot_sector_attribution", plot_sector_attribution),
        ("plot_bootstrap_ci", plot_bootstrap_ci),
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
