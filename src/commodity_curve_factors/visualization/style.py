"""Editorial dark-theme style foundation for all publication-quality charts.

All 15 visualization modules import constants and helpers from here.
Call ``setup()`` once at the top of each plotting script.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from commodity_curve_factors.utils.paths import RESULTS_FIGURES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Palette constants
# ---------------------------------------------------------------------------
BG_COLOR = "#0a0b0d"
PAPER = "#121214"
FG_COLOR = "#eae6de"
GRID_COLOR = "#2c2b28"
ACCENT = "#d4cec0"
UP_COLOR = "#8ca891"
MID_COLOR = "#c5b58c"
DOWN_COLOR = "#b87c6c"

DPI = 300

# ---------------------------------------------------------------------------
# Strategy / benchmark colours and labels
# ---------------------------------------------------------------------------
STRATEGY_COLORS: dict[str, str] = {
    "tsmom": ACCENT,
    "tsi": DOWN_COLOR,
    "carry": "#eae6decc",
    "xsmom": "#eae6deb3",
    "curve": "#eae6de99",
    "macro": "#eae6de80",
    "combined": "#eae6de66",
}

BENCHMARK_COLORS: dict[str, str] = {
    "equal_weight_long": "#eae6de8c",
    "spy": "#b87c6c80",
    "agg": "#eae6de30",
    "cash": "#eae6de20",
}

STRATEGY_LABELS: dict[str, str] = {
    "tsmom": "TS Momentum",
    "tsi": "Trend-Strength Index",
    "carry": "Carry",
    "xsmom": "XS Momentum",
    "curve": "Curve",
    "macro": "Macro",
    "combined": "Combined",
    "equal_weight_long": "Equal-Weight Long",
    "spy": "S&P 500",
    "agg": "US Agg Bond",
    "cash": "Cash",
}

# ---------------------------------------------------------------------------
# Crisis periods and IS/OOS split
# ---------------------------------------------------------------------------
CRISIS_PERIODS: dict[str, tuple[str, str]] = {
    "2008 Crash": ("2008-09-01", "2009-03-31"),
    "Oil Glut 2014": ("2014-06-01", "2016-01-31"),
    "COVID 2020": ("2020-02-01", "2020-05-31"),
    "Energy Spike 2022": ("2021-10-01", "2022-12-31"),
}

IS_OOS_SPLIT = "2017-12-31"

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIGURES_DIR: Path = RESULTS_FIGURES


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
_RCPARAMS: dict[str, object] = {
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": PAPER,
    "text.color": FG_COLOR,
    "axes.labelcolor": "#eae6de8c",
    "xtick.color": "#eae6de70",
    "ytick.color": "#eae6de70",
    "axes.edgecolor": GRID_COLOR,
    "grid.color": "#eae6de1e",
    "grid.alpha": 1.0,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "font.family": ["JetBrains Mono", "DejaVu Sans Mono", "monospace"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "normal",
    "axes.titlecolor": FG_COLOR,
    "axes.labelsize": 10,
    "legend.facecolor": PAPER,
    "legend.edgecolor": "#eae6de1e",
    "legend.labelcolor": FG_COLOR,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "savefig.facecolor": BG_COLOR,
    "savefig.edgecolor": "none",
}


def _register_colormaps() -> None:
    """Register editorial_diverging, editorial_warm, and editorial_cool colormaps."""
    specs: list[tuple[str, list[str]]] = [
        ("editorial_diverging", [DOWN_COLOR, "#1a1a1c", UP_COLOR]),
        ("editorial_warm", [PAPER, "#3a3530", "#7a5e52", DOWN_COLOR, ACCENT]),
        ("editorial_cool", [PAPER, "#2a2a2c", "#6e6a64", "#a8a59d", FG_COLOR]),
    ]
    for name, colors in specs:
        if name not in plt.colormaps():
            cmap = mpl.colors.LinearSegmentedColormap.from_list(name, colors, N=256)
            mpl.colormaps.register(cmap, name=name)
            logger.debug("Registered colormap: %s", name)


def setup() -> None:
    """Configure rcParams for editorial dark theme and register custom colormaps.

    Call once at the top of any plotting script or notebook before creating figures.
    """
    mpl.rcParams.update(_RCPARAMS)
    _register_colormaps()
    logger.debug("Visualization style applied.")


# ---------------------------------------------------------------------------
# Axes helpers
# ---------------------------------------------------------------------------

def add_crisis_shading(ax: plt.Axes, alpha: float = 0.08) -> None:
    """Add translucent vertical bands for known crisis periods.

    Parameters
    ----------
    ax:
        Matplotlib axes to annotate.
    alpha:
        Opacity of each shaded band.
    """
    for label, (start, end) in CRISIS_PERIODS.items():
        ax.axvspan(
            np.datetime64(start),
            np.datetime64(end),
            alpha=alpha,
            color=DOWN_COLOR,
            linewidth=0,
            label=label,
        )


def add_is_oos_divider(ax: plt.Axes) -> None:
    """Add a vertical dashed line at the in-sample / out-of-sample split date.

    Parameters
    ----------
    ax:
        Matplotlib axes to annotate.
    """
    ax.axvline(
        np.datetime64(IS_OOS_SPLIT),
        color=ACCENT,
        linewidth=0.8,
        linestyle="--",
        alpha=0.6,
        label="IS/OOS split",
    )


def savefig(fig: plt.Figure, name: str) -> Path:
    """Save *fig* to ``FIGURES_DIR/<name>.png`` at 300 DPI and close it.

    Parameters
    ----------
    fig:
        Matplotlib figure to save.
    name:
        Output filename stem (no extension).

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"{name}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", out)
    return out
