"""Tests for visualization style foundation module."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import commodity_curve_factors.visualization.style as style


# ---------------------------------------------------------------------------
# setup() — rcParams and colormaps
# ---------------------------------------------------------------------------


def test_setup_registers_colormaps() -> None:
    """setup() registers all three editorial colormaps."""
    style.setup()
    for name in ("editorial_diverging", "editorial_warm", "editorial_cool"):
        assert name in plt.colormaps(), f"Colormap '{name}' not registered after setup()"


def test_setup_sets_figure_facecolor() -> None:
    """setup() sets figure.facecolor to BG_COLOR."""
    import matplotlib as mpl

    style.setup()
    assert mpl.rcParams["figure.facecolor"] == style.BG_COLOR


def test_setup_sets_axes_facecolor() -> None:
    """setup() sets axes.facecolor to PAPER."""
    import matplotlib as mpl

    style.setup()
    assert mpl.rcParams["axes.facecolor"] == style.PAPER


def test_setup_sets_text_color() -> None:
    """setup() sets text.color to FG_COLOR."""
    import matplotlib as mpl

    style.setup()
    assert mpl.rcParams["text.color"] == style.FG_COLOR


# ---------------------------------------------------------------------------
# CRISIS_PERIODS — date validity
# ---------------------------------------------------------------------------


def test_crisis_periods_end_after_start() -> None:
    """Each crisis period has end date strictly after start date."""
    import pandas as pd

    for label, (start, end) in style.CRISIS_PERIODS.items():
        assert pd.Timestamp(end) > pd.Timestamp(start), (
            f"Crisis '{label}': end {end} not after start {start}"
        )


def test_crisis_periods_has_four_entries() -> None:
    """CRISIS_PERIODS contains exactly four entries."""
    assert len(style.CRISIS_PERIODS) == 4


# ---------------------------------------------------------------------------
# add_crisis_shading — adds patches
# ---------------------------------------------------------------------------


def test_add_crisis_shading_adds_patches() -> None:
    """add_crisis_shading() adds axvspan patches to the axes."""
    style.setup()
    fig, ax = plt.subplots()
    n_before = len(ax.patches)
    style.add_crisis_shading(ax)
    n_after = len(ax.patches)
    plt.close(fig)
    assert n_after > n_before


def test_add_crisis_shading_respects_alpha() -> None:
    """add_crisis_shading() uses the supplied alpha value."""
    style.setup()
    fig, ax = plt.subplots()
    style.add_crisis_shading(ax, alpha=0.15)
    for p in ax.patches:
        assert p.get_alpha() == pytest.approx(0.15)
    plt.close(fig)


# ---------------------------------------------------------------------------
# add_is_oos_divider — adds a line
# ---------------------------------------------------------------------------


def test_add_is_oos_divider_adds_line() -> None:
    """add_is_oos_divider() adds at least one vertical line to the axes."""
    style.setup()
    fig, ax = plt.subplots()
    n_before = len(ax.lines)
    style.add_is_oos_divider(ax)
    n_after = len(ax.lines)
    plt.close(fig)
    assert n_after > n_before


# ---------------------------------------------------------------------------
# savefig — creates a file
# ---------------------------------------------------------------------------


def test_savefig_creates_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """savefig() writes a PNG to FIGURES_DIR."""
    monkeypatch.setattr(style, "FIGURES_DIR", tmp_path)
    style.setup()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = style.savefig(fig, "test_output")
    assert out == tmp_path / "test_output.png"
    assert out.exists()


def test_savefig_closes_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """savefig() closes the figure after saving."""
    monkeypatch.setattr(style, "FIGURES_DIR", tmp_path)
    style.setup()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    style.savefig(fig, "test_close")
    assert not plt.fignum_exists(fig.number)


def test_savefig_creates_parent_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """savefig() creates FIGURES_DIR if it does not exist."""
    nested = tmp_path / "nested" / "figures"
    monkeypatch.setattr(style, "FIGURES_DIR", nested)
    style.setup()
    fig, _ = plt.subplots()
    style.savefig(fig, "nested_test")
    assert (nested / "nested_test.png").exists()


# ---------------------------------------------------------------------------
# Strategy / benchmark consistency
# ---------------------------------------------------------------------------


def test_strategy_colors_have_labels() -> None:
    """Every key in STRATEGY_COLORS has a matching entry in STRATEGY_LABELS."""
    for key in style.STRATEGY_COLORS:
        assert key in style.STRATEGY_LABELS, f"Strategy '{key}' missing from STRATEGY_LABELS"


def test_benchmark_colors_have_labels() -> None:
    """Every key in BENCHMARK_COLORS has a matching entry in STRATEGY_LABELS."""
    for key in style.BENCHMARK_COLORS:
        assert key in style.STRATEGY_LABELS, f"Benchmark '{key}' missing from STRATEGY_LABELS"
