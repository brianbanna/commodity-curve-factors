"""Smoke tests for visualization — require real data on disk."""

import matplotlib

matplotlib.use("Agg")

import pytest
from commodity_curve_factors.utils.paths import DATA_PROCESSED

pytestmark = pytest.mark.skipif(
    not (DATA_PROCESSED / "backtest" / "tsmom.parquet").exists(),
    reason="Backtest data not on disk",
)


@pytest.fixture(autouse=True)
def _setup_and_redirect(tmp_path, monkeypatch):
    from commodity_curve_factors.visualization.style import setup

    setup()
    monkeypatch.setattr(
        "commodity_curve_factors.visualization.style.FIGURES_DIR", tmp_path
    )


def test_performance_cumulative_returns():
    from commodity_curve_factors.visualization.performance import plot_cumulative_returns

    path = plot_cumulative_returns()
    assert path.exists()
    assert path.stat().st_size > 10_000


def test_performance_is_oos():
    from commodity_curve_factors.visualization.performance import plot_is_oos_comparison

    path = plot_is_oos_comparison()
    assert path.exists()


def test_factor_correlation():
    from commodity_curve_factors.visualization.factors import plot_factor_correlation

    path = plot_factor_correlation()
    assert path.exists()


def test_stress_test():
    from commodity_curve_factors.visualization.risk import plot_stress_test

    path = plot_stress_test()
    assert path.exists()
