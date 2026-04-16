"""Tests for evaluation/stress.py."""

import numpy as np
import pandas as pd

from commodity_curve_factors.evaluation.stress import (
    drawdown_anatomy,
    historical_stress_test,
)


def _make_returns(n=1000):
    dates = pd.bdate_range("2008-01-01", periods=n)
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_normal(n) * 0.01, index=dates, name="returns")


def test_historical_stress_test_columns():
    returns = _make_returns()
    periods = {
        "test_crash": {"start": "2008-07-01", "end": "2008-12-31"},
    }
    result = historical_stress_test(returns, periods=periods)
    assert "period" in result.columns
    assert "max_drawdown" in result.columns
    assert "worst_day" in result.columns
    assert len(result) == 1


def test_historical_stress_test_skips_short_periods():
    returns = _make_returns()
    periods = {
        "too_short": {"start": "2008-01-01", "end": "2008-01-02"},
    }
    result = historical_stress_test(returns, periods=periods)
    assert len(result) == 0


def test_drawdown_anatomy_count():
    returns = _make_returns()
    result = drawdown_anatomy(returns, top_n=3)
    assert len(result) <= 3
    for dd in result:
        assert "depth" in dd
        assert dd["depth"] < 0


def test_drawdown_anatomy_sorted_by_depth():
    returns = _make_returns()
    result = drawdown_anatomy(returns, top_n=5)
    depths = [d["depth"] for d in result]
    assert depths == sorted(depths), "Drawdowns should be sorted worst-first"


def test_drawdown_anatomy_has_dates():
    returns = _make_returns()
    result = drawdown_anatomy(returns, top_n=1)
    if result:
        assert "peak_date" in result[0]
        assert "trough_date" in result[0]
