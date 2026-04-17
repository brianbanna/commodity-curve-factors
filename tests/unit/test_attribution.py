"""Tests for evaluation/attribution.py."""

import numpy as np
import pandas as pd

from commodity_curve_factors.evaluation.attribution import (
    attribution_by_regime,
    attribution_by_sector,
    attribution_by_year,
    rolling_sharpe,
)


def _make_data(n=500):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(42)
    cols = ["CL", "NG", "GC", "SI", "ZC"]
    weights = pd.DataFrame(rng.uniform(-0.1, 0.1, (n, 5)), index=dates, columns=cols)
    returns = pd.DataFrame(rng.standard_normal((n, 5)) * 0.01, index=dates, columns=cols)
    return weights, returns, dates


def test_attribution_by_sector_has_energy():
    weights, returns, _ = _make_data()
    result = attribution_by_sector(weights, returns)
    assert "sector" in result.columns
    assert "energy" in result["sector"].values


def test_attribution_by_sector_pct_sums_to_one():
    weights, returns, _ = _make_data()
    result = attribution_by_sector(weights, returns)
    total_pct = result["pct_of_total"].sum()
    assert abs(total_pct - 1.0) < 0.1, f"Expected pct sum ~1.0, got {total_pct}"


def test_rolling_sharpe_length():
    _, returns, dates = _make_data()
    ret_series = returns["CL"]
    rs = rolling_sharpe(ret_series, window=60)
    assert len(rs) == len(ret_series)
    assert rs.name == "rolling_sharpe"


def test_attribution_by_regime():
    _, returns, dates = _make_data()
    ret_series = returns["CL"]
    regimes = pd.Series(
        np.where(np.arange(500) < 200, "calm", "turbulent"),
        index=dates,
    )
    result = attribution_by_regime(ret_series, regimes)
    assert "calm" in result
    assert "turbulent" in result
    assert "sharpe" in result["calm"]
    assert result["calm"]["n_days"] > 0


def test_attribution_by_year():
    _, returns, _ = _make_data()
    result = attribution_by_year(returns["CL"])
    assert "year" in result.columns
    assert "sharpe" in result.columns
    assert len(result) >= 1
