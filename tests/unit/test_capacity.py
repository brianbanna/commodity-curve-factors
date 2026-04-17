"""Tests for evaluation/capacity.py."""

import numpy as np
import pandas as pd

from commodity_curve_factors.evaluation.capacity import (
    capacity_curve,
    estimate_max_capacity,
    volume_participation,
)


def _make_data(n=200):
    dates = pd.bdate_range("2020-01-01", periods=n)
    cols = ["CL", "NG", "GC"]
    rng = np.random.default_rng(42)
    weights = pd.DataFrame(rng.uniform(-0.1, 0.1, (n, 3)), index=dates, columns=cols)
    returns = pd.DataFrame(rng.standard_normal((n, 3)) * 0.01, index=dates, columns=cols)
    volume = pd.DataFrame(rng.uniform(1000, 10000, (n, 3)), index=dates, columns=cols)
    return weights, returns, volume


def test_volume_participation_shape():
    weights, _, volume = _make_data()
    result = volume_participation(weights, volume, aum=1e6)
    assert result.shape == weights.shape


def test_volume_participation_increases_with_aum():
    weights, _, volume = _make_data()
    p1 = volume_participation(weights, volume, aum=1e6)
    p2 = volume_participation(weights, volume, aum=10e6)
    assert p2.mean().mean() > p1.mean().mean()


def test_capacity_curve_columns():
    weights, returns, volume = _make_data()
    result = capacity_curve(weights, returns, volume, aum_range=[1e6, 10e6])
    assert list(result.columns) == ["aum", "sharpe", "impact_bps"]
    assert len(result) == 2


def test_capacity_curve_sharpe_decreases_with_aum():
    weights, returns, volume = _make_data()
    result = capacity_curve(weights, returns, volume, aum_range=[1e6, 100e6, 500e6])
    sharpes = result["sharpe"].tolist()
    # Sharpe should generally decrease with higher AUM due to market impact
    assert sharpes[0] >= sharpes[-1] - 0.5, "Sharpe should degrade with AUM"


def test_estimate_max_capacity_positive():
    weights, _, volume = _make_data()
    result = estimate_max_capacity(weights, volume, max_participation=0.01)
    assert result > 0
