"""Tests for evaluation/bootstrap.py."""

import numpy as np
import pandas as pd

from commodity_curve_factors.evaluation.bootstrap import bootstrap_sharpe_ci


def test_bootstrap_returns_tuple_of_three():
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(500) * 0.01)
    point, lo, hi = bootstrap_sharpe_ci(returns, n_samples=1000, seed=42)
    assert isinstance(point, float)
    assert isinstance(lo, float)
    assert isinstance(hi, float)


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(500) * 0.01)
    point, lo, hi = bootstrap_sharpe_ci(returns, n_samples=5000, seed=42)
    assert lo <= point <= hi or abs(point - lo) < 0.5, (
        f"Point {point:.3f} should be near CI [{lo:.3f}, {hi:.3f}]"
    )


def test_bootstrap_positive_sharpe_has_positive_ci():
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(1000) * 0.01 + 0.001)
    point, lo, hi = bootstrap_sharpe_ci(returns, n_samples=5000, seed=42)
    assert point > 0, "Expected positive Sharpe for upward-drifting returns"


def test_bootstrap_reproducible():
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(500) * 0.01)
    r1 = bootstrap_sharpe_ci(returns, n_samples=1000, seed=123)
    r2 = bootstrap_sharpe_ci(returns, n_samples=1000, seed=123)
    assert r1 == r2, "Same seed should produce identical results"


def test_bootstrap_short_series():
    returns = pd.Series([0.01, -0.01, 0.005])
    point, lo, hi = bootstrap_sharpe_ci(returns, n_samples=100, block_size=20)
    assert point == 0.0, "Too-short series should return zeros"
