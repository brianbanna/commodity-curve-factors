"""Tests for evaluation/factor_analysis.py."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.evaluation.factor_analysis import (
    cumulative_factor_returns,
    factor_correlations,
    ic_decay,
    information_coefficient,
    rolling_ic,
)


@pytest.fixture()
def dates():
    return pd.bdate_range("2020-01-01", periods=100)


@pytest.fixture()
def commodities():
    return ["CL", "NG", "GC", "SI", "ZC"]


@pytest.fixture()
def predictive_factor(dates, commodities):
    """Factor that is correlated with forward returns."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.standard_normal((len(dates), len(commodities))),
        index=dates,
        columns=commodities,
    )


@pytest.fixture()
def correlated_returns(predictive_factor, dates, commodities):
    """Returns that have positive correlation with factor (lagged)."""
    rng = np.random.default_rng(99)
    noise = rng.standard_normal(predictive_factor.shape) * 0.5
    return pd.DataFrame(
        predictive_factor.shift(1).values + noise,
        index=dates,
        columns=commodities,
    )


def test_information_coefficient_shape(predictive_factor, correlated_returns):
    ic = information_coefficient(predictive_factor, correlated_returns, lag=1)
    assert isinstance(ic, pd.Series)
    assert len(ic) > 0
    assert ic.name == "ic"


def test_information_coefficient_bounded(predictive_factor, correlated_returns):
    ic = information_coefficient(predictive_factor, correlated_returns, lag=1)
    assert (ic >= -1.0).all()
    assert (ic <= 1.0).all()


def test_information_coefficient_positive_for_correlated(predictive_factor, correlated_returns):
    ic = information_coefficient(predictive_factor, correlated_returns, lag=1)
    assert ic.mean() > 0.0, "Expected positive mean IC for correlated factor-return pair"


def test_information_coefficient_zero_for_noise(dates, commodities):
    rng = np.random.default_rng(123)
    factor = pd.DataFrame(rng.standard_normal((100, 5)), index=dates, columns=commodities)
    returns = pd.DataFrame(rng.standard_normal((100, 5)), index=dates, columns=commodities)
    ic = information_coefficient(factor, returns, lag=1)
    assert abs(ic.mean()) < 0.3, "Expected near-zero mean IC for random factor"


def test_information_coefficient_min_assets(dates):
    """IC requires at least 3 assets; fewer should produce empty result."""
    factor = pd.DataFrame({"A": np.ones(100), "B": np.ones(100)}, index=dates)
    returns = pd.DataFrame({"A": np.ones(100), "B": np.ones(100)}, index=dates)
    ic = information_coefficient(factor, returns, lag=1)
    assert len(ic) == 0


def test_rolling_ic_length(predictive_factor, correlated_returns):
    ric = rolling_ic(predictive_factor, correlated_returns, window=20, lag=1)
    assert isinstance(ric, pd.Series)
    assert len(ric) > 0


def test_ic_decay_columns(predictive_factor, correlated_returns):
    result = ic_decay(predictive_factor, correlated_returns, lags=[1, 5, 10])
    assert list(result.columns) == ["lag", "mean_ic", "std_ic", "t_stat", "n_obs"]
    assert len(result) == 3
    assert list(result["lag"]) == [1, 5, 10]


def test_ic_decay_default_lags(predictive_factor, correlated_returns):
    result = ic_decay(predictive_factor, correlated_returns)
    assert list(result["lag"]) == [1, 5, 10, 20]


def test_factor_correlations_symmetric(dates, commodities):
    rng = np.random.default_rng(42)
    factors = {
        "f1": pd.DataFrame(rng.standard_normal((100, 5)), index=dates, columns=commodities),
        "f2": pd.DataFrame(rng.standard_normal((100, 5)), index=dates, columns=commodities),
        "f3": pd.DataFrame(rng.standard_normal((100, 5)), index=dates, columns=commodities),
    }
    corr = factor_correlations(factors)
    assert corr.shape == (3, 3)
    np.testing.assert_array_almost_equal(corr.values, corr.values.T)
    np.testing.assert_array_almost_equal(np.diag(corr.values), np.ones(3))


def test_factor_correlations_identical():
    dates = pd.bdate_range("2020-01-01", periods=50)
    cols = ["A", "B", "C", "D"]
    rng = np.random.default_rng(7)
    data = pd.DataFrame(rng.standard_normal((50, 4)), index=dates, columns=cols)
    corr = factor_correlations({"x": data, "y": data})
    assert corr.loc["x", "y"] > 0.9, "Identical factors should have correlation ~1"


def test_cumulative_factor_returns_shape(predictive_factor, correlated_returns):
    factors = {"test_factor": predictive_factor}
    cum = cumulative_factor_returns(factors, correlated_returns, long_n=2, short_n=2)
    assert isinstance(cum, pd.DataFrame)
    assert "test_factor" in cum.columns
    assert len(cum) > 0
