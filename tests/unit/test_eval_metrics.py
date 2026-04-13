"""Unit tests for evaluation.metrics."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.evaluation.metrics import (
    compute_all_metrics,
    max_drawdown,
    sharpe_ratio,
    split_is_oos,
    cagr,
)


def _daily_returns(mean: float, std: float, n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = rng.normal(mean / 252, std / np.sqrt(252), n)
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.Series(values, index=dates)


class TestSharpeRatio:
    def test_sharpe_positive_for_positive_mean(self):
        r = _daily_returns(mean=0.10, std=0.15)
        assert sharpe_ratio(r) > 0.0

    def test_sharpe_negative_for_negative_mean(self):
        r = _daily_returns(mean=-0.10, std=0.15)
        assert sharpe_ratio(r) < 0.0

    def test_sharpe_zero_for_empty_series(self):
        r = pd.Series(dtype=float)
        assert sharpe_ratio(r) == pytest.approx(0.0)

    def test_sharpe_zero_for_all_zero_returns(self):
        # All-zero returns → std is 0 → sharpe returns 0
        r = pd.Series([0.0] * 100)
        assert sharpe_ratio(r) == pytest.approx(0.0)

    def test_sharpe_annualisation_scale(self):
        # Sharpe should scale by sqrt(252): use a long series with clear positive drift
        r = _daily_returns(mean=0.20, std=0.10, n=2520, seed=1)
        sr = sharpe_ratio(r)
        # With mean=0.20, std=0.10, SR should be around 2.0 ± noise
        assert sr > 0.5, f"Expected SR > 0.5 for strong drift, got {sr}"


class TestMaxDrawdown:
    def test_max_drawdown_negative(self):
        r = _daily_returns(mean=0.05, std=0.20)
        dd = max_drawdown(r)
        assert dd < 0.0

    def test_max_drawdown_monotone_growth(self):
        r = pd.Series([0.001] * 252)
        dd = max_drawdown(r)
        assert dd == pytest.approx(0.0, abs=1e-9)

    def test_max_drawdown_known_loss(self):
        # A 50% drop: log returns that sum to -ln(2)
        drop = np.log(0.5)
        r = pd.Series([0.0, drop, 0.0])
        dd = max_drawdown(r)
        assert dd == pytest.approx(-0.5, abs=1e-6)


class TestCagr:
    def test_cagr_positive_for_growth(self):
        # Use a long series with strong drift so realized CAGR is reliably positive
        r = _daily_returns(mean=0.20, std=0.10, n=2520, seed=1)
        assert cagr(r) > 0.0

    def test_cagr_empty_returns_zero(self):
        assert cagr(pd.Series(dtype=float)) == pytest.approx(0.0)

    def test_cagr_one_year_known_value(self):
        # log return of 0.10/252 per day for 252 days → total log return 0.10
        # CAGR = exp(0.10) - 1
        r = pd.Series([0.10 / 252] * 252)
        expected = np.exp(0.10) - 1.0
        assert cagr(r) == pytest.approx(expected, rel=1e-6)


class TestSplitIsOos:
    def test_split_is_oos_dates(self):
        dates = pd.date_range("2015-01-01", "2020-12-31", freq="B")
        r = pd.Series(np.ones(len(dates)), index=dates)
        is_r, oos_r = split_is_oos(r)
        assert is_r.index[-1] <= pd.Timestamp("2017-12-31")
        assert oos_r.index[0] >= pd.Timestamp("2018-01-01")

    def test_split_is_oos_no_overlap(self):
        dates = pd.date_range("2015-01-01", "2020-12-31", freq="B")
        r = pd.Series(np.ones(len(dates)), index=dates)
        is_r, oos_r = split_is_oos(r)
        assert len(is_r) + len(oos_r) == len(r)

    def test_split_custom_dates(self):
        dates = pd.date_range("2010-01-01", "2022-12-31", freq="B")
        r = pd.Series(np.ones(len(dates)), index=dates)
        is_r, oos_r = split_is_oos(r, is_end="2015-12-31", oos_start="2016-01-01")
        assert is_r.index[-1] <= pd.Timestamp("2015-12-31")
        assert oos_r.index[0] >= pd.Timestamp("2016-01-01")


class TestComputeAllMetrics:
    def test_compute_all_metrics_keys(self):
        r = _daily_returns(mean=0.08, std=0.15)
        m = compute_all_metrics(r)
        expected_keys = {
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "cagr",
            "volatility",
            "hit_rate",
        }
        assert set(m.keys()) == expected_keys

    def test_compute_all_metrics_values_are_floats(self):
        r = _daily_returns(mean=0.08, std=0.15)
        m = compute_all_metrics(r)
        for key, val in m.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"

    def test_hit_rate_in_unit_interval(self):
        r = _daily_returns(mean=0.08, std=0.15)
        m = compute_all_metrics(r)
        assert 0.0 <= m["hit_rate"] <= 1.0
