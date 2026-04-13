"""Unit tests for backtest.benchmarks."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.backtest.benchmarks import (
    cash_benchmark,
    equal_weight_long,
    load_market_benchmarks,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DATES_20 = pd.date_range("2020-01-01", periods=20, freq="B")
_DATES_252 = pd.date_range("2020-01-01", periods=252, freq="B")


def _make_returns(n: int = 20, cols: list[str] | None = None, seed: int = 3) -> pd.DataFrame:
    if cols is None:
        cols = ["CL", "GC", "NG", "ZC"]
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(0, 0.01, (n, len(cols)))
    return pd.DataFrame(data, index=dates, columns=cols)


def _assert_series_allclose(actual: pd.Series, expected: float, rtol: float = 1e-9) -> None:
    """Assert every element of *actual* equals *expected* within *rtol*."""
    np.testing.assert_allclose(actual.values, expected, rtol=rtol, atol=1e-14)


# ---------------------------------------------------------------------------
# equal_weight_long
# ---------------------------------------------------------------------------


class TestEqualWeightLong:
    def test_is_mean_return(self):
        """equal_weight_long(returns) == returns.mean(axis=1) for all rows."""
        r = _make_returns(20)
        ew = equal_weight_long(r)
        expected = r.mean(axis=1)
        pd.testing.assert_series_equal(ew, expected, check_names=False, rtol=1e-10)

    def test_single_column(self):
        """Single asset: portfolio return = that asset's return."""
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        r = pd.DataFrame({"CL": [0.01, -0.02, 0.03, 0.00, -0.01]}, index=dates)
        ew = equal_weight_long(r)
        pd.testing.assert_series_equal(ew, r["CL"], check_names=False)

    def test_known_value(self):
        """Two assets: equal weight → portfolio return = (r1 + r2) / 2."""
        dates = pd.date_range("2020-01-01", periods=2, freq="B")
        r = pd.DataFrame({"A": [0.04, 0.02], "B": [0.02, -0.02]}, index=dates)
        ew = equal_weight_long(r)
        assert ew.iloc[0] == pytest.approx((0.04 + 0.02) / 2)
        assert ew.iloc[1] == pytest.approx((0.02 + (-0.02)) / 2)

    def test_returns_series(self):
        """Output must be a pd.Series."""
        r = _make_returns(10)
        ew = equal_weight_long(r)
        assert isinstance(ew, pd.Series)

    def test_index_preserved(self):
        """Output index matches the input returns index."""
        r = _make_returns(15)
        ew = equal_weight_long(r)
        assert ew.index.equals(r.index)

    def test_empty_columns_returns_zeros(self):
        """Zero-column returns → all-zero portfolio returns."""
        r = pd.DataFrame(index=_DATES_20)
        ew = equal_weight_long(r)
        assert len(ew) == len(_DATES_20)
        np.testing.assert_allclose(ew.values, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# cash_benchmark
# ---------------------------------------------------------------------------


class TestCashBenchmark:
    def test_constant_daily_rate(self):
        """Every day must have exactly annual_rate / 252."""
        annual = 0.02
        cash = cash_benchmark(_DATES_252, annual_rate=annual)
        _assert_series_allclose(cash, annual / 252.0)

    def test_constant_daily_rate_4pct(self):
        """Verify the daily rate for 4% annual."""
        cash = cash_benchmark(_DATES_20, annual_rate=0.04)
        _assert_series_allclose(cash, 0.04 / 252.0)

    def test_default_rate_is_2pct(self):
        """Default annual_rate is 2%."""
        cash = cash_benchmark(_DATES_20)
        _assert_series_allclose(cash, 0.02 / 252.0)

    def test_index_matches_dates(self):
        """Output index == the provided dates."""
        cash = cash_benchmark(_DATES_20)
        assert cash.index.equals(_DATES_20)

    def test_returns_series(self):
        """Output must be a pd.Series."""
        cash = cash_benchmark(_DATES_20)
        assert isinstance(cash, pd.Series)

    def test_zero_rate(self):
        """Annual rate of 0 → all-zero daily returns."""
        cash = cash_benchmark(_DATES_20, annual_rate=0.0)
        np.testing.assert_allclose(cash.values, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# load_market_benchmarks
# ---------------------------------------------------------------------------


def _make_price_df(n: int = 50, seed: int = 5) -> pd.DataFrame:
    """Build a synthetic OHLCV-like DataFrame with a Close column."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"Close": prices}, index=dates)


class TestLoadMarketBenchmarks:
    def test_keys_are_spy_and_agg(self):
        """When both spy and agg are in macro_data, result has 'SPY' and 'AGG'."""
        macro_data = {
            "spy": _make_price_df(seed=10),
            "agg": _make_price_df(seed=11),
        }
        result = load_market_benchmarks(macro_data)
        assert set(result.keys()) == {"SPY", "AGG"}

    def test_spy_only(self):
        """If only spy is available, result has only 'SPY'."""
        macro_data = {"spy": _make_price_df(seed=12)}
        result = load_market_benchmarks(macro_data)
        assert "SPY" in result
        assert "AGG" not in result

    def test_agg_only(self):
        """If only agg is available, result has only 'AGG'."""
        macro_data = {"agg": _make_price_df(seed=13)}
        result = load_market_benchmarks(macro_data)
        assert "AGG" in result
        assert "SPY" not in result

    def test_empty_macro_data_empty_result(self):
        """No matching keys → empty result dict."""
        result = load_market_benchmarks({})
        assert result == {}

    def test_returns_are_pct_change(self):
        """Returned series are % changes of the Close column (first row dropped)."""
        prices = [100.0, 105.0, 110.25]  # +5%, +5%
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        macro_data = {"spy": pd.DataFrame({"Close": prices}, index=dates)}
        result = load_market_benchmarks(macro_data)
        spy_ret = result["SPY"]
        assert len(spy_ret) == 2  # pct_change drops first row
        assert spy_ret.iloc[0] == pytest.approx(0.05)
        assert spy_ret.iloc[1] == pytest.approx(0.05)

    def test_values_are_series(self):
        """Each value in the returned dict must be a pd.Series."""
        macro_data = {
            "spy": _make_price_df(seed=20),
            "agg": _make_price_df(seed=21),
        }
        result = load_market_benchmarks(macro_data)
        for key, val in result.items():
            assert isinstance(val, pd.Series), f"{key} is not a Series"
