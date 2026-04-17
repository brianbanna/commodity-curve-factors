"""Unit tests for the vectorized backtest engine."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.backtest.engine import (
    compute_portfolio_returns,
    compute_turnover,
    run_backtest,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_COST_CONFIG = {
    "default": {"commission_bps": 3, "slippage_bps": 2, "roll_cost_bps": 2},
    "per_commodity": {
        "CL": {"commission_bps": 2, "slippage_bps": 1, "roll_cost_bps": 1},
    },
}

_DATES_20 = pd.date_range("2020-01-01", periods=20, freq="B")
_DATES_50 = pd.date_range("2020-01-01", periods=50, freq="B")


def _make_weights(n: int = 20, cols: list[str] | None = None, seed: int = 1) -> pd.DataFrame:
    if cols is None:
        cols = ["CL", "GC", "NG"]
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.uniform(-0.1, 0.1, (n, len(cols)))
    return pd.DataFrame(data, index=dates, columns=cols)


def _make_returns(n: int = 20, cols: list[str] | None = None, seed: int = 2) -> pd.DataFrame:
    if cols is None:
        cols = ["CL", "GC", "NG"]
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(0, 0.01, (n, len(cols)))
    return pd.DataFrame(data, index=dates, columns=cols)


def _assert_all_zero(series: pd.Series, tol: float = 1e-10) -> None:
    """Assert every element of *series* is within *tol* of zero."""
    assert (series.abs() < tol).all(), f"Expected all-zero series, max abs = {series.abs().max()}"


# ---------------------------------------------------------------------------
# compute_portfolio_returns
# ---------------------------------------------------------------------------


class TestComputePortfolioReturns:
    def test_zero_weights_zero_return(self):
        """All-zero weights produce zero portfolio return every day."""
        w = pd.DataFrame(0.0, index=_DATES_20, columns=["CL", "GC"])
        r = _make_returns(20, cols=["CL", "GC"])
        gross = compute_portfolio_returns(w, r)
        _assert_all_zero(gross)

    def test_positive_weights_positive_market_positive_return(self):
        """Positive weights on positive returns give a positive gross return."""
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        w = pd.DataFrame({"CL": [0.5, 0.5, 0.5, 0.5, 0.5]}, index=dates)
        r = pd.DataFrame({"CL": [0.02, 0.01, 0.03, 0.01, 0.02]}, index=dates)
        gross = compute_portfolio_returns(w, r)
        assert (gross > 0).all()

    def test_inner_join_drops_missing_columns(self):
        """Columns in returns but not weights are silently dropped."""
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        w = pd.DataFrame({"CL": [0.5] * 5}, index=dates)
        r = pd.DataFrame({"CL": [0.01] * 5, "EXTRA": [0.05] * 5}, index=dates)
        gross = compute_portfolio_returns(w, r)
        # CL gross = 0.5 * 0.01 = 0.005 every day
        np.testing.assert_allclose(gross.values, 0.005, rtol=1e-9)

    def test_known_value(self):
        """Manual: w=[0.3, 0.2], r=[0.05, -0.05] → sum = 0.015 - 0.010 = 0.005."""
        dates = pd.date_range("2020-01-01", periods=1, freq="B")
        w = pd.DataFrame({"A": [0.3], "B": [0.2]}, index=dates)
        r = pd.DataFrame({"A": [0.05], "B": [-0.05]}, index=dates)
        gross = compute_portfolio_returns(w, r)
        assert gross.iloc[0] == pytest.approx(0.3 * 0.05 + 0.2 * (-0.05))


# ---------------------------------------------------------------------------
# compute_turnover
# ---------------------------------------------------------------------------


class TestComputeTurnover:
    def test_first_row_nan(self):
        """First row of turnover is NaN (no prior weights)."""
        w = _make_weights(10)
        t = compute_turnover(w)
        assert pd.isna(t.iloc[0])

    def test_constant_weights_zero_turnover(self):
        """Constant weights produce zero turnover after the first row."""
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        w = pd.DataFrame({"CL": [0.1, 0.1, 0.1, 0.1, 0.1]}, index=dates)
        t = compute_turnover(w)
        _assert_all_zero(t.iloc[1:])

    def test_detects_rebalance(self):
        """A weight change of 0.10 on day 2 → turnover = 0.10."""
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        w = pd.DataFrame({"CL": [0.0, 0.0, 0.10, 0.10, 0.10]}, index=dates)
        t = compute_turnover(w)
        assert t.iloc[2] == pytest.approx(0.10)
        assert t.iloc[3] == pytest.approx(0.0)

    def test_multi_column_summed(self):
        """Turnover sums absolute changes across all columns."""
        dates = pd.date_range("2020-01-01", periods=2, freq="B")
        # Both columns change by 0.05 → total turnover = 0.10
        w = pd.DataFrame({"A": [0.0, 0.05], "B": [0.0, 0.05]}, index=dates)
        t = compute_turnover(w)
        assert t.iloc[1] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = frozenset(
    ["gross_return", "cost", "net_return", "cumulative", "drawdown", "turnover"]
)


class TestRunBacktest:
    def test_output_columns_present(self):
        """run_backtest must return a DataFrame with all 6 required columns."""
        w = _make_weights(20)
        r = _make_returns(20)
        result = run_backtest(w, r, _COST_CONFIG)
        assert _REQUIRED_COLUMNS <= set(result.columns)

    def test_zero_weights_zero_gross(self):
        """All-zero weights → gross_return = 0 on every day."""
        cols = ["CL", "GC"]
        w = pd.DataFrame(0.0, index=_DATES_20, columns=cols)
        r = _make_returns(20, cols=cols)
        result = run_backtest(w, r, _COST_CONFIG)
        _assert_all_zero(result["gross_return"])

    def test_net_return_le_gross_return(self):
        """Net return must be <= gross return (costs are non-negative)."""
        w = _make_weights(30)
        r = _make_returns(30)
        result = run_backtest(w, r, _COST_CONFIG)
        assert (result["net_return"] <= result["gross_return"] + 1e-12).all()

    def test_cumulative_starts_near_one(self):
        """Cumulative wealth on the first day must be close to 1.0."""
        w = _make_weights(20)
        r = _make_returns(20)
        result = run_backtest(w, r, _COST_CONFIG)
        assert result["cumulative"].iloc[0] == pytest.approx(1.0, abs=0.02)

    def test_cumulative_monotone_with_positive_returns(self):
        """When net returns are strictly positive every day, cumulative is monotone."""
        w = pd.DataFrame({"CL": np.full(20, 0.01)}, index=_DATES_20)
        # Large positive returns so net > 0 despite costs
        r = pd.DataFrame({"CL": np.full(20, 0.10)}, index=_DATES_20)
        result = run_backtest(w, r, _COST_CONFIG)
        assert result["cumulative"].is_monotonic_increasing

    def test_drawdown_correct(self):
        """Known peak→trough sequence produces a negative max drawdown."""
        dates = pd.date_range("2020-01-01", periods=6, freq="B")
        # Return stream: +10%, +10%, -20%, 0%, 0%, 0%  (no costs: constant weights after row 0)
        returns_vals = [0.10, 0.10, -0.20, 0.0, 0.0, 0.0]
        w = pd.DataFrame({"CL": np.ones(6)}, index=dates)
        r = pd.DataFrame({"CL": returns_vals}, index=dates)

        zero_cost = {
            "default": {"commission_bps": 0, "slippage_bps": 0, "roll_cost_bps": 0},
            "per_commodity": {},
        }
        result = run_backtest(w, r, zero_cost)
        assert result["drawdown"].min() < 0.0

    def test_drawdown_never_positive(self):
        """Drawdown is always <= 0 (it is a loss measured from the peak)."""
        w = _make_weights(50)
        r = _make_returns(50)
        result = run_backtest(w, r, _COST_CONFIG)
        assert (result["drawdown"] <= 1e-10).all()

    def test_row_count_matches_overlap(self):
        """Output length = number of overlapping date rows in weights and returns."""
        w = _make_weights(30)
        r = _make_returns(30)
        result = run_backtest(w, r, _COST_CONFIG)
        expected_len = len(w.index.intersection(r.index))
        assert len(result) == expected_len

    def test_empty_overlap_returns_empty_dataframe(self):
        """Non-overlapping dates yield an empty result with all 6 columns."""
        dates_w = pd.date_range("2020-01-01", periods=5, freq="B")
        dates_r = pd.date_range("2021-01-01", periods=5, freq="B")
        w = pd.DataFrame({"CL": np.zeros(5)}, index=dates_w)
        r = pd.DataFrame({"CL": np.zeros(5)}, index=dates_r)
        result = run_backtest(w, r, _COST_CONFIG)
        assert result.empty
        assert _REQUIRED_COLUMNS <= set(result.columns)
