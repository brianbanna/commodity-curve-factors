"""Unit tests for backtest.sensitivity.run_cost_sensitivity."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.backtest.sensitivity import run_cost_sensitivity

# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

_DATES_100 = pd.date_range("2020-01-01", periods=100, freq="B")
_COLS = ["CL", "GC", "NG"]

_REQUIRED_COLUMNS = frozenset(["cost_bps", "sharpe", "cagr", "max_drawdown", "cumulative"])


def _make_weights(n: int = 100, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.uniform(-0.1, 0.1, (n, len(_COLS)))
    return pd.DataFrame(data, index=dates, columns=_COLS)


def _make_returns(n: int = 100, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(0, 0.01, (n, len(_COLS)))
    return pd.DataFrame(data, index=dates, columns=_COLS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunCostSensitivity:
    def test_output_shape(self):
        """One row per cost level, required columns present."""
        w = _make_weights()
        r = _make_returns()
        bps_list = [0, 5, 10]
        result = run_cost_sensitivity(w, r, bps_list)
        assert len(result) == len(bps_list)
        assert _REQUIRED_COLUMNS <= set(result.columns)

    def test_cost_bps_column_matches_input(self):
        """cost_bps column must reflect the input list exactly."""
        w = _make_weights()
        r = _make_returns()
        bps_list = [0, 2, 8, 20]
        result = run_cost_sensitivity(w, r, bps_list)
        assert list(result["cost_bps"]) == bps_list

    def test_higher_cost_lower_or_equal_sharpe(self):
        """Increasing costs should not improve the Sharpe ratio."""
        w = _make_weights(200, seed=99)
        r = _make_returns(200, seed=100)
        bps_list = [0, 5, 10, 20]
        result = run_cost_sensitivity(w, r, bps_list)
        sharpes = result["sharpe"].dropna().tolist()
        # Each successive Sharpe must be <= previous (non-strict due to rounding)
        for i in range(1, len(sharpes)):
            assert sharpes[i] <= sharpes[i - 1] + 1e-9, (
                f"Sharpe rose from {sharpes[i - 1]:.4f} to {sharpes[i]:.4f} "
                f"when cost increased from {bps_list[i - 1]} to {bps_list[i]} bps"
            )

    def test_higher_cost_lower_or_equal_cagr(self):
        """Increasing costs should not improve CAGR."""
        w = _make_weights(200, seed=99)
        r = _make_returns(200, seed=100)
        bps_list = [0, 5, 10, 20]
        result = run_cost_sensitivity(w, r, bps_list)
        cagrs = result["cagr"].dropna().tolist()
        for i in range(1, len(cagrs)):
            assert cagrs[i] <= cagrs[i - 1] + 1e-9

    def test_zero_cost_gross_equals_net(self):
        """At 0 bps, the Sharpe from sensitivity must match a zero-cost backtest."""
        from commodity_curve_factors.backtest.engine import run_backtest

        w = _make_weights(100, seed=7)
        r = _make_returns(100, seed=8)

        zero_cfg = {
            "default": {"commission_bps": 0.0, "slippage_bps": 0.0, "roll_cost_bps": 0.0},
            "per_commodity": {},
        }
        bt = run_backtest(w, r, zero_cfg)
        net = bt["net_return"]
        std = float(net.std())
        expected_sharpe = float(net.mean()) / std * np.sqrt(252) if std > 0 else 0.0

        result = run_cost_sensitivity(w, r, [0])
        assert result["sharpe"].iloc[0] == pytest.approx(expected_sharpe, rel=1e-6)

    def test_max_drawdown_non_positive(self):
        """max_drawdown must be <= 0 for all cost levels."""
        w = _make_weights(100)
        r = _make_returns(100)
        result = run_cost_sensitivity(w, r, [0, 5, 10])
        assert (result["max_drawdown"].dropna() <= 1e-10).all()

    def test_empty_bps_list_returns_empty_dataframe(self):
        """Empty cost list produces an empty DataFrame with the right columns."""
        w = _make_weights(50)
        r = _make_returns(50)
        result = run_cost_sensitivity(w, r, [])
        assert result.empty
        assert _REQUIRED_COLUMNS <= set(result.columns)

    def test_single_cost_level(self):
        """Single-element list should produce exactly one output row."""
        w = _make_weights(50)
        r = _make_returns(50)
        result = run_cost_sensitivity(w, r, [7])
        assert len(result) == 1
        assert int(result["cost_bps"].iloc[0]) == 7

    def test_non_overlapping_data_produces_nan_row(self):
        """When weights and returns have no overlapping dates, row should have NaN metrics."""
        dates_w = pd.date_range("2020-01-01", periods=20, freq="B")
        dates_r = pd.date_range("2021-06-01", periods=20, freq="B")
        w = pd.DataFrame(
            np.random.default_rng(0).uniform(-0.1, 0.1, (20, 2)),
            index=dates_w,
            columns=["CL", "GC"],
        )
        r = pd.DataFrame(
            np.random.default_rng(1).normal(0, 0.01, (20, 2)),
            index=dates_r,
            columns=["CL", "GC"],
        )
        result = run_cost_sensitivity(w, r, [5])
        assert len(result) == 1
        # All metric columns should be NaN (empty backtest)
        for col in ["sharpe", "cagr", "max_drawdown", "cumulative"]:
            assert pd.isna(result[col].iloc[0]), f"Expected NaN for {col}"
