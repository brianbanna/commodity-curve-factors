"""Unit tests for signals.portfolio portfolio construction utilities."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.portfolio import (
    apply_execution_lag,
    apply_position_limits,
    apply_sector_limits,
    apply_vol_target,
    build_portfolio,
)

TRADING_DAYS = 252
RNG = np.random.default_rng(0)


def _make_weights(
    n: int = 100,
    cols: list[str] | None = None,
    seed: int = 1,
) -> pd.DataFrame:
    if cols is None:
        cols = ["CL", "NG", "GC", "ZC", "KC"]
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.uniform(-0.2, 0.2, size=(n, len(cols)))
    return pd.DataFrame(data, index=dates, columns=cols)


def _make_returns(
    n: int = 100,
    cols: list[str] | None = None,
    seed: int = 2,
) -> pd.DataFrame:
    if cols is None:
        cols = ["CL", "NG", "GC", "ZC", "KC"]
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(0, 0.01, size=(n, len(cols)))
    return pd.DataFrame(data, index=dates, columns=cols)


# ---------------------------------------------------------------------------
# apply_execution_lag
# ---------------------------------------------------------------------------

class TestExecutionLag:
    def test_lag_by_one_shifts_forward(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        w = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
        lagged = apply_execution_lag(w, lag_days=1)
        assert np.isnan(lagged.iloc[0, 0])
        assert lagged.iloc[1, 0] == pytest.approx(1.0)
        assert lagged.iloc[2, 0] == pytest.approx(2.0)

    def test_lag_zero_is_identity(self):
        w = _make_weights(20)
        lagged = apply_execution_lag(w, lag_days=0)
        pd.testing.assert_frame_equal(lagged, w)

    def test_lag_two_shifts_by_two(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        w = pd.DataFrame({"A": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=dates)
        lagged = apply_execution_lag(w, lag_days=2)
        assert np.isnan(lagged.iloc[0, 0])
        assert np.isnan(lagged.iloc[1, 0])
        assert lagged.iloc[2, 0] == pytest.approx(10.0)

    def test_lag_preserves_shape_and_index(self):
        w = _make_weights(30)
        lagged = apply_execution_lag(w, lag_days=1)
        assert lagged.shape == w.shape
        assert lagged.index.equals(w.index)

    def test_negative_lag_raises(self):
        w = _make_weights(5)
        with pytest.raises(ValueError):
            apply_execution_lag(w, lag_days=-1)


# ---------------------------------------------------------------------------
# apply_vol_target
# ---------------------------------------------------------------------------

class TestVolTargetScalesWeights:
    def test_portfolio_vol_near_target(self):
        # Use a longer series so rolling window is filled
        n = 300
        weights = _make_weights(n)
        returns = _make_returns(n)
        target = 0.10

        scaled = apply_vol_target(weights, returns, target_vol=target, lookback=60)

        # Portfolio returns after scaling
        port_ret = (scaled * returns).sum(axis=1).dropna()
        realized_vol = port_ret.std() * np.sqrt(TRADING_DAYS)

        # With enough data the vol should be reasonably close to target
        assert abs(realized_vol - target) < 0.05

    def test_scalar_capped_at_max_leverage(self):
        # Very low returns volatility → scalar would be huge without cap
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        weights = pd.DataFrame({"A": np.ones(n) * 0.1}, index=dates)
        # Near-zero returns → near-zero portfolio vol → scalar tries to be huge
        returns = pd.DataFrame({"A": np.ones(n) * 1e-10}, index=dates)
        max_lev = 2.0
        scaled = apply_vol_target(weights, returns, target_vol=0.10,
                                   lookback=60, max_leverage=max_lev)
        # No scaled weight should exceed max_leverage * original weight
        assert (scaled.abs() <= weights.abs() * max_lev + 1e-9).all().all()

    def test_output_shape_preserved(self):
        w = _make_weights(100)
        r = _make_returns(100)
        scaled = apply_vol_target(w, r, target_vol=0.10)
        assert scaled.shape == w.shape


# ---------------------------------------------------------------------------
# apply_position_limits
# ---------------------------------------------------------------------------

class TestPositionLimitsCapsWeights:
    def test_no_weight_exceeds_max(self):
        w = _make_weights(50)
        max_w = 0.15
        limited = apply_position_limits(w, max_weight=max_w)
        assert (limited.abs() <= max_w + 1e-9).all().all()

    def test_weights_below_max_unchanged(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        w = pd.DataFrame({"A": [0.05, 0.10, -0.08], "B": [0.12, -0.05, 0.07]}, index=dates)
        limited = apply_position_limits(w, max_weight=0.20)
        pd.testing.assert_frame_equal(limited, w)

    def test_weights_exactly_at_max_not_clipped(self):
        dates = pd.date_range("2020-01-01", periods=1, freq="B")
        w = pd.DataFrame({"A": [0.20]}, index=dates)
        limited = apply_position_limits(w, max_weight=0.20)
        assert limited.iloc[0, 0] == pytest.approx(0.20)

    def test_oversized_positive_and_negative_clipped(self):
        dates = pd.date_range("2020-01-01", periods=1, freq="B")
        w = pd.DataFrame({"A": [0.50], "B": [-0.30]}, index=dates)
        limited = apply_position_limits(w, max_weight=0.20)
        assert limited.iloc[0, 0] == pytest.approx(0.20)
        assert limited.iloc[0, 1] == pytest.approx(-0.20)


# ---------------------------------------------------------------------------
# apply_sector_limits
# ---------------------------------------------------------------------------

class TestSectorLimits:
    def _energy_weights(self, weight_per: float = 0.15) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        # CL, NG, HO, RB are all energy; total |weight| = 4 * weight_per
        data = {c: np.full(5, weight_per) for c in ["CL", "NG", "HO", "RB"]}
        return pd.DataFrame(data, index=dates)

    def test_sector_abs_weight_capped(self):
        w = self._energy_weights(weight_per=0.15)  # 4 * 0.15 = 0.60 > 0.40
        limited = apply_sector_limits(w, max_sector=0.40)
        energy_cols = ["CL", "NG", "HO", "RB"]
        sector_sum = limited[energy_cols].abs().sum(axis=1)
        assert (sector_sum <= 0.40 + 1e-9).all()

    def test_within_limit_not_scaled(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        # 2 energy at 0.10 each = 0.20 total < 0.40
        w = pd.DataFrame({"CL": [0.10, 0.10, 0.10], "NG": [0.10, 0.10, 0.10]}, index=dates)
        limited = apply_sector_limits(w, max_sector=0.40)
        pd.testing.assert_frame_equal(limited, w)

    def test_custom_sectors_respected(self):
        dates = pd.date_range("2020-01-01", periods=2, freq="B")
        w = pd.DataFrame({"X": [0.30, 0.30], "Y": [0.30, 0.30]}, index=dates)
        sectors = {"X": "group_a", "Y": "group_a"}
        limited = apply_sector_limits(w, max_sector=0.40, sectors=sectors)
        sector_sum = limited[["X", "Y"]].abs().sum(axis=1)
        assert (sector_sum <= 0.40 + 1e-9).all()

    def test_sector_scaling_proportional(self):
        dates = pd.date_range("2020-01-01", periods=1, freq="B")
        # CL = 0.30, NG = 0.30 (same size, same direction), total = 0.60
        w = pd.DataFrame({"CL": [0.30], "NG": [0.30]}, index=dates)
        limited = apply_sector_limits(w, max_sector=0.40)
        # Both should be scaled down equally, ratio preserved
        assert abs(limited.iloc[0, 0] - limited.iloc[0, 1]) < 1e-9


# ---------------------------------------------------------------------------
# build_portfolio end-to-end
# ---------------------------------------------------------------------------

class TestBuildPortfolioChainsAll:
    def _make_strategy_config(self) -> dict:
        return {
            "constraints": {
                "vol_target": 0.10,
                "max_position_weight": 0.20,
                "max_sector_weight": 0.40,
                "max_leverage": 2.0,
            },
            "execution": {
                "lag_days": 1,
            },
        }

    def _make_universe_config(self) -> dict:
        return {
            "commodities": {
                "CL": {"sector": "energy"},
                "NG": {"sector": "energy"},
                "GC": {"sector": "metals"},
                "ZC": {"sector": "agriculture"},
                "KC": {"sector": "softs"},
            }
        }

    def test_output_shape_preserved(self):
        w = _make_weights(200)
        r = _make_returns(200)
        result = build_portfolio(w, r, self._make_strategy_config(), self._make_universe_config())
        assert result.shape == w.shape

    def test_execution_lag_applied(self):
        # First row after lag should be NaN
        w = _make_weights(50)
        r = _make_returns(50)
        result = build_portfolio(w, r, self._make_strategy_config(), self._make_universe_config())
        assert result.iloc[0].isna().all()

    def test_position_limits_respected(self):
        w = _make_weights(200)
        r = _make_returns(200)
        cfg = self._make_strategy_config()
        max_w = cfg["constraints"]["max_position_weight"]
        result = build_portfolio(w, r, cfg, self._make_universe_config())
        # Exclude NaN rows (lag)
        valid = result.dropna()
        assert (valid.abs() <= max_w + 1e-9).all().all()

    def test_sector_limits_respected(self):
        # Use only energy commodities to test sector cap
        n = 200
        cols = ["CL", "NG", "HO", "RB"]
        rng = np.random.default_rng(7)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        w = pd.DataFrame(rng.uniform(-0.15, 0.15, (n, 4)), index=dates, columns=cols)
        r = pd.DataFrame(rng.normal(0, 0.01, (n, 4)), index=dates, columns=cols)
        universe = {
            "commodities": {c: {"sector": "energy"} for c in cols}
        }
        cfg = {
            "constraints": {
                "vol_target": 0.10,
                "max_position_weight": 0.20,
                "max_sector_weight": 0.40,
                "max_leverage": 2.0,
            },
            "execution": {"lag_days": 1},
        }
        result = build_portfolio(w, r, cfg, universe)
        valid = result.dropna()
        sector_sum = valid[cols].abs().sum(axis=1)
        assert (sector_sum <= 0.40 + 1e-9).all()

    def test_returns_dataframe(self):
        w = _make_weights(100)
        r = _make_returns(100)
        result = build_portfolio(w, r, self._make_strategy_config(), self._make_universe_config())
        assert isinstance(result, pd.DataFrame)
