"""Unit tests for backtest.costs transaction cost model."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.backtest.costs import (
    apply_costs,
    compute_roll_costs,
    compute_transaction_costs,
    get_cost,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_COST_CONFIG = {
    "default": {"commission_bps": 3, "slippage_bps": 2, "roll_cost_bps": 2},
    "per_commodity": {
        "CL": {"commission_bps": 2, "slippage_bps": 1, "roll_cost_bps": 1},
        "KC": {"commission_bps": 5, "slippage_bps": 3, "roll_cost_bps": 3},
    },
}


def _make_weights(n: int = 20, cols: list[str] | None = None, seed: int = 7) -> pd.DataFrame:
    if cols is None:
        cols = ["CL", "NG", "GC"]
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.uniform(-0.1, 0.1, size=(n, len(cols)))
    return pd.DataFrame(data, index=dates, columns=cols)


def _assert_all_zero(series: pd.Series, tol: float = 1e-10) -> None:
    """Assert every element is within *tol* of zero."""
    assert (series.abs() < tol).all(), f"Expected all-zero series, max abs = {series.abs().max()}"


# ---------------------------------------------------------------------------
# get_cost
# ---------------------------------------------------------------------------


class TestGetCost:
    def test_per_commodity_overrides_default(self):
        """CL commission is 2 bps, not the default 3 bps."""
        value = get_cost("CL", "commission_bps", _COST_CONFIG)
        assert value == pytest.approx(2.0)

    def test_per_commodity_slippage(self):
        """CL slippage is 1 bps, not the default 2 bps."""
        value = get_cost("CL", "slippage_bps", _COST_CONFIG)
        assert value == pytest.approx(1.0)

    def test_falls_back_to_default_for_unknown_commodity(self):
        """An unregistered commodity falls back to the default commission."""
        value = get_cost("UNKNOWN_COMMODITY", "commission_bps", _COST_CONFIG)
        assert value == pytest.approx(3.0)

    def test_falls_back_to_default_slippage_for_ng(self):
        """NG has no per-commodity override → default slippage of 2 bps."""
        value = get_cost("NG", "slippage_bps", _COST_CONFIG)
        assert value == pytest.approx(2.0)

    def test_kc_roll_cost_override(self):
        """KC has 3 bps roll cost per-commodity."""
        value = get_cost("KC", "roll_cost_bps", _COST_CONFIG)
        assert value == pytest.approx(3.0)

    def test_missing_cost_type_returns_zero(self):
        """An unknown cost_type that is not in config defaults to 0.0."""
        value = get_cost("CL", "nonexistent_cost_bps", _COST_CONFIG)
        assert value == pytest.approx(0.0)

    def test_empty_per_commodity_section(self):
        """Config with no per_commodity key falls back to default."""
        config = {"default": {"commission_bps": 5}}
        value = get_cost("CL", "commission_bps", config)
        assert value == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# compute_transaction_costs
# ---------------------------------------------------------------------------


class TestComputeTransactionCosts:
    def test_zero_weights_zero_costs(self):
        """All-zero weights → zero turnover → zero transaction costs."""
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        w = pd.DataFrame(0.0, index=dates, columns=["CL", "NG"])
        costs = compute_transaction_costs(w, _COST_CONFIG)
        _assert_all_zero(costs)

    def test_proportional_to_turnover(self):
        """Doubling the weight change doubles the transaction cost."""
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        w_small = pd.DataFrame({"NG": [0.0, 0.10, 0.10]}, index=dates)
        w_large = pd.DataFrame({"NG": [0.0, 0.20, 0.20]}, index=dates)

        c_small = compute_transaction_costs(w_small, _COST_CONFIG)
        c_large = compute_transaction_costs(w_large, _COST_CONFIG)

        assert c_large.iloc[1] == pytest.approx(2 * c_small.iloc[1])

    def test_cost_uses_per_commodity_rates(self):
        """CL (2+1 bps = 3 bps total) vs NG (3+2 bps = 5 bps) for equal weight change."""
        dates = pd.date_range("2020-01-01", periods=2, freq="B")
        w_cl = pd.DataFrame({"CL": [0.0, 0.10]}, index=dates)
        w_ng = pd.DataFrame({"NG": [0.0, 0.10]}, index=dates)

        cost_cl = compute_transaction_costs(w_cl, _COST_CONFIG)
        cost_ng = compute_transaction_costs(w_ng, _COST_CONFIG)

        expected_cl = 3 / 10_000 * 0.10
        expected_ng = 5 / 10_000 * 0.10
        assert cost_cl.iloc[1] == pytest.approx(expected_cl)
        assert cost_ng.iloc[1] == pytest.approx(expected_ng)

    def test_first_row_zero(self):
        """First row should be zero (no prior row to diff against)."""
        w = _make_weights(10)
        costs = compute_transaction_costs(w, _COST_CONFIG)
        assert costs.iloc[0] == pytest.approx(0.0)

    def test_constant_weights_zero_costs_after_first(self):
        """After the initial rebalance, constant weights produce zero costs."""
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        w = pd.DataFrame({"CL": [0.0, 0.10, 0.10, 0.10, 0.10]}, index=dates)
        costs = compute_transaction_costs(w, _COST_CONFIG)
        assert costs.iloc[2] == pytest.approx(0.0)
        assert costs.iloc[3] == pytest.approx(0.0)
        assert costs.iloc[4] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_roll_costs
# ---------------------------------------------------------------------------


def _make_roll_schedule(roll_dates: list[str], dsmnem: str = "NCL0320") -> pd.DataFrame:
    """Build a minimal roll schedule starting one day before the weights window.

    Starts with a "warm-up" row one business day before the first weights date
    so the first entry in the weights window is never treated as a roll
    (shift(1) comparison uses the warm-up row instead of NaN).
    """
    # Start one extra business day before "2020-01-01" = "2019-12-31"
    all_dates = pd.date_range("2019-12-31", "2020-03-31", freq="B")
    records = []
    current_mnem = dsmnem
    roll_set = {pd.Timestamp(d) for d in roll_dates}
    for d in all_dates:
        if d in roll_set:
            current_mnem = current_mnem + "_next"
        records.append({"trade_date": d, "front_dsmnem": current_mnem, "front_futcode": 1})
    return pd.DataFrame(records)


class TestComputeRollCosts:
    def test_no_roll_zero_cost(self):
        """When there are no rolls in the period, roll costs are zero."""
        w = _make_weights(20, cols=["CL"])
        sched = _make_roll_schedule([])  # no roll dates → constant dsmnem
        costs = compute_roll_costs(w, sched, _COST_CONFIG)
        _assert_all_zero(costs)

    def test_roll_costs_only_on_roll_days(self):
        """Roll costs appear on the roll date and are zero on all other days."""
        roll_date = "2020-01-15"
        sched = _make_roll_schedule([roll_date])
        dates = pd.date_range("2020-01-01", periods=20, freq="B")
        w = pd.DataFrame({"CL": np.full(20, 0.10)}, index=dates)

        costs = compute_roll_costs(w, sched, _COST_CONFIG)

        roll_ts = pd.Timestamp(roll_date)
        assert costs[roll_ts] > 0.0, "Expected nonzero cost on roll day"
        non_roll_costs = costs.drop(index=roll_ts, errors="ignore")
        _assert_all_zero(non_roll_costs)

    def test_roll_cost_magnitude(self):
        """Roll cost = |weight| * roll_cost_bps / 10000 on roll day."""
        roll_date = "2020-01-08"
        sched = _make_roll_schedule([roll_date])
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        weight_val = 0.15
        w = pd.DataFrame({"CL": np.full(10, weight_val)}, index=dates)

        costs = compute_roll_costs(w, sched, _COST_CONFIG)

        roll_ts = pd.Timestamp(roll_date)
        # CL roll_cost_bps = 1 (from per_commodity override)
        expected = weight_val * 1.0 / 10_000
        assert costs[roll_ts] == pytest.approx(expected)

    def test_zero_weight_no_roll_cost(self):
        """Zero weight on the roll day → no roll cost charged."""
        roll_date = "2020-01-08"
        sched = _make_roll_schedule([roll_date])
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        w = pd.DataFrame({"CL": np.zeros(10)}, index=dates)

        costs = compute_roll_costs(w, sched, _COST_CONFIG)
        _assert_all_zero(costs)

    def test_empty_roll_schedule_returns_zeros(self):
        """Empty roll schedule → all-zero roll costs."""
        w = _make_weights(10)
        empty_sched = pd.DataFrame(columns=["trade_date", "front_dsmnem", "front_futcode"])
        costs = compute_roll_costs(w, empty_sched, _COST_CONFIG)
        _assert_all_zero(costs)


# ---------------------------------------------------------------------------
# apply_costs
# ---------------------------------------------------------------------------


class TestApplyCosts:
    def test_net_equals_gross_minus_costs(self):
        """net_returns = gross_returns - costs, element-wise."""
        w = _make_weights(30)
        gross = pd.Series(
            np.random.default_rng(1).normal(0, 0.005, 30),
            index=w.index,
        )
        costs, net = apply_costs(gross, w, _COST_CONFIG)
        pd.testing.assert_series_equal(net, gross - costs, check_names=False)

    def test_costs_nonnegative(self):
        """Costs must always be >= 0."""
        w = _make_weights(20)
        gross = pd.Series(
            np.random.default_rng(2).normal(0, 0.005, 20),
            index=w.index,
        )
        costs, _ = apply_costs(gross, w, _COST_CONFIG)
        assert (costs >= 0).all()

    def test_returns_tuple_of_series(self):
        """apply_costs returns a two-tuple of pd.Series."""
        w = _make_weights(10)
        gross = pd.Series(np.zeros(10), index=w.index)
        result = apply_costs(gross, w, _COST_CONFIG)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.Series)
        assert isinstance(result[1], pd.Series)
