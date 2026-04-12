"""Unit tests for signals.ranking.rank_and_select."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.ranking import rank_and_select


def _make_scores(values: dict[str, float], date: str = "2020-01-03") -> pd.DataFrame:
    """Build a single-row DataFrame of scores."""
    idx = pd.DatetimeIndex([date])
    return pd.DataFrame(values, index=idx)


class TestWeightsSumToZero:
    def test_single_row(self):
        scores = _make_scores({"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.0, "E": -2.0})
        w = rank_and_select(scores, long_n=2, short_n=2)
        assert abs(w.iloc[0].sum()) < 1e-10

    def test_multiple_rows(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=20, freq="B")
        scores = pd.DataFrame(rng.standard_normal((20, 6)), index=dates,
                               columns=["A", "B", "C", "D", "E", "F"])
        w = rank_and_select(scores, long_n=2, short_n=2)
        row_sums = w.sum(axis=1)
        assert (row_sums.abs() < 1e-10).all()

    def test_standard_3_3(self):
        scores = _make_scores(
            {"CL": 1.5, "NG": 0.8, "HO": 0.2, "GC": -0.3, "SI": -1.0, "ZC": -1.8}
        )
        w = rank_and_select(scores, long_n=3, short_n=3)
        assert abs(w.iloc[0].sum()) < 1e-10


class TestTopNGetPositiveWeight:
    def test_top_2_positive(self):
        scores = _make_scores({"A": 3.0, "B": 2.0, "C": 1.0, "D": -1.0, "E": -2.0})
        w = rank_and_select(scores, long_n=2, short_n=2)
        assert w.loc[w.index[0], "A"] == pytest.approx(0.5)
        assert w.loc[w.index[0], "B"] == pytest.approx(0.5)

    def test_long_weight_equals_one_over_long_n(self):
        scores = _make_scores({"A": 5.0, "B": 4.0, "C": 3.0, "D": -3.0, "E": -4.0, "F": -5.0})
        w = rank_and_select(scores, long_n=3, short_n=3)
        for col in ["A", "B", "C"]:
            assert w.loc[w.index[0], col] == pytest.approx(1.0 / 3)

    def test_middle_gets_zero(self):
        scores = _make_scores({"A": 3.0, "B": 2.0, "C": 0.0, "D": -2.0, "E": -3.0})
        w = rank_and_select(scores, long_n=1, short_n=1)
        assert w.loc[w.index[0], "C"] == pytest.approx(0.0)


class TestBottomNGetNegativeWeight:
    def test_bottom_2_negative(self):
        scores = _make_scores({"A": 3.0, "B": 2.0, "C": 1.0, "D": -1.0, "E": -2.0})
        w = rank_and_select(scores, long_n=2, short_n=2)
        assert w.loc[w.index[0], "D"] == pytest.approx(-0.5)
        assert w.loc[w.index[0], "E"] == pytest.approx(-0.5)

    def test_short_weight_equals_neg_one_over_short_n(self):
        scores = _make_scores({"A": 5.0, "B": 4.0, "C": 3.0, "D": -3.0, "E": -4.0, "F": -5.0})
        w = rank_and_select(scores, long_n=3, short_n=3)
        for col in ["D", "E", "F"]:
            assert w.loc[w.index[0], col] == pytest.approx(-1.0 / 3)


class TestNanScoresExcluded:
    def test_nan_column_not_selected(self):
        scores = _make_scores({"A": 2.0, "B": 1.0, "C": np.nan, "D": -1.0, "E": -2.0})
        w = rank_and_select(scores, long_n=1, short_n=1)
        assert w.loc[w.index[0], "C"] == pytest.approx(0.0)

    def test_nan_does_not_displace_valid_top(self):
        scores = _make_scores({"A": np.nan, "B": 3.0, "C": 2.0, "D": -2.0, "E": -3.0})
        w = rank_and_select(scores, long_n=1, short_n=1)
        # B should be top long (A is NaN), E should be bottom short
        assert w.loc[w.index[0], "B"] == pytest.approx(1.0)
        assert w.loc[w.index[0], "E"] == pytest.approx(-1.0)
        assert w.loc[w.index[0], "A"] == pytest.approx(0.0)

    def test_insufficient_valid_returns_zeros(self):
        # Only 3 valid but need long_n+short_n=4
        scores = _make_scores({"A": 2.0, "B": np.nan, "C": np.nan, "D": -1.0, "E": -2.0})
        w = rank_and_select(scores, long_n=2, short_n=2)
        assert (w.iloc[0] == 0.0).all()
