"""Unit tests for signals.threshold.threshold_signal."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.threshold import threshold_signal


def _make_zscore(values: dict[str, float], date: str = "2020-01-03") -> pd.DataFrame:
    idx = pd.DatetimeIndex([date])
    return pd.DataFrame(values, index=idx)


class TestThresholdPositiveZGivesLong:
    def test_positive_above_threshold(self):
        z = _make_zscore({"A": 1.5})
        sig = threshold_signal(z, threshold=1.0)
        assert sig.loc[sig.index[0], "A"] == pytest.approx(1.0)

    def test_default_threshold_any_positive(self):
        z = _make_zscore({"A": 0.01})
        sig = threshold_signal(z, threshold=0.0)
        assert sig.loc[sig.index[0], "A"] == pytest.approx(1.0)

    def test_multiple_commodities_positive(self):
        z = _make_zscore({"A": 2.0, "B": 0.5, "C": 0.0})
        sig = threshold_signal(z, threshold=0.0)
        assert sig.loc[sig.index[0], "A"] == pytest.approx(1.0)
        assert sig.loc[sig.index[0], "B"] == pytest.approx(1.0)


class TestThresholdNegativeZGivesShort:
    def test_negative_below_neg_threshold(self):
        z = _make_zscore({"A": -1.5})
        sig = threshold_signal(z, threshold=1.0)
        assert sig.loc[sig.index[0], "A"] == pytest.approx(-1.0)

    def test_default_threshold_any_negative(self):
        z = _make_zscore({"A": -0.01})
        sig = threshold_signal(z, threshold=0.0)
        assert sig.loc[sig.index[0], "A"] == pytest.approx(-1.0)

    def test_multiple_commodities_negative(self):
        z = _make_zscore({"A": -2.0, "B": -0.5})
        sig = threshold_signal(z, threshold=0.0)
        assert sig.loc[sig.index[0], "A"] == pytest.approx(-1.0)
        assert sig.loc[sig.index[0], "B"] == pytest.approx(-1.0)


class TestThresholdNearZeroGivesFlat:
    def test_exactly_zero_gives_flat(self):
        z = _make_zscore({"A": 0.0})
        sig = threshold_signal(z, threshold=0.0)
        assert sig.loc[sig.index[0], "A"] == pytest.approx(0.0)

    def test_within_threshold_gives_flat(self):
        z = _make_zscore({"A": 0.5, "B": -0.5, "C": 0.0})
        sig = threshold_signal(z, threshold=1.0)
        assert sig.loc[sig.index[0], "A"] == pytest.approx(0.0)
        assert sig.loc[sig.index[0], "B"] == pytest.approx(0.0)
        assert sig.loc[sig.index[0], "C"] == pytest.approx(0.0)

    def test_at_positive_threshold_boundary_gives_flat(self):
        # Boundary: exactly at threshold is NOT strictly greater, so flat
        z = _make_zscore({"A": 1.0})
        sig = threshold_signal(z, threshold=1.0)
        assert sig.loc[sig.index[0], "A"] == pytest.approx(0.0)

    def test_at_negative_threshold_boundary_gives_flat(self):
        z = _make_zscore({"A": -1.0})
        sig = threshold_signal(z, threshold=1.0)
        assert sig.loc[sig.index[0], "A"] == pytest.approx(0.0)


class TestThresholdNanPreserved:
    def test_nan_input_gives_nan_output(self):
        z = _make_zscore({"A": np.nan, "B": 2.0})
        sig = threshold_signal(z, threshold=0.0)
        assert np.isnan(sig.loc[sig.index[0], "A"])
        assert sig.loc[sig.index[0], "B"] == pytest.approx(1.0)

    def test_all_nan_row(self):
        z = _make_zscore({"A": np.nan, "B": np.nan})
        sig = threshold_signal(z, threshold=0.5)
        assert sig.iloc[0].isna().all()

    def test_multiple_rows_with_nans(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        z = pd.DataFrame(
            {"A": [1.5, np.nan, -1.5], "B": [np.nan, 0.5, 0.5]},
            index=dates,
        )
        sig = threshold_signal(z, threshold=1.0)
        assert sig.loc[dates[0], "A"] == pytest.approx(1.0)
        assert np.isnan(sig.loc[dates[1], "A"])
        assert sig.loc[dates[2], "A"] == pytest.approx(-1.0)
        assert np.isnan(sig.loc[dates[0], "B"])
        assert sig.loc[dates[1], "B"] == pytest.approx(0.0)
