"""Unit tests for signals.positioning_filter.apply_positioning_filter."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.positioning_filter import apply_positioning_filter


def _make_df(values: dict[str, float], date: str = "2020-01-03") -> pd.DataFrame:
    idx = pd.DatetimeIndex([date])
    return pd.DataFrame(values, index=idx)


class TestFilterZerosOutCrowdedLong:
    def test_long_signal_with_high_positioning_is_zeroed(self):
        sig = _make_df({"CL": 1.0})
        pos = _make_df({"CL": 0.95})
        result = apply_positioning_filter(sig, pos, crowded_threshold=0.90)
        assert result.loc[result.index[0], "CL"] == pytest.approx(0.0)

    def test_long_signal_exactly_at_threshold_not_zeroed(self):
        # Strictly greater than: exactly at threshold is NOT crowded
        sig = _make_df({"CL": 1.0})
        pos = _make_df({"CL": 0.90})
        result = apply_positioning_filter(sig, pos, crowded_threshold=0.90)
        assert result.loc[result.index[0], "CL"] == pytest.approx(1.0)

    def test_flat_signal_not_affected_by_high_positioning(self):
        sig = _make_df({"CL": 0.0})
        pos = _make_df({"CL": 0.99})
        result = apply_positioning_filter(sig, pos, crowded_threshold=0.90)
        assert result.loc[result.index[0], "CL"] == pytest.approx(0.0)

    def test_multiple_commodities_only_crowded_zeroed(self):
        sig = _make_df({"CL": 1.0, "NG": 1.0})
        pos = _make_df({"CL": 0.95, "NG": 0.70})
        result = apply_positioning_filter(sig, pos, crowded_threshold=0.90)
        assert result.loc[result.index[0], "CL"] == pytest.approx(0.0)
        assert result.loc[result.index[0], "NG"] == pytest.approx(1.0)


class TestFilterZerosOutCrowdedShort:
    def test_short_signal_with_low_positioning_is_zeroed(self):
        sig = _make_df({"CL": -1.0})
        pos = _make_df({"CL": 0.05})
        result = apply_positioning_filter(sig, pos, crowded_threshold=0.90)
        assert result.loc[result.index[0], "CL"] == pytest.approx(0.0)

    def test_short_signal_exactly_at_lower_boundary_not_zeroed(self):
        # 1 - 0.90 = 0.10; strictly less than: exactly 0.10 is NOT crowded
        sig = _make_df({"CL": -1.0})
        pos = _make_df({"CL": 0.10})
        result = apply_positioning_filter(sig, pos, crowded_threshold=0.90)
        assert result.loc[result.index[0], "CL"] == pytest.approx(-1.0)

    def test_flat_signal_not_affected_by_low_positioning(self):
        sig = _make_df({"CL": 0.0})
        pos = _make_df({"CL": 0.01})
        result = apply_positioning_filter(sig, pos, crowded_threshold=0.90)
        assert result.loc[result.index[0], "CL"] == pytest.approx(0.0)


class TestFilterPassesThroughUncrowded:
    def test_uncrowded_long_passes_through(self):
        sig = _make_df({"CL": 1.0})
        pos = _make_df({"CL": 0.50})
        result = apply_positioning_filter(sig, pos, crowded_threshold=0.90)
        assert result.loc[result.index[0], "CL"] == pytest.approx(1.0)

    def test_uncrowded_short_passes_through(self):
        sig = _make_df({"CL": -1.0})
        pos = _make_df({"CL": 0.50})
        result = apply_positioning_filter(sig, pos, crowded_threshold=0.90)
        assert result.loc[result.index[0], "CL"] == pytest.approx(-1.0)

    def test_fractional_signals_pass_through(self):
        sig = _make_df({"CL": 0.33, "NG": -0.33})
        pos = _make_df({"CL": 0.55, "NG": 0.55})
        result = apply_positioning_filter(sig, pos, crowded_threshold=0.90)
        assert result.loc[result.index[0], "CL"] == pytest.approx(0.33)
        assert result.loc[result.index[0], "NG"] == pytest.approx(-0.33)


class TestFilterHandlesNan:
    def test_nan_signal_remains_nan(self):
        sig = _make_df({"CL": np.nan})
        pos = _make_df({"CL": 0.95})
        result = apply_positioning_filter(sig, pos, crowded_threshold=0.90)
        assert np.isnan(result.loc[result.index[0], "CL"])

    def test_nan_positioning_preserves_signal_as_nan(self):
        # When positioning is NaN, the crowded condition is False → signal passes through
        # but the underlying comparison NaN > threshold = False, so signal is unchanged
        sig = _make_df({"CL": 1.0})
        pos = _make_df({"CL": np.nan})
        result = apply_positioning_filter(sig, pos, crowded_threshold=0.90)
        # NaN > 0.90 evaluates to False → signal NOT zeroed → passes through unchanged
        assert result.loc[result.index[0], "CL"] == pytest.approx(1.0)

    def test_mixed_nan_and_valid(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        sig = pd.DataFrame({"CL": [1.0, np.nan, -1.0]}, index=dates)
        pos = pd.DataFrame({"CL": [0.95, 0.95, 0.05]}, index=dates)
        result = apply_positioning_filter(sig, pos, crowded_threshold=0.90)
        assert result.loc[dates[0], "CL"] == pytest.approx(0.0)  # crowded long → zeroed
        assert np.isnan(result.loc[dates[1], "CL"])  # NaN signal → NaN
        assert result.loc[dates[2], "CL"] == pytest.approx(0.0)  # crowded short → zeroed
