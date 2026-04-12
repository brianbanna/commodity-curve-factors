"""Unit tests for signals.calendar_spreads.calendar_spread_signal."""

import pandas as pd
import pytest

from commodity_curve_factors.signals.calendar_spreads import calendar_spread_signal


def _make_carry(values: dict[str, float], date: str = "2020-01-03") -> pd.DataFrame:
    idx = pd.DatetimeIndex([date])
    return pd.DataFrame(values, index=idx)


class TestLongFrontWhenBackwardated:
    def test_carry_z_above_long_threshold(self):
        z = _make_carry({"CL": 2.0})
        sig = calendar_spread_signal(z, long_threshold=1.0, short_threshold=-1.0)
        assert sig.loc[sig.index[0], ("CL", "front")] == pytest.approx(1.0)
        assert sig.loc[sig.index[0], ("CL", "back")] == pytest.approx(-1.0)

    def test_carry_z_at_boundary_not_triggered(self):
        # Strictly above — boundary itself is flat
        z = _make_carry({"CL": 1.0})
        sig = calendar_spread_signal(z, long_threshold=1.0, short_threshold=-1.0)
        assert sig.loc[sig.index[0], ("CL", "front")] == pytest.approx(0.0)
        assert sig.loc[sig.index[0], ("CL", "back")] == pytest.approx(0.0)

    def test_multiple_commodities_backwardation(self):
        z = _make_carry({"CL": 2.5, "NG": 1.5, "GC": 0.5})
        sig = calendar_spread_signal(z, long_threshold=1.0, short_threshold=-1.0)
        assert sig.loc[sig.index[0], ("CL", "front")] == pytest.approx(1.0)
        assert sig.loc[sig.index[0], ("NG", "front")] == pytest.approx(1.0)
        assert sig.loc[sig.index[0], ("GC", "front")] == pytest.approx(0.0)


class TestShortFrontWhenContango:
    def test_carry_z_below_short_threshold(self):
        z = _make_carry({"CL": -2.0})
        sig = calendar_spread_signal(z, long_threshold=1.0, short_threshold=-1.0)
        assert sig.loc[sig.index[0], ("CL", "front")] == pytest.approx(-1.0)
        assert sig.loc[sig.index[0], ("CL", "back")] == pytest.approx(1.0)

    def test_carry_z_at_short_boundary_not_triggered(self):
        z = _make_carry({"CL": -1.0})
        sig = calendar_spread_signal(z, long_threshold=1.0, short_threshold=-1.0)
        assert sig.loc[sig.index[0], ("CL", "front")] == pytest.approx(0.0)
        assert sig.loc[sig.index[0], ("CL", "back")] == pytest.approx(0.0)

    def test_multiple_commodities_contango(self):
        z = _make_carry({"CL": -2.0, "NG": -1.5, "GC": -0.5})
        sig = calendar_spread_signal(z, long_threshold=1.0, short_threshold=-1.0)
        assert sig.loc[sig.index[0], ("CL", "front")] == pytest.approx(-1.0)
        assert sig.loc[sig.index[0], ("NG", "front")] == pytest.approx(-1.0)
        assert sig.loc[sig.index[0], ("GC", "front")] == pytest.approx(0.0)


class TestFlatWhenNeutral:
    def test_zero_carry_z_gives_flat(self):
        z = _make_carry({"CL": 0.0})
        sig = calendar_spread_signal(z, long_threshold=1.0, short_threshold=-1.0)
        assert sig.loc[sig.index[0], ("CL", "front")] == pytest.approx(0.0)
        assert sig.loc[sig.index[0], ("CL", "back")] == pytest.approx(0.0)

    def test_between_thresholds_gives_flat(self):
        z = _make_carry({"CL": 0.5, "NG": -0.5})
        sig = calendar_spread_signal(z, long_threshold=1.0, short_threshold=-1.0)
        for comm in ["CL", "NG"]:
            for leg in ["front", "back"]:
                assert sig.loc[sig.index[0], (comm, leg)] == pytest.approx(0.0)

    def test_mixed_dates(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        z = pd.DataFrame({"CL": [2.0, 0.0, -2.0]}, index=dates)
        sig = calendar_spread_signal(z, long_threshold=1.0, short_threshold=-1.0)
        assert sig.loc[dates[0], ("CL", "front")] == pytest.approx(1.0)
        assert sig.loc[dates[1], ("CL", "front")] == pytest.approx(0.0)
        assert sig.loc[dates[2], ("CL", "front")] == pytest.approx(-1.0)


class TestMultiIndexColumns:
    def test_columns_are_multiindex(self):
        z = _make_carry({"CL": 1.5, "NG": -1.5})
        sig = calendar_spread_signal(z)
        assert isinstance(sig.columns, pd.MultiIndex)

    def test_column_names(self):
        z = _make_carry({"CL": 1.5, "NG": -1.5})
        sig = calendar_spread_signal(z)
        assert sig.columns.names == ["commodity", "leg"]

    def test_all_leg_values_present(self):
        z = _make_carry({"CL": 1.5})
        sig = calendar_spread_signal(z)
        assert ("CL", "front") in sig.columns
        assert ("CL", "back") in sig.columns

    def test_column_order(self):
        z = _make_carry({"CL": 0.0, "NG": 0.0, "GC": 0.0})
        sig = calendar_spread_signal(z)
        expected_cols = [
            ("CL", "front"), ("CL", "back"),
            ("NG", "front"), ("NG", "back"),
            ("GC", "front"), ("GC", "back"),
        ]
        assert list(sig.columns) == expected_cols
