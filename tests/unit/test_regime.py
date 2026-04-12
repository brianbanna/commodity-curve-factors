"""Unit tests for signals.regime.classify_regime."""

import numpy as np
import pandas as pd

from commodity_curve_factors.signals.regime import classify_regime


def _make_vix(values: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, name="VIX")


class TestClassifyRegimeBasic:
    def test_calm_below_15(self):
        vix = _make_vix([10.0, 12.5, 14.9])
        result = classify_regime(vix)
        assert (result == "calm").all()

    def test_moderate_between_15_and_25(self):
        vix = _make_vix([15.0, 20.0, 24.9])
        result = classify_regime(vix)
        assert (result == "moderate").all()

    def test_turbulent_at_or_above_25(self):
        vix = _make_vix([25.0, 30.0, 50.0])
        result = classify_regime(vix)
        assert (result == "turbulent").all()

    def test_mixed_values(self):
        vix = _make_vix([10.0, 20.0, 35.0])
        result = classify_regime(vix)
        assert result.iloc[0] == "calm"
        assert result.iloc[1] == "moderate"
        assert result.iloc[2] == "turbulent"

    def test_returns_series(self):
        vix = _make_vix([10.0])
        result = classify_regime(vix)
        assert isinstance(result, pd.Series)

    def test_preserves_index(self):
        vix = _make_vix([10.0, 20.0, 35.0])
        result = classify_regime(vix)
        assert result.index.equals(vix.index)


class TestClassifyRegimeBoundary:
    def test_exactly_15_is_moderate_not_calm(self):
        vix = _make_vix([15.0])
        result = classify_regime(vix)
        assert result.iloc[0] == "moderate"

    def test_just_below_15_is_calm(self):
        vix = _make_vix([14.999])
        result = classify_regime(vix)
        assert result.iloc[0] == "calm"

    def test_exactly_25_is_turbulent_not_moderate(self):
        vix = _make_vix([25.0])
        result = classify_regime(vix)
        assert result.iloc[0] == "turbulent"

    def test_just_below_25_is_moderate(self):
        vix = _make_vix([24.999])
        result = classify_regime(vix)
        assert result.iloc[0] == "moderate"

    def test_custom_thresholds(self):
        vix = _make_vix([5.0, 12.0, 20.0])
        result = classify_regime(vix, thresholds=[10.0, 15.0])
        assert result.iloc[0] == "calm"
        assert result.iloc[1] == "moderate"
        assert result.iloc[2] == "turbulent"


class TestClassifyRegimeNan:
    def test_nan_vix_gives_nan_regime(self):
        vix = _make_vix([np.nan])
        result = classify_regime(vix)
        assert pd.isna(result.iloc[0])

    def test_partial_nan(self):
        vix = _make_vix([10.0, np.nan, 30.0])
        result = classify_regime(vix)
        assert result.iloc[0] == "calm"
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == "turbulent"

    def test_all_nan(self):
        vix = pd.Series([np.nan, np.nan, np.nan],
                        index=pd.date_range("2020-01-01", periods=3, freq="B"))
        result = classify_regime(vix)
        assert result.isna().all()

    def test_nan_does_not_pollute_adjacent(self):
        vix = _make_vix([14.0, np.nan, 26.0])
        result = classify_regime(vix)
        assert result.iloc[0] == "calm"
        assert result.iloc[2] == "turbulent"
