"""Unit tests for signals.ranking.resample_weights_weekly."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.ranking import resample_weights_weekly


def _make_daily_weights(start: str = "2020-01-06", periods: int = 10) -> pd.DataFrame:
    """Build a daily DataFrame whose values change every row (random-ish)."""
    dates = pd.date_range(start, periods=periods, freq="B")
    rng = np.random.default_rng(0)
    data = rng.standard_normal((periods, 3))
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C"])


class TestWeightsConstantWithinWeek:
    def test_weights_constant_within_week(self):
        """Mon-Thu should carry Friday's weight unchanged."""
        weights = _make_daily_weights("2020-01-06", periods=10)  # Mon 6 Jan to Fri 17 Jan
        result = resample_weights_weekly(weights, rebalance_day="friday")

        for date, row in result.iterrows():
            weekday = date.weekday()
            if weekday == 4:  # Friday itself
                continue
            # Find the most recent prior Friday
            friday = date - pd.Timedelta(days=(weekday - 4) % 7)
            if friday in result.index:
                expected = result.loc[friday]
                pd.testing.assert_series_equal(row, expected, check_names=False)


class TestWeightsChangeOnFriday:
    def test_weights_change_on_friday(self):
        """Consecutive Fridays should (generally) have different weights."""
        # Use a 2-week window so we get two Fridays
        weights = _make_daily_weights("2020-01-06", periods=10)
        result = resample_weights_weekly(weights, rebalance_day="friday")

        fridays = result.index[result.index.weekday == 4]
        assert len(fridays) >= 2, "Need at least 2 Fridays in test data"

        # The two fridays come from the original weights which are random → different
        w_fri1 = weights.loc[fridays[0]]
        w_fri2 = weights.loc[fridays[1]]
        assert not w_fri1.equals(w_fri2), "Consecutive Friday weights should differ"

        # Confirm resample_weights_weekly picks those weights up on Fridays
        pd.testing.assert_series_equal(result.loc[fridays[0]], w_fri1, check_names=False)
        pd.testing.assert_series_equal(result.loc[fridays[1]], w_fri2, check_names=False)


class TestFfillFromFriday:
    def test_monday_gets_previous_friday_weight(self):
        """Monday's weight should equal the preceding Friday's weight."""
        # 2020-01-06 Mon, 2020-01-07 Tue, ..., 2020-01-10 Fri, 2020-01-13 Mon
        weights = _make_daily_weights("2020-01-06", periods=6)
        result = resample_weights_weekly(weights, rebalance_day="friday")

        friday = pd.Timestamp("2020-01-10")
        monday = pd.Timestamp("2020-01-13")

        assert friday in result.index, "Friday must be present"
        assert monday in result.index, "Monday must be present"

        pd.testing.assert_series_equal(result.loc[monday], result.loc[friday], check_names=False)

    def test_nan_before_first_friday(self):
        """Days before the first Friday carry NaN (no signal yet)."""
        weights = _make_daily_weights("2020-01-06", periods=6)
        result = resample_weights_weekly(weights, rebalance_day="friday")

        # Monday-Thursday before the first Friday should be NaN
        pre_friday = result.loc[:"2020-01-09"]
        assert pre_friday.isna().all().all(), "Pre-first-Friday rows should be NaN"


class TestInvalidRebalanceDay:
    def test_invalid_day_raises(self):
        weights = _make_daily_weights()
        with pytest.raises(ValueError, match="rebalance_day"):
            resample_weights_weekly(weights, rebalance_day="sunday")
