"""Tests for factor transform primitives.

These transforms underpin every factor module in the project.  The single
most important property tested here is that ``expanding_zscore`` has ZERO
lookahead bias — a value at time t must never see data from t+1 onward.
"""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.factors.transforms import (
    cross_sectional_rank,
    expanding_zscore,
    expanding_zscore_df,
    percentile_rank,
)


# ---------------------------------------------------------------------------
# expanding_zscore
# ---------------------------------------------------------------------------


class TestExpandingZscore:
    def test_no_lookahead(self) -> None:
        """THE most critical test: z(t) must ONLY use data from times 0..t.

        Specifically: z(3) = (x(3) - mean(x[0:4])) / std(x[0:4], ddof=1)
        where x = [1, 2, 3, 4, 100].  The value 100 at index 4 must NOT
        affect z(3).
        """
        s = pd.Series(
            [1.0, 2.0, 3.0, 4.0, 100.0],
            index=pd.date_range("2020-01-01", periods=5),
        )
        z = expanding_zscore(s, min_periods=2)

        # First value is NaN (only 1 obs, need min_periods=2)
        assert pd.isna(z.iloc[0])

        # z(3) uses only [1, 2, 3, 4] — NOT 100
        mean_0_3 = np.mean([1.0, 2.0, 3.0, 4.0])
        std_0_3 = np.std([1.0, 2.0, 3.0, 4.0], ddof=1)
        expected_z3 = (4.0 - mean_0_3) / std_0_3
        np.testing.assert_allclose(z.iloc[3], expected_z3, rtol=1e-10)

        # z(4) uses [1, 2, 3, 4, 100] — 100 is a huge outlier → z near 1
        mean_0_4 = np.mean([1.0, 2.0, 3.0, 4.0, 100.0])
        std_0_4 = np.std([1.0, 2.0, 3.0, 4.0, 100.0], ddof=1)
        expected_z4 = (100.0 - mean_0_4) / std_0_4
        np.testing.assert_allclose(z.iloc[4], expected_z4, rtol=1e-10)

    def test_min_periods_nans(self) -> None:
        """First min_periods-1 values are NaN."""
        s = pd.Series(
            range(10),
            dtype=float,
            index=pd.date_range("2020-01-01", periods=10),
        )
        z = expanding_zscore(s, min_periods=5)
        assert z.iloc[:4].isna().all()
        assert z.iloc[4:].notna().all()

    def test_constant_series(self) -> None:
        """Constant input → z-score is 0.0, not NaN or inf."""
        s = pd.Series(
            [5.0] * 20,
            index=pd.date_range("2020-01-01", periods=20),
        )
        z = expanding_zscore(s, min_periods=2)
        assert (z.iloc[1:] == 0.0).all()

    def test_nan_input_passthrough(self) -> None:
        """NaN in input → NaN in output; does not corrupt subsequent values."""
        s = pd.Series(
            [1.0, np.nan, 3.0, 4.0, 5.0],
            index=pd.date_range("2020-01-01", periods=5),
        )
        z = expanding_zscore(s, min_periods=2)
        assert pd.isna(z.iloc[1])
        assert pd.notna(z.iloc[3])

    def test_output_index_preserved(self) -> None:
        """Output index matches input index exactly."""
        idx = pd.date_range("2021-06-01", periods=8)
        s = pd.Series(range(8), dtype=float, index=idx)
        z = expanding_zscore(s, min_periods=3)
        pd.testing.assert_index_equal(z.index, idx)

    def test_output_name_preserved(self) -> None:
        """Output Series name is preserved from input."""
        s = pd.Series(
            range(10), dtype=float, name="carry", index=pd.date_range("2020-01-01", periods=10)
        )
        z = expanding_zscore(s, min_periods=2)
        assert z.name == "carry"


# ---------------------------------------------------------------------------
# expanding_zscore_df
# ---------------------------------------------------------------------------


class TestExpandingZscoreDf:
    def test_applies_per_column(self) -> None:
        """Each column is z-scored independently; same relative pattern → same z."""
        df = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0],
                "B": [10.0, 20.0, 30.0, 40.0],
            },
            index=pd.date_range("2020-01-01", periods=4),
        )
        result = expanding_zscore_df(df, min_periods=2)
        assert result.shape == df.shape
        # A and B are perfectly correlated linear series → identical z-scores
        np.testing.assert_allclose(
            result["A"].dropna().values,
            result["B"].dropna().values,
        )

    def test_output_shape_and_columns(self) -> None:
        """Output has same shape, index, and columns as input."""
        df = pd.DataFrame(
            np.random.default_rng(42).standard_normal((20, 5)),
            columns=list("ABCDE"),
            index=pd.date_range("2020-01-01", periods=20),
        )
        result = expanding_zscore_df(df, min_periods=5)
        assert result.shape == df.shape
        assert list(result.columns) == list(df.columns)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_columns_independent(self) -> None:
        """Adding a constant to one column does not affect the other."""
        df = pd.DataFrame(
            {
                "X": [1.0, 2.0, 3.0, 4.0, 5.0],
                "Y": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            },
            index=pd.date_range("2020-01-01", periods=5),
        )
        result = expanding_zscore_df(df, min_periods=2)
        # Scale difference shouldn't matter; z-scores should be identical
        np.testing.assert_allclose(
            result["X"].dropna().values,
            result["Y"].dropna().values,
        )


# ---------------------------------------------------------------------------
# cross_sectional_rank
# ---------------------------------------------------------------------------


class TestCrossSectionalRank:
    def test_scales_to_01(self) -> None:
        """Lowest → 0.0, highest → 1.0."""
        row = pd.Series({"CL": 10.0, "NG": 20.0, "GC": 30.0})
        ranked = cross_sectional_rank(row)
        assert ranked["CL"] == 0.0
        assert ranked["GC"] == 1.0
        assert ranked["NG"] == pytest.approx(0.5)

    def test_handles_nan(self) -> None:
        """NaN values are excluded from ranking; NaN in → NaN out."""
        row = pd.Series({"CL": 10.0, "NG": np.nan, "GC": 30.0})
        ranked = cross_sectional_rank(row)
        assert pd.isna(ranked["NG"])
        assert ranked["CL"] == 0.0
        assert ranked["GC"] == 1.0

    def test_single_value(self) -> None:
        """Single non-NaN → 0.5."""
        row = pd.Series({"CL": 10.0, "NG": np.nan, "GC": np.nan})
        ranked = cross_sectional_rank(row)
        assert ranked["CL"] == pytest.approx(0.5)

    def test_all_nan(self) -> None:
        """All NaN input → all NaN output."""
        row = pd.Series({"CL": np.nan, "NG": np.nan})
        ranked = cross_sectional_rank(row)
        assert ranked.isna().all()

    def test_ties_average(self) -> None:
        """Tied values get average rank → value between 0 and 1."""
        row = pd.Series({"A": 5.0, "B": 5.0, "C": 10.0})
        ranked = cross_sectional_rank(row)
        # A and B are tied: raw ranks are 1 and 2 with average → 1.5
        # scaled: (1.5 - 1) / (3 - 1) = 0.25
        assert ranked["A"] == pytest.approx(0.25)
        assert ranked["B"] == pytest.approx(0.25)
        assert ranked["C"] == pytest.approx(1.0)

    def test_index_preserved(self) -> None:
        """Output index matches input index."""
        row = pd.Series({"X": 1.0, "Y": 2.0, "Z": 3.0})
        ranked = cross_sectional_rank(row)
        pd.testing.assert_index_equal(ranked.index, row.index)

    def test_two_values(self) -> None:
        """Two values: smaller → 0.0, larger → 1.0."""
        row = pd.Series({"A": 5.0, "B": 10.0})
        ranked = cross_sectional_rank(row)
        assert ranked["A"] == 0.0
        assert ranked["B"] == 1.0


# ---------------------------------------------------------------------------
# percentile_rank
# ---------------------------------------------------------------------------


class TestPercentileRank:
    def test_ascending_current_is_max(self) -> None:
        """Ascending series: current is always the max → percentile = 1.0."""
        s = pd.Series(
            range(1, 21),
            dtype=float,
            index=pd.date_range("2020-01-01", periods=20),
        )
        pct = percentile_rank(s, window=10)
        # First 9 are NaN (need 10 observations)
        assert pct.iloc[:9].isna().all()
        # From index 9 onward, current is always the max → percentile = 1.0
        assert (pct.iloc[9:] == 1.0).all()

    def test_descending_current_is_min(self) -> None:
        """Descending series: current is always the min → low percentile."""
        s = pd.Series(
            range(20, 0, -1),
            dtype=float,
            index=pd.date_range("2020-01-01", periods=20),
        )
        pct = percentile_rank(s, window=10)
        valid = pct.iloc[9:]
        # current is smallest in window → percentile = 1/window = 0.1
        assert (valid <= 0.15).all()

    def test_min_periods_nans(self) -> None:
        """Fewer than window observations → NaN."""
        s = pd.Series(
            range(15),
            dtype=float,
            index=pd.date_range("2020-01-01", periods=15),
        )
        pct = percentile_rank(s, window=10)
        assert pct.iloc[:9].isna().all()
        assert pct.iloc[9:].notna().all()

    def test_output_range(self) -> None:
        """All non-NaN outputs are in [0, 1]."""
        rng = np.random.default_rng(0)
        s = pd.Series(
            rng.standard_normal(100),
            index=pd.date_range("2020-01-01", periods=100),
        )
        pct = percentile_rank(s, window=20)
        valid = pct.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_nan_input_propagates(self) -> None:
        """NaN in input series does not crash; NaN current value → NaN output."""
        s = pd.Series(
            [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            index=pd.date_range("2020-01-01", periods=11),
        )
        # Should not raise
        pct = percentile_rank(s, window=10)
        # NaN at index 2 → NaN output
        assert pd.isna(pct.iloc[2])

    def test_output_index_preserved(self) -> None:
        """Output index matches input index."""
        idx = pd.date_range("2021-01-01", periods=15)
        s = pd.Series(range(15), dtype=float, index=idx)
        pct = percentile_rank(s, window=10)
        pd.testing.assert_index_equal(pct.index, idx)
