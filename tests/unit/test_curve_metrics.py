"""Tests for curve metrics computed from daily tenor DataFrames.

All metric formulas are taken from configs/curve.yaml and must match exactly:
  slope:       (F12M - F1M) / F1M
  front_slope: (F3M - F1M) / F1M
  curvature:   F1M - 2*F6M + F12M
  carry:       (F1M - F2M) / F2M * 12
  term_carry:  (F1M - F12M) / F12M
"""

import pandas as pd
import pytest


def _make_curve(values: dict[str, list[float]]) -> pd.DataFrame:
    """Construct a minimal curve DataFrame.

    Parameters
    ----------
    values : dict[str, list[float]]
        Tenor column → list of prices.  All lists must have the same length.
    """
    n = len(next(iter(values.values())))
    idx = pd.date_range("2020-01-02", periods=n, freq="B", name="trade_date")
    return pd.DataFrame(values, index=idx)


class TestComputeSlope:
    def test_formula(self) -> None:
        """slope = (F12M - F1M) / F1M."""
        from commodity_curve_factors.curves.metrics import compute_slope

        curve = _make_curve({"F1M": [100.0, 110.0, 90.0], "F12M": [110.0, 115.0, 85.0]})
        result = compute_slope(curve)

        expected = (curve["F12M"] - curve["F1M"]) / curve["F1M"]
        pd.testing.assert_series_equal(result, expected)

    def test_backwardation_negative(self) -> None:
        """Backwardation (F12M < F1M) → negative slope."""
        from commodity_curve_factors.curves.metrics import compute_slope

        curve = _make_curve({"F1M": [100.0], "F12M": [90.0]})
        result = compute_slope(curve)
        assert result.iloc[0] < 0

    def test_contango_positive(self) -> None:
        """Contango (F12M > F1M) → positive slope."""
        from commodity_curve_factors.curves.metrics import compute_slope

        curve = _make_curve({"F1M": [100.0], "F12M": [110.0]})
        result = compute_slope(curve)
        assert result.iloc[0] > 0


class TestComputeFrontSlope:
    def test_formula(self) -> None:
        """front_slope = (F3M - F1M) / F1M."""
        from commodity_curve_factors.curves.metrics import compute_front_slope

        curve = _make_curve({"F1M": [100.0, 95.0], "F3M": [102.0, 97.0]})
        result = compute_front_slope(curve)

        expected = (curve["F3M"] - curve["F1M"]) / curve["F1M"]
        pd.testing.assert_series_equal(result, expected)


class TestComputeCurvature:
    def test_formula(self) -> None:
        """curvature = F1M - 2*F6M + F12M."""
        from commodity_curve_factors.curves.metrics import compute_curvature

        curve = _make_curve({"F1M": [100.0, 105.0], "F6M": [103.0, 107.0], "F12M": [106.0, 109.0]})
        result = compute_curvature(curve)

        expected = curve["F1M"] - 2 * curve["F6M"] + curve["F12M"]
        pd.testing.assert_series_equal(result, expected)

    def test_flat_curve_zero_curvature(self) -> None:
        """Flat curve (all tenors equal) → curvature = 0."""
        from commodity_curve_factors.curves.metrics import compute_curvature

        curve = _make_curve({"F1M": [100.0], "F6M": [100.0], "F12M": [100.0]})
        result = compute_curvature(curve)
        assert abs(result.iloc[0]) < 1e-10


class TestComputeCarry:
    def test_formula_exact(self) -> None:
        """carry = (F1M - F2M) / F2M * 12. With F1M=100, F2M=95 → ~0.6316."""
        from commodity_curve_factors.curves.metrics import compute_carry

        curve = _make_curve({"F1M": [100.0], "F2M": [95.0]})
        result = compute_carry(curve)

        expected_carry = (100.0 - 95.0) / 95.0 * 12
        assert abs(result.iloc[0] - expected_carry) < 1e-10, (
            f"Expected {expected_carry:.6f}, got {result.iloc[0]:.6f}"
        )

    def test_approximate_value(self) -> None:
        """F1M=100, F2M=95 → carry ≈ 0.6316."""
        from commodity_curve_factors.curves.metrics import compute_carry

        curve = _make_curve({"F1M": [100.0], "F2M": [95.0]})
        result = compute_carry(curve)
        assert abs(result.iloc[0] - 0.6316) < 0.001, f"Expected ~0.6316, got {result.iloc[0]:.6f}"

    def test_backwardation_positive_carry(self) -> None:
        """Backwardated market (F1M > F2M) → positive carry."""
        from commodity_curve_factors.curves.metrics import compute_carry

        curve = _make_curve({"F1M": [100.0], "F2M": [98.0]})
        result = compute_carry(curve)
        assert result.iloc[0] > 0


class TestComputeTermCarry:
    def test_formula(self) -> None:
        """term_carry = (F1M - F12M) / F12M."""
        from commodity_curve_factors.curves.metrics import compute_term_carry

        curve = _make_curve({"F1M": [100.0, 95.0], "F12M": [110.0, 90.0]})
        result = compute_term_carry(curve)

        expected = (curve["F1M"] - curve["F12M"]) / curve["F12M"]
        pd.testing.assert_series_equal(result, expected)

    def test_contango_negative_term_carry(self) -> None:
        """Contango (F1M < F12M) → negative term carry."""
        from commodity_curve_factors.curves.metrics import compute_term_carry

        curve = _make_curve({"F1M": [90.0], "F12M": [100.0]})
        result = compute_term_carry(curve)
        assert result.iloc[0] < 0


class TestComputeAllMetrics:
    def test_keys_and_shapes(self) -> None:
        """compute_all_metrics returns all 5 metric keys; each DataFrame has 2 columns."""
        from commodity_curve_factors.curves.metrics import compute_all_metrics

        n = 5
        curve = _make_curve(
            {
                "F1M": [100.0] * n,
                "F2M": [101.0] * n,
                "F3M": [102.0] * n,
                "F6M": [104.0] * n,
                "F9M": [106.0] * n,
                "F12M": [108.0] * n,
            }
        )
        curves = {"CL": curve, "NG": curve.copy()}

        result = compute_all_metrics(curves)

        expected_keys = {"slope", "front_slope", "curvature", "carry", "term_carry"}
        assert set(result.keys()) == expected_keys, (
            f"Expected metric keys {expected_keys}, got {set(result.keys())}"
        )

        for name, df in result.items():
            assert isinstance(df, pd.DataFrame), f"Expected DataFrame for {name}"
            assert set(df.columns) == {"CL", "NG"}, (
                f"Expected columns {{'CL', 'NG'}} for metric {name}, got {set(df.columns)}"
            )
            assert len(df) == n, f"Expected {n} rows for {name}, got {len(df)}"

    def test_missing_tenor_logged_not_raised(self, caplog: "pytest.LogCaptureFixture") -> None:
        """Missing tenor column in one commodity → warning logged, other commodity fine."""
        import logging

        from commodity_curve_factors.curves.metrics import compute_all_metrics

        # NG curve is missing F2M → compute_carry will fail for NG
        n = 3
        cl_curve = _make_curve(
            {
                "F1M": [100.0] * n,
                "F2M": [101.0] * n,
                "F3M": [102.0] * n,
                "F6M": [104.0] * n,
                "F9M": [106.0] * n,
                "F12M": [108.0] * n,
            }
        )
        ng_curve = _make_curve(
            {
                "F1M": [3.0] * n,
                "F3M": [3.1] * n,
                "F6M": [3.2] * n,
                "F9M": [3.3] * n,
                "F12M": [3.4] * n,
                # F2M is intentionally missing
            }
        )

        with caplog.at_level(logging.WARNING):
            result = compute_all_metrics({"CL": cl_curve, "NG": ng_curve})

        # CL carry should be present
        assert "CL" in result["carry"].columns
        # A warning should have been emitted for NG
        warning_text = caplog.text
        assert "NG" in warning_text or "carry" in warning_text.lower(), (
            f"Expected a warning about NG or carry, got: {caplog.text!r}"
        )
