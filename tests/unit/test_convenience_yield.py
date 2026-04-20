"""Tests for curves/convenience_yield.py."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.curves.convenience_yield import (
    compute_convenience_yield,
    estimate_storage_cost,
    monthly_convenience_yield,
)


@pytest.fixture()
def dates():
    return pd.bdate_range("2010-01-04", periods=500)


@pytest.fixture()
def curves(dates):
    """Synthetic curves: 3 commodities, backwardation for CL, contango for NG."""
    rng = np.random.default_rng(42)
    n = len(dates)
    cl_f1m = 80.0 + rng.standard_normal(n) * 2
    cl_f6m = cl_f1m - 2.0 + rng.standard_normal(n) * 0.5  # backwardation
    ng_f1m = 4.0 + rng.standard_normal(n) * 0.3
    ng_f6m = ng_f1m + 0.3 + rng.standard_normal(n) * 0.1  # contango
    gc_f1m = 1500.0 + rng.standard_normal(n) * 20
    gc_f6m = gc_f1m + 5.0 + rng.standard_normal(n) * 3  # slight contango
    return {
        "CL": pd.DataFrame({"F1M": cl_f1m, "F6M": cl_f6m}, index=dates),
        "NG": pd.DataFrame({"F1M": ng_f1m, "F6M": ng_f6m}, index=dates),
        "GC": pd.DataFrame({"F1M": gc_f1m, "F6M": gc_f6m}, index=dates),
    }


@pytest.fixture()
def risk_free(dates):
    """Constant 2% annualised risk-free rate as daily series."""
    return pd.Series(2.0, index=dates, name="DGS3MO")


def test_estimate_storage_cost_returns_dict(curves):
    result = estimate_storage_cost(curves, is_end="2011-12-31")
    assert isinstance(result, dict)
    assert set(result.keys()) == {"CL", "NG", "GC"}
    for v in result.values():
        assert isinstance(v, float)


def test_storage_cost_positive_for_contango(curves):
    """Contango commodities should have positive storage cost estimates."""
    result = estimate_storage_cost(curves, is_end="2011-12-31")
    assert result["NG"] > 0, "NG (contango) should have positive storage cost"


def test_compute_convenience_yield_shape(curves, risk_free):
    storage = {"CL": 0.02, "NG": 0.05, "GC": 0.01}
    cy = compute_convenience_yield(curves, risk_free, storage, tenor="F6M")
    assert isinstance(cy, pd.DataFrame)
    assert set(cy.columns) == {"CL", "NG", "GC"}
    assert len(cy) == 500


def test_convenience_yield_higher_for_backwardation(curves, risk_free):
    """CL (backwardated) should have higher mean CY than NG (contango)."""
    storage = {"CL": 0.02, "NG": 0.05, "GC": 0.01}
    cy = compute_convenience_yield(curves, risk_free, storage, tenor="F6M")
    assert cy["CL"].mean() > cy["NG"].mean(), (
        "Backwardated CL should have higher convenience yield than contango NG"
    )


def test_convenience_yield_handles_nan(curves, risk_free):
    """NaN in curve data should propagate as NaN in CY, not crash."""
    curves["CL"].iloc[0, 1] = np.nan  # NaN in F6M
    storage = {"CL": 0.02, "NG": 0.05, "GC": 0.01}
    cy = compute_convenience_yield(curves, risk_free, storage, tenor="F6M")
    assert np.isnan(cy["CL"].iloc[0])


def test_monthly_convenience_yield_reduces_rows(curves, risk_free):
    storage = {"CL": 0.02, "NG": 0.05, "GC": 0.01}
    daily_cy = compute_convenience_yield(curves, risk_free, storage, tenor="F6M")
    monthly = monthly_convenience_yield(daily_cy)
    assert len(monthly) < len(daily_cy)
    assert set(monthly.columns) == set(daily_cy.columns)


def test_monthly_convenience_yield_index_is_month_end(curves, risk_free):
    storage = {"CL": 0.02, "NG": 0.05, "GC": 0.01}
    daily_cy = compute_convenience_yield(curves, risk_free, storage, tenor="F6M")
    monthly = monthly_convenience_yield(daily_cy)
    # All index dates should be the last day of each month group
    for dt in monthly.index:
        assert dt.day >= 28 or dt == monthly.index[-1]
