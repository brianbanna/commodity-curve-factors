"""Tests for the volatility regime factor module."""

import numpy as np
import pandas as pd

from commodity_curve_factors.factors.volatility import realized_volatility, vol_regime_ratio


def _make_returns(
    n_days: int = 600,
    symbols: list[str] | None = None,
    start: str = "2018-01-01",
    seed: int = 42,
    daily_vol: float = 0.01,
) -> pd.DataFrame:
    """Build synthetic daily log returns DataFrame."""
    if symbols is None:
        symbols = ["CL", "GC"]
    dates = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(seed)
    data = rng.normal(0, daily_vol, (n_days, len(symbols)))
    return pd.DataFrame(data, index=dates, columns=symbols)


# ---------------------------------------------------------------------------
# realized_volatility
# ---------------------------------------------------------------------------


def test_realized_vol_annualized() -> None:
    """Check that the annualization factor sqrt(252) is applied correctly."""
    n_days = 30
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    # Constant daily return of 0.01 → std = 0 → vol = 0
    rets = pd.DataFrame({"CL": [0.01] * n_days}, index=dates)
    vol = realized_volatility(rets, window=20)
    assert vol["CL"].dropna().abs().max() < 1e-10, "Constant returns should yield near-zero vol"

    # Non-constant: known daily vol ≈ 0.02 → annualized ≈ 0.02 * sqrt(252)
    rng = np.random.default_rng(1)
    daily_vol = 0.02
    ret_data = rng.normal(0, daily_vol, n_days)
    rets2 = pd.DataFrame({"CL": ret_data}, index=dates)
    vol2 = realized_volatility(rets2, window=20)
    last_vol = vol2["CL"].dropna().iloc[-1]
    expected = float(np.std(ret_data[-20:], ddof=1) * np.sqrt(252))
    assert abs(last_vol - expected) < 0.001, (
        f"Annualized vol mismatch: got {last_vol:.4f}, expected ≈ {expected:.4f}"
    )


def test_realized_vol_shape() -> None:
    """Output has the same shape and columns as input returns."""
    rets = _make_returns(n_days=100, symbols=["CL", "GC", "ZC"])
    vol = realized_volatility(rets, window=20)
    assert vol.shape == rets.shape
    assert list(vol.columns) == ["CL", "GC", "ZC"]
    # First window - 1 rows should be NaN
    assert vol.iloc[:19].isna().all().all()
    assert vol.iloc[19:].notna().any().any()


# ---------------------------------------------------------------------------
# vol_regime_ratio
# ---------------------------------------------------------------------------


def test_vol_ratio_shape() -> None:
    """Output has the same shape and columns as input returns."""
    rets = _make_returns(n_days=700, symbols=["CL", "GC"])
    factor = vol_regime_ratio(rets, short_window=20, long_window=100, min_periods=120)
    assert factor.shape == rets.shape
    assert list(factor.columns) == ["CL", "GC"]


def test_vol_ratio_elevated_in_crisis() -> None:
    """Injecting a volatility spike should produce a clearly elevated ratio.

    We construct a series of calm returns, then inject a period of high
    volatility. During the spike, the short-term vol should exceed the
    long-term vol, and the z-scored ratio should be positive (elevated).
    """
    n_days = 700
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    rng = np.random.default_rng(3)

    # 600 days of normal (calm) returns
    calm = rng.normal(0, 0.005, 600)
    # 100 days of crisis returns (10x higher vol)
    crisis = rng.normal(0, 0.05, 100)
    combined = np.concatenate([calm, crisis])

    rets = pd.DataFrame({"CL": combined}, index=dates)

    factor = vol_regime_ratio(rets, short_window=20, long_window=100, min_periods=150)
    cl = factor["CL"].dropna()

    # The crisis period starts around row 600; check that the factor is
    # elevated (positive z-score) during the crisis relative to the calm period.
    # Use the last 30 rows of the crisis as the "crisis" window.
    crisis_start = dates[600]
    crisis_vals = cl[cl.index >= crisis_start]
    calm_vals = cl[cl.index < crisis_start].iloc[-100:]

    assert len(crisis_vals) > 0, "No crisis-period values in factor"
    assert crisis_vals.mean() > calm_vals.mean(), (
        f"Vol ratio should be elevated during crisis; "
        f"crisis_mean={crisis_vals.mean():.3f}, calm_mean={calm_vals.mean():.3f}"
    )


def test_vol_ratio_no_lookahead() -> None:
    """Changing the last return should not affect earlier factor values."""
    rets = _make_returns(n_days=600)
    factor1 = vol_regime_ratio(rets, short_window=20, long_window=100, min_periods=120)

    rets2 = rets.copy()
    rets2.iloc[-1] = 99.0  # extreme spike on last day only

    factor2 = vol_regime_ratio(rets2, short_window=20, long_window=100, min_periods=120)

    # All rows before the last must be identical (expanding z-score is
    # causal: the spike at t affects only t, not t-1, t-2, ...).
    # Note: expanding std at last row uses all data INCLUDING the spike,
    # so the z-score at the last row changes. But earlier rows must be
    # unchanged because neither the rolling vol nor the expanding mean/std
    # at those positions are affected by a future spike.
    pd.testing.assert_frame_equal(
        factor1.iloc[:-1],
        factor2.iloc[:-1],
        check_exact=False,
        rtol=1e-9,
    )
