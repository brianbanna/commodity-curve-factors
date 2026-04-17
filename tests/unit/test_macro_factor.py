"""Tests for the macro exposure factor module."""

import numpy as np
import pandas as pd

from commodity_curve_factors.factors.macro import compute_macro_factor


def _make_returns(
    n_days: int = 400,
    n_assets: int = 3,
    symbols: list[str] | None = None,
    start: str = "2018-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Build synthetic daily log returns DataFrame."""
    if symbols is None:
        symbols = [f"C{i}" for i in range(n_assets)]
    dates = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 0.01, (n_days, len(symbols)))
    return pd.DataFrame(data, index=dates, columns=symbols)


def _make_macro(
    n_days: int = 400,
    start: str = "2018-01-01",
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Build a synthetic macro_data dict with usd_index, dgs10, t5yie."""
    dates = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(seed)

    # USD index as a level series (cumulative)
    usd_level = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n_days)))
    usd_df = pd.DataFrame({"value": usd_level}, index=dates)

    # 10Y yield as a level series
    rate_level = 3.0 + np.cumsum(rng.normal(0, 0.02, n_days))
    rate_df = pd.DataFrame({"value": rate_level}, index=dates)

    # Inflation breakeven as a level series
    infl_level = 2.0 + np.cumsum(rng.normal(0, 0.01, n_days))
    infl_df = pd.DataFrame({"value": infl_level}, index=dates)

    return {
        "usd_index": usd_df,
        "dgs10": rate_df,
        "t5yie": infl_df,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_macro_factor_shape() -> None:
    """Output DataFrame has correct shape: same rows as returns, same columns."""
    symbols = ["CL", "GC", "ZC"]
    rets = _make_returns(n_days=400, symbols=symbols)
    macro = _make_macro(n_days=400)

    factor = compute_macro_factor(rets, macro, window=60, min_periods=20)

    assert isinstance(factor, pd.DataFrame)
    assert list(factor.columns) == symbols
    assert len(factor) == len(rets)
    assert isinstance(factor.index, pd.DatetimeIndex)


def test_macro_factor_no_lookahead() -> None:
    """Changing the last row of returns should not affect earlier factor values."""
    symbols = ["CL", "GC"]
    rets = _make_returns(n_days=400, symbols=symbols)
    macro = _make_macro(n_days=400)

    factor1 = compute_macro_factor(rets, macro, window=60, min_periods=20)

    rets2 = rets.copy()
    rets2.iloc[-1] = 99.0  # extreme spike on last day

    factor2 = compute_macro_factor(rets2, macro, window=60, min_periods=20)

    # All rows except the last should be identical
    pd.testing.assert_frame_equal(
        factor1.iloc[:-1],
        factor2.iloc[:-1],
        check_exact=False,
        rtol=1e-9,
    )


def test_macro_factor_handles_missing_macro() -> None:
    """When macro_data is empty, factor returns all-NaN with correct shape."""
    symbols = ["CL", "GC"]
    rets = _make_returns(n_days=200, symbols=symbols)

    factor = compute_macro_factor(rets, macro_data={}, window=60, min_periods=20)

    assert isinstance(factor, pd.DataFrame)
    # Should return all NaN (graceful degradation)
    assert factor.isna().all().all()


def test_macro_factor_partial_macro_graceful() -> None:
    """If only one macro series is available, factor still runs without error."""
    symbols = ["CL"]
    rets = _make_returns(n_days=300, symbols=symbols)
    # Only provide usd_index; missing dgs10 and t5yie
    macro = {"usd_index": _make_macro(n_days=300)["usd_index"]}

    factor = compute_macro_factor(rets, macro, window=60, min_periods=20)

    assert isinstance(factor, pd.DataFrame)
    assert "CL" in factor.columns
    # Some values should be non-NaN (with just r_usd)
    assert factor["CL"].notna().any()


def test_macro_factor_positive_for_usd_sensitive_commodity() -> None:
    """A commodity with known positive USD beta should have predictable factor behavior.

    We construct a synthetic commodity whose returns are exactly r_usd + noise.
    The rolling OLS should recover beta_usd ≈ 1.0, and the macro signal should
    be positively correlated with r_usd.

    Note on sign convention: the signal = beta_usd * r_usd(t) + beta_rate * d_rate(t) + ...
    A commodity that moves up when USD strengthens (positive beta_usd) will have
    a positive signal when USD is rising, and negative signal when USD is falling.
    """
    n_days = 400
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    rng = np.random.default_rng(99)

    # Build USD series
    usd_level = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n_days)))
    usd_df = pd.DataFrame({"value": usd_level}, index=dates)
    r_usd = np.log(usd_level / np.roll(usd_level, 1))
    r_usd[0] = np.nan

    # Build commodity = r_usd + small noise (high beta_usd ≈ 1)
    noise = rng.normal(0, 0.001, n_days)
    r_commodity = r_usd + noise
    rets = pd.DataFrame({"CL": r_commodity}, index=dates)

    # Flat macro for rate and infl
    rate_df = pd.DataFrame({"value": np.zeros(n_days) + 3.0}, index=dates)
    infl_df = pd.DataFrame({"value": np.zeros(n_days) + 2.0}, index=dates)
    macro = {"usd_index": usd_df, "dgs10": rate_df, "t5yie": infl_df}

    factor = compute_macro_factor(rets, macro, window=100, min_periods=30)
    cl = factor["CL"].dropna()

    r_usd_aligned = pd.Series(r_usd, index=dates).reindex(cl.index).dropna()
    cl_aligned = cl.reindex(r_usd_aligned.index).dropna()

    # Correlation between factor signal and r_usd should be positive
    corr = cl_aligned.corr(r_usd_aligned)
    assert corr > 0.3, (
        f"Expected positive correlation between macro factor and r_usd for "
        f"a USD-sensitive commodity, got corr={corr:.3f}"
    )
