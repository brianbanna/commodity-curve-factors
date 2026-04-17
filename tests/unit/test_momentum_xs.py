"""Tests for the cross-sectional momentum factor module."""

import numpy as np
import pandas as pd

from commodity_curve_factors.factors.momentum_xs import xsmom_signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_diverging_prices(n: int = 200) -> pd.DataFrame:
    """Three commodities with clearly different trend speeds.

    CL: strong uptrend, GC: flat, NG: strong downtrend.
    """
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "CL": np.linspace(100.0, 300.0, n),  # strong uptrend
            "GC": np.full(n, 100.0),  # flat
            "NG": np.linspace(100.0, 10.0, n),  # strong downtrend
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# xsmom_signal
# ---------------------------------------------------------------------------


def test_xsmom_output_in_unit_interval() -> None:
    """All non-NaN output values are in [0, 1]."""
    prices = _make_diverging_prices()
    signal = xsmom_signal(prices, lookback_days=10)
    valid = signal.stack().dropna()
    assert (valid >= 0.0).all(), "Some values are below 0"
    assert (valid <= 1.0).all(), "Some values are above 1"


def test_xsmom_best_performer_gets_rank_one() -> None:
    """The commodity with the highest trailing return ranks at 1.0."""
    prices = _make_diverging_prices(n=200)
    signal = xsmom_signal(prices, lookback_days=10)
    # After the first 10 NaN rows, CL (uptrend) should always be rank 1.0
    cl_signal = signal["CL"].dropna()
    assert cl_signal.eq(1.0).all(), "CL (best uptrend) should always rank 1.0"


def test_xsmom_worst_performer_gets_rank_zero() -> None:
    """The commodity with the lowest trailing return ranks at 0.0."""
    prices = _make_diverging_prices(n=200)
    signal = xsmom_signal(prices, lookback_days=10)
    ng_signal = signal["NG"].dropna()
    assert ng_signal.eq(0.0).all(), "NG (strong downtrend) should always rank 0.0"


def test_xsmom_no_lookahead() -> None:
    """Changing the last price does not affect earlier cross-sectional ranks."""
    rng = np.random.default_rng(11)
    n = 150
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        {
            "CL": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
            "GC": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
            "NG": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        },
        index=dates,
    )
    signal1 = xsmom_signal(prices, lookback_days=10)

    prices2 = prices.copy()
    prices2.iloc[-1] = prices2.iloc[-1] * 100  # extreme spike on last row only

    signal2 = xsmom_signal(prices2, lookback_days=10)

    pd.testing.assert_frame_equal(signal1.iloc[:-1], signal2.iloc[:-1])


def test_xsmom_handles_nan_prices() -> None:
    """NaN prices produce NaN ranks; non-NaN prices still get valid ranks."""
    n = 50
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        {
            "CL": np.linspace(100.0, 150.0, n),
            "GC": np.linspace(100.0, 80.0, n),
            "NG": [np.nan] * n,  # always NaN
        },
        index=dates,
    )
    signal = xsmom_signal(prices, lookback_days=5)

    # NG is always NaN — ranks should be NaN
    assert signal["NG"].isna().all(), "All-NaN commodity should yield all-NaN ranks"

    # CL and GC should have valid values after the warmup period
    cl_valid = signal["CL"].dropna()
    gc_valid = signal["GC"].dropna()
    assert len(cl_valid) > 0
    assert len(gc_valid) > 0
    assert (cl_valid >= 0.0).all() and (cl_valid <= 1.0).all()
    assert (gc_valid >= 0.0).all() and (gc_valid <= 1.0).all()


def test_xsmom_shape_and_columns_preserved() -> None:
    """Output shape, index, and column names match the input prices."""
    rng = np.random.default_rng(55)
    n = 100
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        {sym: 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for sym in ["CL", "GC", "NG"]},
        index=dates,
    )
    signal = xsmom_signal(prices, lookback_days=20)
    assert signal.shape == prices.shape
    assert list(signal.columns) == list(prices.columns)
    assert signal.index.equals(prices.index)
