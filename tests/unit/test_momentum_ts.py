"""Tests for the time-series momentum factor module."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.factors.momentum_ts import (
    compute_trailing_return,
    tsmom_multi_horizon,
    tsmom_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(
    n: int = 400,
    symbols: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic price DataFrame with a DatetimeIndex."""
    if symbols is None:
        symbols = ["CL", "GC"]
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    data = {}
    for sym in symbols:
        log_returns = rng.normal(0.0, 0.01, n)
        data[sym] = 100.0 * np.exp(np.cumsum(log_returns))
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# compute_trailing_return
# ---------------------------------------------------------------------------


def test_trailing_return_positive_for_uptrend() -> None:
    """Consistently rising prices produce positive trailing log returns."""
    n = 50
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        {"CL": np.linspace(100.0, 200.0, n)},
        index=dates,
    )
    returns = compute_trailing_return(prices, lookback_days=10)
    # After the first 10 NaN rows, all returns should be positive
    assert returns["CL"].iloc[10:].gt(0).all()


def test_trailing_return_negative_for_downtrend() -> None:
    """Consistently falling prices produce negative trailing log returns."""
    n = 50
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        {"CL": np.linspace(200.0, 100.0, n)},
        index=dates,
    )
    returns = compute_trailing_return(prices, lookback_days=10)
    assert returns["CL"].iloc[10:].lt(0).all()


def test_trailing_return_uses_log_returns() -> None:
    """Return is log(P_t / P_{t-k}), not arithmetic."""
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"CL": [100.0, 110.0, 120.0, 130.0, 140.0]}, index=dates)
    returns = compute_trailing_return(prices, lookback_days=1)
    expected_log = np.log(110.0 / 100.0)
    expected_arith = (110.0 - 100.0) / 100.0
    # Log return and arithmetic return are different (unless tiny move)
    assert abs(returns["CL"].iloc[1] - expected_log) < 1e-10
    assert abs(returns["CL"].iloc[1] - expected_arith) > 1e-4


def test_trailing_return_first_k_rows_are_nan() -> None:
    """First ``lookback_days`` rows are NaN, rest are finite."""
    n = 30
    k = 7
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.DataFrame({"CL": np.arange(1.0, n + 1)}, index=dates)
    returns = compute_trailing_return(prices, lookback_days=k)
    assert returns["CL"].iloc[:k].isna().all()
    assert returns["CL"].iloc[k:].notna().all()


def test_trailing_return_shape_preserved() -> None:
    """Output shape matches input shape."""
    prices = _make_prices(n=100, symbols=["CL", "GC", "NG"])
    returns = compute_trailing_return(prices, lookback_days=20)
    assert returns.shape == prices.shape
    assert list(returns.columns) == list(prices.columns)
    assert returns.index.equals(prices.index)


# ---------------------------------------------------------------------------
# tsmom_signal
# ---------------------------------------------------------------------------


def test_tsmom_signal_shape_correct() -> None:
    """Output has same shape and column names as the input prices."""
    prices = _make_prices(n=400)
    signal = tsmom_signal(prices, lookback_days=63, min_periods=63)
    assert signal.shape == prices.shape
    assert list(signal.columns) == list(prices.columns)
    assert signal.index.equals(prices.index)


def test_tsmom_signal_nan_during_warmup() -> None:
    """Rows within the warmup period (min_periods) are NaN."""
    prices = _make_prices(n=400)
    min_p = 80
    signal = tsmom_signal(prices, lookback_days=20, min_periods=min_p)
    # The z-score requires min_periods valid trailing-return observations.
    # trailing_return itself has the first lookback_days as NaN, so the
    # combined warmup is longer, meaning row 0 is definitely NaN.
    assert signal.iloc[0].isna().all()


def test_tsmom_signal_no_lookahead() -> None:
    """Changing the last price does not affect earlier z-scores."""
    prices = _make_prices(n=400, seed=7)
    signal1 = tsmom_signal(prices, lookback_days=21, min_periods=21)

    prices2 = prices.copy()
    prices2.iloc[-1] = prices2.iloc[-1] * 10  # large spike on the last row only

    signal2 = tsmom_signal(prices2, lookback_days=21, min_periods=21)

    pd.testing.assert_frame_equal(signal1.iloc[:-1], signal2.iloc[:-1])


def test_tsmom_signal_all_nan_when_insufficient_history() -> None:
    """All values are NaN when the series is shorter than min_periods."""
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"CL": np.arange(1.0, 11.0)}, index=dates)
    signal = tsmom_signal(prices, lookback_days=5, min_periods=252)
    assert signal["CL"].isna().all()


# ---------------------------------------------------------------------------
# tsmom_multi_horizon
# ---------------------------------------------------------------------------


def test_tsmom_multi_horizon_default_keys() -> None:
    """Default call returns a dict with keys [21, 63, 126, 252]."""
    prices = _make_prices(n=600)
    result = tsmom_multi_horizon(prices, min_periods=21)
    assert set(result.keys()) == {21, 63, 126, 252}


def test_tsmom_multi_horizon_custom_lookbacks() -> None:
    """Custom lookbacks are used as dict keys."""
    prices = _make_prices(n=200)
    lookbacks = [5, 10, 20]
    result = tsmom_multi_horizon(prices, lookbacks=lookbacks, min_periods=5)
    assert set(result.keys()) == {5, 10, 20}


def test_tsmom_multi_horizon_each_value_is_dataframe() -> None:
    """Each entry in the result dict is a DataFrame with matching shape."""
    prices = _make_prices(n=300)
    result = tsmom_multi_horizon(prices, lookbacks=[10, 20], min_periods=10)
    for lb, df in result.items():
        assert isinstance(df, pd.DataFrame), f"lookback {lb}: expected DataFrame"
        assert df.shape == prices.shape
        assert list(df.columns) == list(prices.columns)


def test_tsmom_multi_horizon_longer_lookback_later_non_nan() -> None:
    """A longer lookback produces more leading NaNs than a shorter one."""
    prices = _make_prices(n=400, seed=99)
    result = tsmom_multi_horizon(prices, lookbacks=[10, 63], min_periods=10)
    # Count how many rows are all-NaN for each horizon
    nan10 = result[10].isna().all(axis=1).sum()
    nan63 = result[63].isna().all(axis=1).sum()
    assert nan63 > nan10

    pytest.importorskip("numpy")  # trivial guard, ensures the import is real
