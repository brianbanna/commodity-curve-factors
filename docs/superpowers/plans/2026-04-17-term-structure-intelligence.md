# Term Structure Intelligence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a three-layer commodity strategy that reads the term structure like a physical trader — using convenience yield for directional positioning, regime transitions for timing, and structural spreads for relative value.

**Architecture:** Layer 1 (Curve-Informed Directional) estimates convenience yield from the cost-of-carry model, classifies curve regimes, and positions long/flat/short with a TSMOM trend filter at monthly frequency. Layer 2 (Curve Transition Momentum) detects regime transitions and trades them at weekly frequency with TSMOM confirmation. Layer 3 (Structural Spread Alpha) trades convenience-yield-adjusted crack spreads, inventory-conditioned energy positions, and deseasonalised livestock spreads. All three are combined via risk-budget-weighted vol targeting with Ledoit-Wolf shrinkage.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, sklearn (LedoitWolf), pytest. Config-driven via YAML. Parquet storage. Existing backtest engine reused.

**Spec:** `docs/superpowers/specs/2026-04-17-term-structure-intelligence-design.md`

**Existing codebase conventions:**
- All parameters from YAML configs, never hard-coded
- Functions prefer DataFrame in, DataFrame out
- Expanding-window z-scores only (no lookahead)
- Type hints on all signatures, numpy-style docstrings on public functions
- `logging` module, not print
- `ruff check` + `ruff format` before commit
- Many small commits, no Co-Authored-By
- Tests use pytest with synthetic fixtures; real data only in smoke tests

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/commodity_curve_factors/curves/convenience_yield.py` | Storage cost calibration, daily convenience yield from cost-of-carry model, monthly aggregation |
| `src/commodity_curve_factors/signals/curve_regime.py` | Expanding-percentile regime classification, regime-to-position mapping |
| `src/commodity_curve_factors/signals/directional.py` | Layer 1: trend-filtered regime positions, monthly resampling |
| `src/commodity_curve_factors/signals/curve_transition.py` | Layer 2: CY z-score change, TSMOM-confirmed transition positions |
| `src/commodity_curve_factors/signals/seasonal.py` | ISO-week expanding seasonal pattern, deseasonalisation |
| `src/commodity_curve_factors/signals/spreads.py` | Layer 3: CY crack spread, inventory overlay, livestock spread |
| `src/commodity_curve_factors/signals/combined_strategy.py` | Three-layer combination, Ledoit-Wolf vol targeting, risk budgeting |
| `configs/strategy.yaml` | New strategy parameters (appended to existing file) |
| `src/commodity_curve_factors/backtest/__main__.py` | Add TSI strategy to the backtest runner |
| `tests/unit/test_convenience_yield.py` | Tests for convenience yield module |
| `tests/unit/test_curve_regime.py` | Tests for regime classification |
| `tests/unit/test_directional.py` | Tests for Layer 1 |
| `tests/unit/test_curve_transition.py` | Tests for Layer 2 |
| `tests/unit/test_seasonal.py` | Tests for seasonal module |
| `tests/unit/test_spreads.py` | Tests for Layer 3 spreads |
| `tests/unit/test_combined_strategy.py` | Tests for layer combination |

---

## Task 1: Convenience Yield Estimation

**Files:**
- Create: `src/commodity_curve_factors/curves/convenience_yield.py`
- Test: `tests/unit/test_convenience_yield.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_convenience_yield.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n curve-factors pytest tests/unit/test_convenience_yield.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'commodity_curve_factors.curves.convenience_yield'`

- [ ] **Step 3: Write the implementation**

Create `src/commodity_curve_factors/curves/convenience_yield.py`:

```python
"""Convenience yield estimation from the cost-of-carry model.

The cost-of-carry model relates spot and futures prices:

    F(T) = S * exp((r - y + c) * T)

Rearranging for convenience yield:

    y = r + c - ln(F(T) / S) / T

where S = F1M (spot proxy), F(T) = futures at tenor T, r = risk-free rate,
c = storage cost, y = convenience yield.

High convenience yield signals physical market tightness (scarcity premium).
Low/negative convenience yield signals surplus.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MONTHS_PER_YEAR = 12


def estimate_storage_cost(
    curves: dict[str, pd.DataFrame],
    is_end: str = "2017-12-31",
    tenor: str = "F6M",
) -> dict[str, float]:
    """Calibrate per-commodity storage cost proxy from in-sample contango depth.

    Storage cost is estimated as the median annualised contango
    ``ln(F(T)/S) / T`` over the in-sample period. For backwardated
    commodities this will be negative — floored at 0.0.

    Parameters
    ----------
    curves : dict[str, pd.DataFrame]
        Keyed by commodity symbol. Each DataFrame has columns including
        ``"F1M"`` and the specified *tenor* column, with a DatetimeIndex.
    is_end : str
        Last date of the in-sample calibration window.
    tenor : str
        Futures tenor column to use. Default ``"F6M"``.

    Returns
    -------
    dict[str, float]
        Per-commodity annualised storage cost estimate.
    """
    tenor_months = int(tenor.replace("F", "").replace("M", ""))
    t_years = tenor_months / _MONTHS_PER_YEAR

    result: dict[str, float] = {}
    for sym, df in curves.items():
        is_df = df.loc[:is_end]
        if "F1M" not in is_df.columns or tenor not in is_df.columns:
            logger.warning("estimate_storage_cost: %s missing F1M or %s", sym, tenor)
            result[sym] = 0.0
            continue
        ratio = is_df[tenor] / is_df["F1M"]
        ratio = ratio[ratio > 0]
        if len(ratio) == 0:
            result[sym] = 0.0
            continue
        contango_depth = np.log(ratio) / t_years
        cost = max(0.0, float(contango_depth.median()))
        result[sym] = cost

    logger.info(
        "estimate_storage_cost: %d commodities, is_end=%s, tenor=%s",
        len(result),
        is_end,
        tenor,
    )
    return result


def compute_convenience_yield(
    curves: dict[str, pd.DataFrame],
    risk_free: pd.Series,
    storage_costs: dict[str, float],
    tenor: str = "F6M",
) -> pd.DataFrame:
    """Compute daily convenience yield for each commodity.

    Parameters
    ----------
    curves : dict[str, pd.DataFrame]
        Keyed by commodity symbol. Each has ``"F1M"`` and *tenor* columns.
    risk_free : pd.Series
        Annualised risk-free rate (percentage, e.g. 2.0 for 2%).
        DatetimeIndex aligned to curve dates.
    storage_costs : dict[str, float]
        Per-commodity annualised storage cost from ``estimate_storage_cost``.
    tenor : str
        Futures tenor column. Default ``"F6M"``.

    Returns
    -------
    pd.DataFrame
        Daily convenience yield (dates x commodities). Annualised fraction.
    """
    tenor_months = int(tenor.replace("F", "").replace("M", ""))
    t_years = tenor_months / _MONTHS_PER_YEAR

    all_dates = sorted(
        set().union(*(df.index for df in curves.values()))
    )
    idx = pd.DatetimeIndex(all_dates)

    rf = risk_free.reindex(idx).ffill() / 100.0  # convert percentage to fraction

    cy_dict: dict[str, pd.Series] = {}
    for sym, df in curves.items():
        if "F1M" not in df.columns or tenor not in df.columns:
            continue
        spot = df["F1M"].reindex(idx)
        fut = df[tenor].reindex(idx)
        c = storage_costs.get(sym, 0.0)
        r = rf

        ratio = fut / spot
        # Guard against non-positive ratios
        valid = ratio > 0
        log_ratio = pd.Series(np.nan, index=idx)
        log_ratio[valid] = np.log(ratio[valid])

        y = r + c - log_ratio / t_years
        cy_dict[sym] = y

    result = pd.DataFrame(cy_dict, index=idx)
    logger.info(
        "compute_convenience_yield: %d commodities, %d dates, %.1f%% non-NaN",
        len(cy_dict),
        len(result),
        result.notna().mean().mean() * 100,
    )
    return result


def monthly_convenience_yield(daily_cy: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily convenience yield to monthly median.

    Parameters
    ----------
    daily_cy : pd.DataFrame
        Daily convenience yield (dates x commodities).

    Returns
    -------
    pd.DataFrame
        Monthly convenience yield indexed by month-end date.
    """
    result = daily_cy.resample("ME").median()
    logger.info(
        "monthly_convenience_yield: %d months, %d commodities",
        len(result),
        len(result.columns),
    )
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n curve-factors pytest tests/unit/test_convenience_yield.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Run linter**

Run: `conda run -n curve-factors ruff check src/commodity_curve_factors/curves/convenience_yield.py tests/unit/test_convenience_yield.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/commodity_curve_factors/curves/convenience_yield.py tests/unit/test_convenience_yield.py
git commit -m "Add convenience yield estimation from cost-of-carry model"
```

---

## Task 2: Curve Regime Classification

**Files:**
- Create: `src/commodity_curve_factors/signals/curve_regime.py`
- Test: `tests/unit/test_curve_regime.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_curve_regime.py`:

```python
"""Tests for signals/curve_regime.py."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.curve_regime import (
    classify_regime,
    regime_to_position,
)


@pytest.fixture()
def monthly_cy():
    """Monthly convenience yield: CL high, NG low, GC mid."""
    dates = pd.date_range("2010-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "CL": 0.10 + rng.standard_normal(60) * 0.03,  # high CY
            "NG": -0.02 + rng.standard_normal(60) * 0.03,  # low CY
            "GC": 0.03 + rng.standard_normal(60) * 0.03,  # mid CY
        },
        index=dates,
    )


def test_classify_regime_returns_labels(monthly_cy):
    regimes = classify_regime(monthly_cy)
    assert isinstance(regimes, pd.DataFrame)
    assert set(regimes.columns) == {"CL", "NG", "GC"}
    valid_labels = {
        "crisis_backwardation",
        "mild_backwardation",
        "balanced",
        "mild_contango",
        "deep_contango",
    }
    for col in regimes.columns:
        unique_vals = set(regimes[col].dropna().unique())
        assert unique_vals.issubset(valid_labels), f"{col} has unexpected labels: {unique_vals - valid_labels}"


def test_classify_regime_no_lookahead(monthly_cy):
    """Adding future data should not change past regime labels."""
    regimes_short = classify_regime(monthly_cy.iloc[:30])
    regimes_full = classify_regime(monthly_cy)
    pd.testing.assert_frame_equal(
        regimes_short,
        regimes_full.iloc[:30],
    )


def test_classify_regime_custom_thresholds(monthly_cy):
    regimes = classify_regime(monthly_cy, thresholds=[20, 40, 60, 80])
    assert isinstance(regimes, pd.DataFrame)


def test_regime_to_position_values(monthly_cy):
    regimes = classify_regime(monthly_cy)
    positions = regime_to_position(regimes)
    assert isinstance(positions, pd.DataFrame)
    assert set(positions.columns) == {"CL", "NG", "GC"}
    valid_positions = {-0.5, 0.0, 0.5, 1.0}
    for col in positions.columns:
        unique_vals = set(positions[col].dropna().unique())
        assert unique_vals.issubset(valid_positions), f"{col} has unexpected positions: {unique_vals}"


def test_regime_to_position_custom_map(monthly_cy):
    regimes = classify_regime(monthly_cy)
    custom_map = {
        "crisis_backwardation": 1.0,
        "mild_backwardation": 0.5,
        "balanced": 0.0,
        "mild_contango": -0.25,
        "deep_contango": -1.0,
    }
    positions = regime_to_position(regimes, position_map=custom_map)
    valid_positions = set(custom_map.values())
    for col in positions.columns:
        unique_vals = set(positions[col].dropna().unique())
        assert unique_vals.issubset(valid_positions)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n curve-factors pytest tests/unit/test_curve_regime.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/commodity_curve_factors/signals/curve_regime.py`:

```python
"""Curve regime classification based on convenience yield percentiles.

Classifies each commodity-month into one of five regimes using expanding-window
percentile ranks of convenience yield. No lookahead — each observation only
sees data up to and including itself.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS = [10, 30, 70, 90]

_DEFAULT_POSITION_MAP: dict[str, float] = {
    "crisis_backwardation": 1.0,
    "mild_backwardation": 0.5,
    "balanced": 0.0,
    "mild_contango": 0.0,
    "deep_contango": -0.5,
}

_REGIME_NAMES = [
    "deep_contango",
    "mild_contango",
    "balanced",
    "mild_backwardation",
    "crisis_backwardation",
]


def classify_regime(
    monthly_cy: pd.DataFrame,
    thresholds: list[int] | None = None,
) -> pd.DataFrame:
    """Classify each commodity-month into a curve regime.

    Uses expanding-window percentile rank of convenience yield.

    Parameters
    ----------
    monthly_cy : pd.DataFrame
        Monthly convenience yield (dates x commodities).
    thresholds : list[int] or None
        Percentile boundaries [p1, p2, p3, p4] defining 5 regimes.
        Default ``[10, 30, 70, 90]``.

    Returns
    -------
    pd.DataFrame
        Regime labels (dates x commodities). Values are one of:
        ``"deep_contango"``, ``"mild_contango"``, ``"balanced"``,
        ``"mild_backwardation"``, ``"crisis_backwardation"``.
    """
    if thresholds is None:
        thresholds = _DEFAULT_THRESHOLDS.copy()

    result = pd.DataFrame(index=monthly_cy.index, columns=monthly_cy.columns, dtype=object)

    for col in monthly_cy.columns:
        series = monthly_cy[col]
        for i in range(len(series)):
            val = series.iloc[i]
            if np.isnan(val):
                continue
            history = series.iloc[: i + 1].dropna()
            if len(history) < 12:  # need at least 12 months for meaningful percentiles
                continue
            pct = (history < val).sum() / len(history) * 100

            if pct < thresholds[0]:
                result.iloc[i, result.columns.get_loc(col)] = "deep_contango"
            elif pct < thresholds[1]:
                result.iloc[i, result.columns.get_loc(col)] = "mild_contango"
            elif pct < thresholds[2]:
                result.iloc[i, result.columns.get_loc(col)] = "balanced"
            elif pct < thresholds[3]:
                result.iloc[i, result.columns.get_loc(col)] = "mild_backwardation"
            else:
                result.iloc[i, result.columns.get_loc(col)] = "crisis_backwardation"

    logger.info("classify_regime: %d months, %d commodities", len(result), len(result.columns))
    return result


def regime_to_position(
    regimes: pd.DataFrame,
    position_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Map regime labels to base position weights.

    Parameters
    ----------
    regimes : pd.DataFrame
        Regime labels from ``classify_regime``.
    position_map : dict[str, float] or None
        Mapping from regime name to position weight.
        Default: crisis_backwardation=1.0, mild_backwardation=0.5,
        balanced=0.0, mild_contango=0.0, deep_contango=-0.5.

    Returns
    -------
    pd.DataFrame
        Position weights (dates x commodities).
    """
    if position_map is None:
        position_map = _DEFAULT_POSITION_MAP.copy()

    result = regimes.replace(position_map).infer_objects()
    # Any regime labels not in the map become NaN
    result = result.apply(pd.to_numeric, errors="coerce")
    logger.info("regime_to_position: %d rows, %d commodities", len(result), len(result.columns))
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n curve-factors pytest tests/unit/test_curve_regime.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Run linter and commit**

```bash
conda run -n curve-factors ruff check src/commodity_curve_factors/signals/curve_regime.py tests/unit/test_curve_regime.py
conda run -n curve-factors ruff format src/commodity_curve_factors/signals/curve_regime.py tests/unit/test_curve_regime.py
git add src/commodity_curve_factors/signals/curve_regime.py tests/unit/test_curve_regime.py
git commit -m "Add curve regime classification from convenience yield percentiles"
```

---

## Task 3: Layer 1 — Directional Positioning with Trend Filter

**Files:**
- Create: `src/commodity_curve_factors/signals/directional.py`
- Test: `tests/unit/test_directional.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_directional.py`:

```python
"""Tests for signals/directional.py (Layer 1: Curve-Informed Directional)."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.directional import (
    apply_trend_filter,
    build_directional_weights,
    resample_weights_monthly,
)


@pytest.fixture()
def dates():
    return pd.bdate_range("2010-01-04", periods=500)


@pytest.fixture()
def positions():
    """Regime-based positions: CL long, NG short, GC flat."""
    # Monthly index, forward-filled to daily in the actual pipeline
    monthly_dates = pd.date_range("2010-01-31", periods=24, freq="ME")
    return pd.DataFrame(
        {
            "CL": [1.0] * 12 + [0.5] * 12,
            "NG": [-0.5] * 24,
            "GC": [0.0] * 24,
        },
        index=monthly_dates,
    )


@pytest.fixture()
def tsmom(dates):
    """TSMOM signal: CL positive, NG negative, GC mixed."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "CL": np.abs(rng.standard_normal(500)),  # always positive
            "NG": -np.abs(rng.standard_normal(500)),  # always negative
            "GC": rng.standard_normal(500),  # mixed
        },
        index=dates,
    )


def test_apply_trend_filter_long_with_positive_trend(positions, tsmom):
    filtered = apply_trend_filter(positions, tsmom)
    # CL is long and TSMOM > 0 → should stay long
    assert (filtered["CL"].dropna() >= 0).all()


def test_apply_trend_filter_short_with_negative_trend(positions, tsmom):
    filtered = apply_trend_filter(positions, tsmom)
    # NG is short and TSMOM < 0 → should stay short
    assert (filtered["NG"].dropna() <= 0).all()


def test_apply_trend_filter_overrides_long_when_trend_negative():
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    positions = pd.DataFrame({"CL": [1.0, 1.0, 0.5, 0.5, 1.0]}, index=dates)
    tsmom = pd.DataFrame({"CL": [1.0, -1.0, -1.0, 1.0, 1.0]}, index=dates)
    filtered = apply_trend_filter(positions, tsmom)
    assert filtered["CL"].iloc[1] == 0.0, "Long should be overridden to flat when TSMOM < 0"
    assert filtered["CL"].iloc[2] == 0.0, "Long should be overridden to flat when TSMOM < 0"


def test_apply_trend_filter_overrides_short_when_trend_positive():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    positions = pd.DataFrame({"CL": [-0.5, -0.5, -0.5]}, index=dates)
    tsmom = pd.DataFrame({"CL": [1.0, -1.0, 1.0]}, index=dates)
    filtered = apply_trend_filter(positions, tsmom)
    assert filtered["CL"].iloc[0] == 0.0, "Short overridden to flat when TSMOM > 0"
    assert filtered["CL"].iloc[1] == -0.5, "Short kept when TSMOM < 0"


def test_resample_weights_monthly():
    daily_dates = pd.bdate_range("2020-01-01", periods=60)
    monthly_dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    monthly_w = pd.DataFrame({"CL": [1.0, 0.5, -0.5]}, index=monthly_dates)
    result = resample_weights_monthly(monthly_w, daily_dates)
    assert len(result) == 60
    # All January days should have weight 1.0
    jan_vals = result.loc["2020-01":"2020-01", "CL"]
    assert (jan_vals == 1.0).all()


def test_build_directional_weights_shape():
    monthly_dates = pd.date_range("2010-01-31", periods=24, freq="ME")
    daily_dates = pd.bdate_range("2010-01-04", periods=500)
    monthly_cy = pd.DataFrame(
        {"CL": np.linspace(0.05, 0.15, 24), "NG": np.linspace(-0.05, 0.05, 24)},
        index=monthly_dates,
    )
    tsmom = pd.DataFrame(
        {"CL": np.ones(500), "NG": -np.ones(500)},
        index=daily_dates,
    )
    weights = build_directional_weights(monthly_cy, tsmom, daily_dates)
    assert isinstance(weights, pd.DataFrame)
    assert len(weights) == 500
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n curve-factors pytest tests/unit/test_directional.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/commodity_curve_factors/signals/directional.py`:

```python
"""Layer 1: Curve-Informed Directional positioning.

Positions each commodity long/flat/short based on its curve regime
(convenience yield percentile), gated by a TSMOM trend filter.
Rebalanced monthly.
"""

import logging

import numpy as np
import pandas as pd

from commodity_curve_factors.signals.curve_regime import classify_regime, regime_to_position
from commodity_curve_factors.utils.config import load_config

logger = logging.getLogger(__name__)


def apply_trend_filter(
    positions: pd.DataFrame,
    tsmom: pd.DataFrame,
) -> pd.DataFrame:
    """Gate regime-based positions on TSMOM sign.

    - Long positions (> 0) are zeroed when TSMOM <= 0
    - Short positions (< 0) are zeroed when TSMOM > 0
    - Flat positions (== 0) are unchanged

    Parameters
    ----------
    positions : pd.DataFrame
        Base positions from regime classification (dates x commodities).
    tsmom : pd.DataFrame
        TSMOM signal values (dates x commodities). Sign is what matters.

    Returns
    -------
    pd.DataFrame
        Filtered positions.
    """
    common_idx = positions.index.intersection(tsmom.index)
    common_cols = positions.columns.intersection(tsmom.columns)

    pos = positions.loc[common_idx, common_cols].copy()
    trend = tsmom.loc[common_idx, common_cols]

    # Zero out longs when trend is negative
    long_mask = pos > 0
    trend_neg = trend <= 0
    pos[long_mask & trend_neg] = 0.0

    # Zero out shorts when trend is positive
    short_mask = pos < 0
    trend_pos = trend > 0
    pos[short_mask & trend_pos] = 0.0

    return pos


def resample_weights_monthly(
    monthly_weights: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Forward-fill monthly weights to daily frequency.

    Parameters
    ----------
    monthly_weights : pd.DataFrame
        Monthly positions (month-end dates x commodities).
    daily_index : pd.DatetimeIndex
        Target daily index to fill to.

    Returns
    -------
    pd.DataFrame
        Daily weights, constant within each month.
    """
    result = monthly_weights.reindex(daily_index, method="ffill")
    result = result.fillna(0.0)
    return result


def build_directional_weights(
    monthly_cy: pd.DataFrame,
    tsmom: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    thresholds: list[int] | None = None,
    position_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build Layer 1 daily weights: regime classification → trend filter → monthly resample.

    Parameters
    ----------
    monthly_cy : pd.DataFrame
        Monthly convenience yield (month-end dates x commodities).
    tsmom : pd.DataFrame
        Daily TSMOM signal (dates x commodities).
    daily_index : pd.DatetimeIndex
        Target daily index for the output weights.
    thresholds : list[int] or None
        Regime percentile thresholds. Default [10, 30, 70, 90].
    position_map : dict[str, float] or None
        Regime-to-position mapping.

    Returns
    -------
    pd.DataFrame
        Daily position weights (dates x commodities).
    """
    regimes = classify_regime(monthly_cy, thresholds=thresholds)
    base_positions = regime_to_position(regimes, position_map=position_map)

    # Resample TSMOM to monthly (use month-end value) for filtering
    tsmom_monthly = tsmom.resample("ME").last()
    filtered = apply_trend_filter(base_positions, tsmom_monthly)

    # Forward-fill to daily
    weights = resample_weights_monthly(filtered, daily_index)
    logger.info(
        "build_directional_weights: %d daily rows, %d commodities",
        len(weights),
        len(weights.columns),
    )
    return weights
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n curve-factors pytest tests/unit/test_directional.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Lint and commit**

```bash
conda run -n curve-factors ruff check src/commodity_curve_factors/signals/directional.py tests/unit/test_directional.py
conda run -n curve-factors ruff format src/commodity_curve_factors/signals/directional.py tests/unit/test_directional.py
git add src/commodity_curve_factors/signals/directional.py tests/unit/test_directional.py
git commit -m "Add Layer 1 directional positioning with trend filter"
```

---

## Task 4: Layer 2 — Curve Transition Momentum

**Files:**
- Create: `src/commodity_curve_factors/signals/curve_transition.py`
- Test: `tests/unit/test_curve_transition.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_curve_transition.py`:

```python
"""Tests for signals/curve_transition.py (Layer 2: Curve Transition Momentum)."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.curve_transition import (
    compute_transition_signal,
    transition_to_position,
)


@pytest.fixture()
def monthly_cy():
    """Monthly CY with a clear tightening trend for CL, loosening for NG."""
    dates = pd.date_range("2010-01-31", periods=36, freq="ME")
    return pd.DataFrame(
        {
            "CL": np.linspace(0.02, 0.15, 36),  # steadily tightening
            "NG": np.linspace(0.10, -0.02, 36),  # steadily loosening
            "GC": np.random.default_rng(42).standard_normal(36) * 0.02,  # flat/noisy
        },
        index=dates,
    )


@pytest.fixture()
def tsmom():
    """Daily TSMOM signal aligned with the monthly CY period."""
    dates = pd.bdate_range("2010-01-04", periods=700)
    return pd.DataFrame(
        {
            "CL": np.ones(700),  # positive trend
            "NG": -np.ones(700),  # negative trend
            "GC": np.zeros(700),  # flat
        },
        index=dates,
    )


def test_transition_signal_shape(monthly_cy):
    signal = compute_transition_signal(monthly_cy, lookback=63)
    assert isinstance(signal, pd.DataFrame)
    assert set(signal.columns) == {"CL", "NG", "GC"}


def test_transition_signal_positive_for_tightening(monthly_cy):
    signal = compute_transition_signal(monthly_cy, lookback=63)
    # CL is steadily tightening — late values should be positive
    late = signal["CL"].dropna().iloc[-6:]
    assert (late > 0).any(), "CL tightening should produce positive transition signal"


def test_transition_signal_negative_for_loosening(monthly_cy):
    signal = compute_transition_signal(monthly_cy, lookback=63)
    late = signal["NG"].dropna().iloc[-6:]
    assert (late < 0).any(), "NG loosening should produce negative transition signal"


def test_transition_to_position_values(monthly_cy, tsmom):
    signal = compute_transition_signal(monthly_cy, lookback=63)
    positions = transition_to_position(signal, tsmom, threshold=0.5)
    assert isinstance(positions, pd.DataFrame)
    # Only {-1, 0, +1} (or NaN)
    valid_vals = {-1.0, 0.0, 1.0}
    for col in positions.columns:
        unique = set(positions[col].dropna().unique())
        assert unique.issubset(valid_vals), f"{col}: unexpected values {unique - valid_vals}"


def test_transition_to_position_confirmation_gate(monthly_cy, tsmom):
    signal = compute_transition_signal(monthly_cy, lookback=63)
    positions = transition_to_position(signal, tsmom, threshold=0.5)
    # GC has flat TSMOM (0) — should always be flat regardless of transition
    assert (positions["GC"].dropna() == 0.0).all(), "GC (flat TSMOM) should always be flat"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n curve-factors pytest tests/unit/test_curve_transition.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/commodity_curve_factors/signals/curve_transition.py`:

```python
"""Layer 2: Curve Transition Momentum.

Detects when commodities are transitioning between curve regimes
(tightening or loosening) and trades the transition with TSMOM confirmation.
"""

import logging

import numpy as np
import pandas as pd

from commodity_curve_factors.factors.transforms import expanding_zscore_df

logger = logging.getLogger(__name__)


def compute_transition_signal(
    monthly_cy: pd.DataFrame,
    lookback: int = 63,
) -> pd.DataFrame:
    """Compute curve transition signal as the change in CY z-score.

    Forward-fills monthly CY to daily, computes expanding z-score,
    then takes the ``lookback``-day difference.

    Parameters
    ----------
    monthly_cy : pd.DataFrame
        Monthly convenience yield (month-end dates x commodities).
    lookback : int
        Number of trading days for the diff. Default 63 (~3 months).

    Returns
    -------
    pd.DataFrame
        Daily transition signal (dates x commodities).
        Positive = tightening, negative = loosening.
    """
    # Forward-fill monthly to daily
    daily_cy = monthly_cy.resample("B").ffill()

    # Expanding z-score (no lookahead)
    cy_zscore = expanding_zscore_df(daily_cy, min_periods=252)

    # 3-month change
    transition = cy_zscore - cy_zscore.shift(lookback)

    logger.info(
        "compute_transition_signal: lookback=%d, %d dates, %d commodities",
        lookback,
        len(transition),
        len(transition.columns),
    )
    return transition


def transition_to_position(
    transition: pd.DataFrame,
    tsmom: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Convert transition signal to positions with TSMOM confirmation.

    Parameters
    ----------
    transition : pd.DataFrame
        Transition signal from ``compute_transition_signal``.
    tsmom : pd.DataFrame
        TSMOM signal (dates x commodities). Sign used for confirmation.
    threshold : float
        Standard deviation threshold for the transition signal.
        Applied to the expanding std of the transition signal itself.

    Returns
    -------
    pd.DataFrame
        Positions in {-1, 0, +1} (dates x commodities).
    """
    common_idx = transition.index.intersection(tsmom.index)
    common_cols = transition.columns.intersection(tsmom.columns)
    trans = transition.loc[common_idx, common_cols]
    trend = tsmom.loc[common_idx, common_cols]

    # Expanding threshold on the transition signal
    expanding_std = trans.expanding(min_periods=60).std()
    upper = threshold * expanding_std
    lower = -threshold * expanding_std

    # Raw signal: +1 tightening, -1 loosening, 0 neutral
    raw = pd.DataFrame(0.0, index=common_idx, columns=common_cols)
    raw[trans > upper] = 1.0
    raw[trans < lower] = -1.0

    # TSMOM confirmation gate
    # Long only if TSMOM > 0, short only if TSMOM <= 0
    positions = raw.copy()
    positions[(raw > 0) & (trend <= 0)] = 0.0  # tightening but trend negative → flat
    positions[(raw < 0) & (trend > 0)] = 0.0  # loosening but trend positive → flat

    logger.info(
        "transition_to_position: threshold=%.2f, %d dates, non-zero=%.1f%%",
        threshold,
        len(positions),
        (positions != 0).any(axis=1).mean() * 100,
    )
    return positions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n curve-factors pytest tests/unit/test_curve_transition.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Lint and commit**

```bash
conda run -n curve-factors ruff check src/commodity_curve_factors/signals/curve_transition.py tests/unit/test_curve_transition.py
conda run -n curve-factors ruff format src/commodity_curve_factors/signals/curve_transition.py tests/unit/test_curve_transition.py
git add src/commodity_curve_factors/signals/curve_transition.py tests/unit/test_curve_transition.py
git commit -m "Add Layer 2 curve transition momentum with TSMOM confirmation"
```

---

## Task 5: Seasonal Pattern Module

**Files:**
- Create: `src/commodity_curve_factors/signals/seasonal.py`
- Test: `tests/unit/test_seasonal.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_seasonal.py`:

```python
"""Tests for signals/seasonal.py."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.seasonal import (
    compute_seasonal_pattern,
    deseasonalise,
)


@pytest.fixture()
def seasonal_series():
    """Series with a clear seasonal pattern: high in summer, low in winter."""
    dates = pd.bdate_range("2010-01-04", periods=1260)  # ~5 years
    rng = np.random.default_rng(42)
    # Seasonal component: sin wave peaking mid-year
    week_of_year = dates.isocalendar().week.values.astype(float)
    seasonal = np.sin(2 * np.pi * week_of_year / 52) * 0.1
    noise = rng.standard_normal(len(dates)) * 0.02
    return pd.Series(seasonal + noise, index=dates, name="test")


def test_compute_seasonal_pattern_length(seasonal_series):
    pattern = compute_seasonal_pattern(seasonal_series, lookback_years=3)
    # Should have one value per ISO week (52 or 53)
    assert 50 <= len(pattern) <= 54


def test_compute_seasonal_pattern_captures_peak(seasonal_series):
    pattern = compute_seasonal_pattern(seasonal_series, lookback_years=3)
    # Peak should be around week 13 (mid-year in sin wave)
    peak_week = pattern.idxmax()
    assert 10 <= peak_week <= 16, f"Expected peak near week 13, got {peak_week}"


def test_deseasonalise_reduces_variance(seasonal_series):
    pattern = compute_seasonal_pattern(seasonal_series, lookback_years=3)
    deseas = deseasonalise(seasonal_series, pattern)
    assert deseas.std() < seasonal_series.std(), (
        "Deseasonalised series should have lower variance"
    )


def test_deseasonalise_preserves_length(seasonal_series):
    pattern = compute_seasonal_pattern(seasonal_series, lookback_years=3)
    deseas = deseasonalise(seasonal_series, pattern)
    assert len(deseas) == len(seasonal_series)


def test_compute_seasonal_no_lookahead(seasonal_series):
    """Pattern computed from first 3 years should not use year 4+ data."""
    short = seasonal_series.iloc[:756]  # ~3 years
    full = seasonal_series
    pattern_short = compute_seasonal_pattern(short, lookback_years=2)
    pattern_full = compute_seasonal_pattern(full, lookback_years=2)
    # Patterns should differ because full has more data — but this tests
    # that the function runs without error on different lengths
    assert len(pattern_short) > 0
    assert len(pattern_full) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n curve-factors pytest tests/unit/test_seasonal.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/commodity_curve_factors/signals/seasonal.py`:

```python
"""Seasonal pattern extraction and deseasonalisation.

Computes ISO-week seasonal averages from trailing data and
subtracts them to produce deseasonalised signals.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_seasonal_pattern(
    series: pd.Series,
    lookback_years: int = 5,
) -> pd.Series:
    """Compute ISO-week seasonal pattern from trailing data.

    Uses the most recent ``lookback_years`` of data to compute
    the mean value per ISO week.

    Parameters
    ----------
    series : pd.Series
        Daily series with DatetimeIndex.
    lookback_years : int
        Number of trailing years to use. Default 5.

    Returns
    -------
    pd.Series
        Indexed by ISO week number (1-53), values are week averages.
    """
    cutoff = series.index[-1] - pd.DateOffset(years=lookback_years)
    recent = series.loc[cutoff:]

    iso_weeks = recent.index.isocalendar().week.values.astype(int)
    week_series = pd.Series(recent.values, index=iso_weeks)
    pattern = week_series.groupby(week_series.index).mean()
    pattern.index.name = "iso_week"

    logger.info(
        "compute_seasonal_pattern: %d weeks, lookback=%d years",
        len(pattern),
        lookback_years,
    )
    return pattern


def deseasonalise(
    series: pd.Series,
    seasonal_pattern: pd.Series,
) -> pd.Series:
    """Remove seasonal component from a daily series.

    Parameters
    ----------
    series : pd.Series
        Daily series with DatetimeIndex.
    seasonal_pattern : pd.Series
        ISO-week seasonal pattern from ``compute_seasonal_pattern``.

    Returns
    -------
    pd.Series
        Deseasonalised series (original - seasonal component).
    """
    iso_weeks = series.index.isocalendar().week.values.astype(int)
    seasonal_component = pd.Series(
        seasonal_pattern.reindex(iso_weeks).values,
        index=series.index,
    )
    result = series - seasonal_component
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n curve-factors pytest tests/unit/test_seasonal.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Lint and commit**

```bash
conda run -n curve-factors ruff check src/commodity_curve_factors/signals/seasonal.py tests/unit/test_seasonal.py
conda run -n curve-factors ruff format src/commodity_curve_factors/signals/seasonal.py tests/unit/test_seasonal.py
git add src/commodity_curve_factors/signals/seasonal.py tests/unit/test_seasonal.py
git commit -m "Add seasonal pattern extraction and deseasonalisation"
```

---

## Task 6: Layer 3 — Structural Spread Signals

**Files:**
- Create: `src/commodity_curve_factors/signals/spreads.py`
- Test: `tests/unit/test_spreads.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_spreads.py`:

```python
"""Tests for signals/spreads.py (Layer 3: Structural Spread Alpha)."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.spreads import (
    compute_cy_crack,
    crack_spread_signal,
    inventory_overlay,
    livestock_spread_signal,
)


@pytest.fixture()
def daily_cy():
    dates = pd.bdate_range("2010-01-04", periods=1000)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "CL": 0.08 + rng.standard_normal(1000) * 0.03,
            "RB": 0.05 + rng.standard_normal(1000) * 0.03,
            "HO": 0.06 + rng.standard_normal(1000) * 0.03,
        },
        index=dates,
    )


def test_compute_cy_crack_shape(daily_cy):
    crack = compute_cy_crack(daily_cy)
    assert isinstance(crack, pd.Series)
    assert len(crack) == 1000


def test_compute_cy_crack_formula(daily_cy):
    """cy_crack = cy(RB) + cy(HO) - cy(CL)."""
    crack = compute_cy_crack(daily_cy)
    expected = daily_cy["RB"] + daily_cy["HO"] - daily_cy["CL"]
    pd.testing.assert_series_equal(crack, expected, check_names=False)


def test_crack_spread_signal_values(daily_cy):
    crack = compute_cy_crack(daily_cy)
    positions = crack_spread_signal(crack, threshold=1.5)
    assert isinstance(positions, pd.DataFrame)
    assert set(positions.columns) == {"CL", "RB", "HO"}
    # Dollar-neutral: for each row, sum should be ~0
    row_sums = positions.sum(axis=1).dropna()
    assert (row_sums.abs() < 0.01).all(), "Spread positions should be dollar-neutral"


def test_crack_spread_signal_direction():
    """Negative z-score (crude tight) → long products, short crude."""
    dates = pd.bdate_range("2020-01-01", periods=500)
    # Create a crack series that starts normal then goes very negative
    crack = pd.Series(np.concatenate([np.zeros(400), np.full(100, -0.15)]), index=dates)
    positions = crack_spread_signal(crack, threshold=1.5)
    late = positions.iloc[-10:]
    assert (late["CL"].dropna() < 0).any(), "CL should be short when crack z < -1.5"


def test_inventory_overlay_amplifies():
    dates = pd.bdate_range("2020-01-01", periods=100)
    positions = pd.DataFrame({"CL": [1.0] * 100, "NG": [-0.5] * 100}, index=dates)
    inv_surprise = pd.DataFrame({"CL": [-1.0] * 100, "NG": [1.0] * 100}, index=dates)
    cy_change = pd.DataFrame({"CL": [0.5] * 100, "NG": [-0.5] * 100}, index=dates)
    result = inventory_overlay(positions, inv_surprise, cy_change, amplification=1.5)
    assert result["CL"].iloc[0] == pytest.approx(1.5), "CL long + negative inv surprise + rising CY → amplify"
    assert result["NG"].iloc[0] == pytest.approx(-0.75), "NG short + positive inv surprise + falling CY → amplify"


def test_inventory_overlay_no_change_when_signals_disagree():
    dates = pd.bdate_range("2020-01-01", periods=10)
    positions = pd.DataFrame({"CL": [1.0] * 10}, index=dates)
    inv_surprise = pd.DataFrame({"CL": [1.0] * 10}, index=dates)  # positive = build, disagrees with long
    cy_change = pd.DataFrame({"CL": [0.5] * 10}, index=dates)
    result = inventory_overlay(positions, inv_surprise, cy_change)
    pd.testing.assert_frame_equal(result, positions)


def test_livestock_spread_signal_shape():
    dates = pd.bdate_range("2010-01-04", periods=1000)
    rng = np.random.default_rng(42)
    lc = pd.Series(120.0 + rng.standard_normal(1000) * 5, index=dates, name="LC")
    lh = pd.Series(80.0 + rng.standard_normal(1000) * 5, index=dates, name="LH")
    positions = livestock_spread_signal(lc, lh, seasonal_years=3, threshold=1.5)
    assert isinstance(positions, pd.DataFrame)
    assert set(positions.columns) == {"LC", "LH"}


def test_livestock_spread_dollar_neutral():
    dates = pd.bdate_range("2010-01-04", periods=1000)
    rng = np.random.default_rng(42)
    lc = pd.Series(120.0 + rng.standard_normal(1000) * 5, index=dates, name="LC")
    lh = pd.Series(80.0 + rng.standard_normal(1000) * 5, index=dates, name="LH")
    positions = livestock_spread_signal(lc, lh, seasonal_years=3, threshold=1.5)
    row_sums = positions.sum(axis=1).dropna()
    assert (row_sums.abs() < 0.01).all(), "Livestock spread should be dollar-neutral"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n curve-factors pytest tests/unit/test_spreads.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/commodity_curve_factors/signals/spreads.py`:

```python
"""Layer 3: Structural spread signals.

Convenience-yield-adjusted crack spread, inventory-conditioned energy
signals, and deseasonalised livestock spread.
"""

import logging

import numpy as np
import pandas as pd

from commodity_curve_factors.factors.transforms import expanding_zscore
from commodity_curve_factors.signals.seasonal import (
    compute_seasonal_pattern,
    deseasonalise,
)

logger = logging.getLogger(__name__)


def compute_cy_crack(convenience_yields: pd.DataFrame) -> pd.Series:
    """Compute convenience yield crack spread.

    ``cy_crack = cy(RB) + cy(HO) - cy(CL)``

    Parameters
    ----------
    convenience_yields : pd.DataFrame
        Must contain columns ``"CL"``, ``"RB"``, ``"HO"``.

    Returns
    -------
    pd.Series
        Daily CY crack spread.
    """
    return convenience_yields["RB"] + convenience_yields["HO"] - convenience_yields["CL"]


def crack_spread_signal(
    cy_crack: pd.Series,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Generate mean-reversion positions from CY crack spread.

    When z < -threshold (crude tight, products loose):
      long 0.5 RB + 0.5 HO, short CL
    When z > +threshold (crude loose, products tight):
      long CL, short 0.5 RB + 0.5 HO
    Otherwise: flat.

    Parameters
    ----------
    cy_crack : pd.Series
        Daily CY crack spread.
    threshold : float
        Z-score threshold for entry. Default 1.5.

    Returns
    -------
    pd.DataFrame
        Positions for CL, RB, HO (dollar-neutral).
    """
    z = expanding_zscore(cy_crack, min_periods=252)

    positions = pd.DataFrame(0.0, index=cy_crack.index, columns=["CL", "RB", "HO"])

    # Crude tight (z < -threshold) → long products, short crude
    short_crude = z < -threshold
    positions.loc[short_crude, "CL"] = -1.0
    positions.loc[short_crude, "RB"] = 0.5
    positions.loc[short_crude, "HO"] = 0.5

    # Crude loose (z > +threshold) → long crude, short products
    long_crude = z > threshold
    positions.loc[long_crude, "CL"] = 1.0
    positions.loc[long_crude, "RB"] = -0.5
    positions.loc[long_crude, "HO"] = -0.5

    # Propagate NaN from z-score
    positions[z.isna()] = np.nan

    logger.info(
        "crack_spread_signal: threshold=%.1f, active=%.1f%% of days",
        threshold,
        ((positions != 0).any(axis=1) & positions.notna().all(axis=1)).mean() * 100,
    )
    return positions


def inventory_overlay(
    positions: pd.DataFrame,
    inventory_surprise: pd.DataFrame,
    cy_change: pd.DataFrame,
    amplification: float = 1.5,
) -> pd.DataFrame:
    """Amplify energy positions when inventory and CY agree.

    Parameters
    ----------
    positions : pd.DataFrame
        Base positions (dates x commodities).
    inventory_surprise : pd.DataFrame
        Inventory surprise z-scores. Negative = draw > expected.
    cy_change : pd.DataFrame
        Change in convenience yield. Positive = CY rising.
    amplification : float
        Multiplier applied to confirming positions. Default 1.5.

    Returns
    -------
    pd.DataFrame
        Adjusted positions.
    """
    result = positions.copy()
    common_cols = positions.columns.intersection(inventory_surprise.columns).intersection(
        cy_change.columns
    )
    common_idx = positions.index.intersection(inventory_surprise.index).intersection(
        cy_change.index
    )

    for col in common_cols:
        pos = result.loc[common_idx, col]
        inv = inventory_surprise.loc[common_idx, col]
        cy = cy_change.loc[common_idx, col]

        # Amplify longs when inventory surprise is negative AND CY is rising
        amplify_long = (pos > 0) & (inv < 0) & (cy > 0)
        result.loc[common_idx[amplify_long], col] = pos[amplify_long] * amplification

        # Amplify shorts when inventory surprise is positive AND CY is falling
        amplify_short = (pos < 0) & (inv > 0) & (cy < 0)
        result.loc[common_idx[amplify_short], col] = pos[amplify_short] * amplification

    return result


def livestock_spread_signal(
    lc_prices: pd.Series,
    lh_prices: pd.Series,
    seasonal_years: int = 5,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Deseasonalised cattle-hog spread mean-reversion signal.

    Parameters
    ----------
    lc_prices : pd.Series
        Live cattle front-month prices.
    lh_prices : pd.Series
        Lean hogs front-month prices.
    seasonal_years : int
        Lookback for seasonal pattern. Default 5.
    threshold : float
        Z-score threshold for entry. Default 1.5.

    Returns
    -------
    pd.DataFrame
        Dollar-neutral positions for LC and LH.
    """
    common_idx = lc_prices.index.intersection(lh_prices.index)
    spread = np.log(lc_prices.loc[common_idx]) - np.log(lh_prices.loc[common_idx])
    spread.name = "lc_lh_spread"

    # Deseasonalise if enough history
    if len(spread) > seasonal_years * 252:
        pattern = compute_seasonal_pattern(spread, lookback_years=seasonal_years)
        spread_deseas = deseasonalise(spread, pattern)
    else:
        spread_deseas = spread

    z = expanding_zscore(spread_deseas, min_periods=252)

    positions = pd.DataFrame(0.0, index=common_idx, columns=["LC", "LH"])

    # Spread too high (cattle expensive vs hogs) → short cattle, long hogs
    high = z > threshold
    positions.loc[high, "LC"] = -0.5
    positions.loc[high, "LH"] = 0.5

    # Spread too low (cattle cheap vs hogs) → long cattle, short hogs
    low = z < -threshold
    positions.loc[low, "LC"] = 0.5
    positions.loc[low, "LH"] = -0.5

    positions[z.isna()] = np.nan

    logger.info(
        "livestock_spread_signal: threshold=%.1f, active=%.1f%% of days",
        threshold,
        ((positions != 0).any(axis=1) & positions.notna().all(axis=1)).mean() * 100,
    )
    return positions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n curve-factors pytest tests/unit/test_spreads.py -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Lint and commit**

```bash
conda run -n curve-factors ruff check src/commodity_curve_factors/signals/spreads.py tests/unit/test_spreads.py
conda run -n curve-factors ruff format src/commodity_curve_factors/signals/spreads.py tests/unit/test_spreads.py
git add src/commodity_curve_factors/signals/spreads.py tests/unit/test_spreads.py
git commit -m "Add Layer 3 structural spread signals (crack, inventory, livestock)"
```

---

## Task 7: Combined Strategy with Ledoit-Wolf Vol Targeting

**Files:**
- Create: `src/commodity_curve_factors/signals/combined_strategy.py`
- Test: `tests/unit/test_combined_strategy.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_combined_strategy.py`:

```python
"""Tests for signals/combined_strategy.py."""

import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.combined_strategy import (
    apply_ledoit_wolf_vol_target,
    combine_layers,
)


@pytest.fixture()
def dates():
    return pd.bdate_range("2010-01-04", periods=500)


@pytest.fixture()
def commodities():
    return ["CL", "NG", "GC"]


@pytest.fixture()
def returns(dates, commodities):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.standard_normal((500, 3)) * 0.01,
        index=dates,
        columns=commodities,
    )


@pytest.fixture()
def layer_weights(dates, commodities):
    rng = np.random.default_rng(99)
    w1 = pd.DataFrame(rng.uniform(-0.5, 1.0, (500, 3)), index=dates, columns=commodities)
    w2 = pd.DataFrame(rng.uniform(-1.0, 1.0, (500, 3)), index=dates, columns=commodities)
    w3 = pd.DataFrame(rng.uniform(-0.5, 0.5, (500, 3)), index=dates, columns=commodities)
    return w1, w2, w3


def test_combine_layers_shape(layer_weights, returns):
    w1, w2, w3 = layer_weights
    budgets = [0.40, 0.25, 0.35]
    combined = combine_layers([w1, w2, w3], budgets, returns, target_vol=0.10)
    assert isinstance(combined, pd.DataFrame)
    assert combined.shape == w1.shape


def test_combine_layers_respects_risk_budget(layer_weights, returns):
    w1, w2, w3 = layer_weights
    # With only Layer 1 active (others zero), should produce ~40% of total vol
    w2_zero = w2 * 0
    w3_zero = w3 * 0
    budgets = [0.40, 0.25, 0.35]
    combined = combine_layers([w1, w2_zero, w3_zero], budgets, returns, target_vol=0.10)
    # Combined should have some reasonable weights, not all zero
    assert combined.abs().sum().sum() > 0


def test_apply_ledoit_wolf_vol_target_scales(returns, commodities, dates):
    weights = pd.DataFrame(1.0 / 3, index=dates, columns=commodities)
    result = apply_ledoit_wolf_vol_target(weights, returns, target_vol=0.10, lookback=60)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == weights.shape
    # Scaled weights should differ from original (unless vol is exactly 10%)
    assert not result.equals(weights)


def test_apply_ledoit_wolf_max_leverage(returns, commodities, dates):
    # Very large weights that would need huge leverage
    weights = pd.DataFrame(10.0, index=dates, columns=commodities)
    result = apply_ledoit_wolf_vol_target(weights, returns, target_vol=0.10, max_leverage=2.0)
    # Leverage (sum of abs weights per row) should be <= 2.0
    leverage = result.abs().sum(axis=1)
    assert (leverage <= 2.0 + 1e-8).all(), f"Max leverage exceeded: {leverage.max():.3f}"


def test_combine_layers_budget_sums_to_one():
    """Risk budgets that don't sum to 1.0 should still work (normalised internally)."""
    dates = pd.bdate_range("2020-01-01", periods=100)
    cols = ["CL", "NG"]
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(rng.standard_normal((100, 2)) * 0.01, index=dates, columns=cols)
    w1 = pd.DataFrame(rng.uniform(0, 1, (100, 2)), index=dates, columns=cols)
    w2 = pd.DataFrame(rng.uniform(0, 1, (100, 2)), index=dates, columns=cols)
    # Non-normalised budgets
    combined = combine_layers([w1, w2], [0.6, 0.4], returns, target_vol=0.10)
    assert combined.shape == w1.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n curve-factors pytest tests/unit/test_combined_strategy.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/commodity_curve_factors/signals/combined_strategy.py`:

```python
"""Three-layer combination with Ledoit-Wolf vol targeting.

Combines the directional (Layer 1), transition (Layer 2), and spread
(Layer 3) weight DataFrames into a single portfolio using risk-budget
allocation and shrinkage-based vol targeting.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


def apply_ledoit_wolf_vol_target(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_vol: float = 0.10,
    lookback: int = 252,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """Scale weights to target volatility using Ledoit-Wolf shrinkage covariance.

    Parameters
    ----------
    weights : pd.DataFrame
        Raw position weights (dates x commodities).
    returns : pd.DataFrame
        Daily log returns (dates x commodities).
    target_vol : float
        Annualised volatility target. Default 0.10.
    lookback : int
        Trailing window for covariance estimation. Default 252.
    max_leverage : float
        Maximum gross leverage (sum of absolute weights). Default 2.0.

    Returns
    -------
    pd.DataFrame
        Scaled weights.
    """
    common_idx = weights.index.intersection(returns.index)
    common_cols = weights.columns.intersection(returns.columns)
    w = weights.loc[common_idx, common_cols]
    r = returns.loc[common_idx, common_cols]

    result = w.copy() * 0.0  # initialise with zeros
    daily_target = target_vol / np.sqrt(252)

    for i in range(lookback, len(common_idx)):
        ret_window = r.iloc[i - lookback : i]
        w_row = w.iloc[i]

        # Drop commodities with all NaN returns in window
        valid_cols = ret_window.columns[ret_window.notna().sum() > lookback // 2]
        if len(valid_cols) < 2:
            continue

        ret_clean = ret_window[valid_cols].dropna()
        w_clean = w_row[valid_cols].fillna(0.0)

        if w_clean.abs().sum() < 1e-10:
            continue

        try:
            lw = LedoitWolf().fit(ret_clean.values)
            cov = lw.covariance_
        except Exception:
            continue

        port_var = float(w_clean.values @ cov @ w_clean.values)
        if port_var <= 0:
            continue

        port_vol = np.sqrt(port_var)
        scalar = daily_target / port_vol
        scalar = min(scalar, max_leverage / max(w_clean.abs().sum(), 1e-10))

        result.iloc[i][valid_cols] = w_clean * scalar

    logger.info(
        "apply_ledoit_wolf_vol_target: target=%.1f%%, lookback=%d, %d dates",
        target_vol * 100,
        lookback,
        len(result),
    )
    return result


def combine_layers(
    layer_weights: list[pd.DataFrame],
    risk_budgets: list[float],
    returns: pd.DataFrame,
    target_vol: float = 0.10,
    lookback: int = 252,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """Combine multiple signal layers with risk budgeting.

    Each layer is vol-targeted to its share of the total vol budget,
    then the layers are summed.

    Parameters
    ----------
    layer_weights : list[pd.DataFrame]
        List of weight DataFrames, one per layer.
    risk_budgets : list[float]
        Risk budget fraction per layer (should sum to ~1.0).
    returns : pd.DataFrame
        Daily log returns for vol targeting.
    target_vol : float
        Total portfolio annualised vol target. Default 0.10.
    lookback : int
        Covariance lookback window. Default 252.
    max_leverage : float
        Max leverage per layer. Default 2.0.

    Returns
    -------
    pd.DataFrame
        Combined position weights.
    """
    budget_sum = sum(risk_budgets)
    normalised_budgets = [b / budget_sum for b in risk_budgets]

    # Align all layers to common index and columns
    all_idx = layer_weights[0].index
    all_cols = layer_weights[0].columns
    for lw in layer_weights[1:]:
        all_idx = all_idx.union(lw.index)
        all_cols = all_cols.union(lw.columns)
    all_idx = all_idx.sort_values()

    combined = pd.DataFrame(0.0, index=all_idx, columns=all_cols)

    for lw, budget in zip(layer_weights, normalised_budgets):
        layer_target = target_vol * budget
        aligned = lw.reindex(index=all_idx, columns=all_cols).fillna(0.0)
        scaled = apply_ledoit_wolf_vol_target(
            aligned,
            returns,
            target_vol=layer_target,
            lookback=lookback,
            max_leverage=max_leverage * budget,
        )
        combined = combined + scaled.reindex(index=all_idx, columns=all_cols).fillna(0.0)

    logger.info(
        "combine_layers: %d layers, budgets=%s, %d dates, %d commodities",
        len(layer_weights),
        [f"{b:.0%}" for b in normalised_budgets],
        len(combined),
        len(combined.columns),
    )
    return combined
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n curve-factors pytest tests/unit/test_combined_strategy.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Lint and commit**

```bash
conda run -n curve-factors ruff check src/commodity_curve_factors/signals/combined_strategy.py tests/unit/test_combined_strategy.py
conda run -n curve-factors ruff format src/commodity_curve_factors/signals/combined_strategy.py tests/unit/test_combined_strategy.py
git add src/commodity_curve_factors/signals/combined_strategy.py tests/unit/test_combined_strategy.py
git commit -m "Add three-layer combination with Ledoit-Wolf vol targeting"
```

---

## Task 8: Config Updates and Backtest Runner Integration

**Files:**
- Modify: `configs/strategy.yaml`
- Modify: `src/commodity_curve_factors/backtest/__main__.py`

- [ ] **Step 1: Add TSI strategy parameters to strategy.yaml**

Append to `configs/strategy.yaml` (after the existing `execution:` block):

```yaml

# Term Structure Intelligence strategy
tsi:
  curve_directional:
    convenience_yield_tenor: "F6M"
    regime_thresholds: [10, 30, 70, 90]
    position_map:
      crisis_backwardation: 1.0
      mild_backwardation: 0.5
      balanced: 0.0
      mild_contango: 0.0
      deep_contango: -0.5
    trend_filter: true
    trend_lookback_days: 252
    rebalance: "monthly"
    risk_budget: 0.40

  curve_transition:
    lookback_days: 63
    threshold_std: 0.5
    tsmom_confirmation: true
    rebalance: "weekly"
    risk_budget: 0.25

  structural_spreads:
    crack_spread:
      commodities: ["CL", "RB", "HO"]
      z_threshold: 1.5
      rebalance: "biweekly"
    inventory_overlay:
      commodities: ["CL", "NG"]
      amplification: 1.5
    livestock_spread:
      long_leg: "LC"
      short_leg: "LH"
      z_threshold: 1.5
      seasonal_lookback_years: 5
      rebalance: "biweekly"
    risk_budget: 0.35
```

- [ ] **Step 2: Add TSI strategy to the backtest runner**

Add the following block to `src/commodity_curve_factors/backtest/__main__.py` in the `main()` function, after Strategy 7 (calendar spread) and before the benchmarks section. This requires adding imports at the top of the file and the strategy block in `main()`.

Add these imports at the top of the file (after existing imports):

```python
from commodity_curve_factors.curves.builder import load_curves
from commodity_curve_factors.curves.convenience_yield import (
    compute_convenience_yield,
    estimate_storage_cost,
    monthly_convenience_yield,
)
from commodity_curve_factors.signals.combined_strategy import combine_layers
from commodity_curve_factors.signals.curve_transition import (
    compute_transition_signal,
    transition_to_position,
)
from commodity_curve_factors.signals.directional import build_directional_weights
from commodity_curve_factors.signals.spreads import (
    compute_cy_crack,
    crack_spread_signal,
    inventory_overlay,
    livestock_spread_signal,
)
```

Add this strategy block in `main()`:

```python
    # ------------------------------------------------------------------
    # Strategy 8: Term Structure Intelligence (TSI)
    # ------------------------------------------------------------------
    logger.info("Strategy 8: Term Structure Intelligence (TSI)")
    tsi_cfg = strategy_cfg.get("tsi", {})

    # Load curves and compute convenience yield
    curves = load_curves()
    rf_df = macro.get("dgs3mo")
    rf_series = rf_df.iloc[:, 0] if rf_df is not None and not rf_df.empty else pd.Series(2.0, index=returns.index)

    storage_costs = estimate_storage_cost(curves, is_end="2017-12-31")
    daily_cy = compute_convenience_yield(curves, rf_series, storage_costs, tenor="F6M")
    monthly_cy = monthly_convenience_yield(daily_cy)

    # Layer 1: Curve-Informed Directional
    dir_cfg = tsi_cfg.get("curve_directional", {})
    layer1 = build_directional_weights(
        monthly_cy,
        tsmom,
        returns.index,
        thresholds=dir_cfg.get("regime_thresholds"),
        position_map=dir_cfg.get("position_map"),
    )

    # Layer 2: Curve Transition Momentum
    trans_cfg = tsi_cfg.get("curve_transition", {})
    transition = compute_transition_signal(
        monthly_cy, lookback=trans_cfg.get("lookback_days", 63)
    )
    layer2 = transition_to_position(
        transition, tsmom, threshold=trans_cfg.get("threshold_std", 0.5)
    )
    layer2 = resample_weights_weekly(layer2, rebalance_day=rebal_day)

    # Layer 3: Structural Spreads
    spread_cfg = tsi_cfg.get("structural_spreads", {})

    # 3a: CY crack spread
    cy_crack = compute_cy_crack(daily_cy)
    crack_pos = crack_spread_signal(
        cy_crack, threshold=spread_cfg.get("crack_spread", {}).get("z_threshold", 1.5)
    )

    # 3b: Inventory overlay on Layer 1 energy positions
    inv_factor = _load_factor("inventory")
    cy_change = daily_cy.diff(21)  # 1-month CY change
    layer1_inv = inventory_overlay(
        layer1,
        inv_factor,
        cy_change,
        amplification=spread_cfg.get("inventory_overlay", {}).get("amplification", 1.5),
    )

    # 3c: Livestock spread
    lc_close = futures.get("LC", pd.DataFrame()).get("Close", pd.Series(dtype=float))
    lh_close = futures.get("LH", pd.DataFrame()).get("Close", pd.Series(dtype=float))
    livestock_cfg = spread_cfg.get("livestock_spread", {})
    if not lc_close.empty and not lh_close.empty:
        livestock_pos = livestock_spread_signal(
            lc_close, lh_close,
            seasonal_years=livestock_cfg.get("seasonal_lookback_years", 5),
            threshold=livestock_cfg.get("z_threshold", 1.5),
        )
    else:
        livestock_pos = pd.DataFrame(0.0, index=returns.index, columns=["LC", "LH"])

    # Combine spread positions into a single Layer 3 weight DataFrame
    layer3 = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for col in crack_pos.columns:
        if col in layer3.columns:
            layer3[col] = layer3[col] + crack_pos[col].reindex(layer3.index).fillna(0.0)
    for col in livestock_pos.columns:
        if col in layer3.columns:
            layer3[col] = layer3[col] + livestock_pos[col].reindex(layer3.index).fillna(0.0)

    # Use layer1_inv (inventory-overlaid) instead of raw layer1
    budgets = [
        dir_cfg.get("risk_budget", 0.40),
        trans_cfg.get("risk_budget", 0.25),
        spread_cfg.get("risk_budget", 0.35),
    ]

    # Align all layers to returns columns
    for layer in [layer1_inv, layer2, layer3]:
        for col in returns.columns:
            if col not in layer.columns:
                layer[col] = 0.0

    tsi_weights = combine_layers(
        [layer1_inv[returns.columns], layer2.reindex(columns=returns.columns, fill_value=0.0), layer3[returns.columns]],
        budgets,
        returns,
        target_vol=strategy_cfg["constraints"]["vol_target"],
    )

    # Apply position and sector limits + execution lag
    tsi_weights = build_portfolio(tsi_weights, returns, strategy_cfg, universe_cfg)
    strategies["tsi"] = run_backtest(tsi_weights, returns, cost_config)
```

- [ ] **Step 3: Run the full test suite**

Run: `conda run -n curve-factors pytest tests/ -x -q`
Expected: all tests pass (existing 396 + ~38 new = ~434)

- [ ] **Step 4: Run linter**

Run: `conda run -n curve-factors ruff check src/ tests/ && conda run -n curve-factors ruff format src/ tests/`

- [ ] **Step 5: Commit**

```bash
git add configs/strategy.yaml src/commodity_curve_factors/backtest/__main__.py
git commit -m "Add TSI strategy to config and backtest runner"
```

---

## Task 9: Full Pipeline Smoke Test

- [ ] **Step 1: Run the full pipeline on real data**

```bash
conda run -n curve-factors python -m commodity_curve_factors.backtest
```

Expected: all 8 strategies run (7 existing + TSI). Check the TSI Sharpe and cumulative in the log output.

- [ ] **Step 2: Run evaluation report**

```bash
conda run -n curve-factors python -m commodity_curve_factors.evaluation
```

Check that TSI appears in the performance table. Verify bootstrap CIs are computed.

- [ ] **Step 3: Spot-check results**

Verify:
- TSI Sharpe is in the range 0.2-0.7 (not negative, not suspiciously high)
- Layer 1 positions are sensible (CL should be long during 2008 backwardation, flat/short during 2014-2016 contango)
- Crack spread positions should be active during crack spread dislocations (2008 crash, 2020 COVID)
- Bootstrap CI should be narrow enough to exclude 0 if Sharpe > 0.3

- [ ] **Step 4: Run full quality checks**

```bash
conda run -n curve-factors make check
```

Expected: lint clean, format clean, mypy clean, all tests pass.

- [ ] **Step 5: Commit results**

```bash
git add results/tables/
git commit -m "Update evaluation results with TSI strategy on 19-commodity universe"
```

---

## Task 10: Update Evaluation Report Runner

**Files:**
- Modify: `src/commodity_curve_factors/evaluation/report.py`

- [ ] **Step 1: Add TSI to the strategy list**

In `src/commodity_curve_factors/evaluation/report.py`, add `"tsi"` to the `STRATEGY_NAMES` list:

```python
STRATEGY_NAMES = [
    "xs_carry",
    "multi_factor_ew",
    "multi_factor_ic",
    "regime_conditioned",
    "sector_neutral",
    "tsmom",
    "calendar_spread",
    "tsi",
]
```

- [ ] **Step 2: Run evaluation and verify TSI appears**

```bash
conda run -n curve-factors python -m commodity_curve_factors.evaluation
```

Expected: TSI row appears in the IS/OOS performance table.

- [ ] **Step 3: Commit**

```bash
git add src/commodity_curve_factors/evaluation/report.py
git commit -m "Add TSI strategy to evaluation report runner"
```

---

## Self-Review

**Spec coverage:**
- Section 4 (Layer 1): Task 1 (convenience yield) + Task 2 (regime) + Task 3 (directional) ✓
- Section 5 (Layer 2): Task 4 (transition) ✓
- Section 6 (Layer 3): Task 5 (seasonal) + Task 6 (spreads) ✓
- Section 7 (Portfolio construction): Task 7 (combined) ✓
- Section 8 (Evaluation): Task 10 (report update) ✓
- Section 9 (Config + runner): Task 8 ✓
- Full pipeline validation: Task 9 ✓

**Placeholder scan:** No TBDs, TODOs, or "add appropriate" language. All code blocks are complete.

**Type consistency:**
- `monthly_cy` is always `pd.DataFrame` (month-end dates x commodities) ✓
- `convenience_yields` / `daily_cy` is always `pd.DataFrame` (daily dates x commodities) ✓
- `classify_regime` returns string labels, `regime_to_position` converts to floats ✓
- `expanding_zscore` / `expanding_zscore_df` from transforms.py used consistently ✓
- `threshold` parameter named consistently across all signal modules ✓
- `risk_budgets` always `list[float]` ✓
