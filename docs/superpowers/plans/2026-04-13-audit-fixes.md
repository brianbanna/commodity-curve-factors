# Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address the top 5 findings from the 10-reviewer committee audit to improve strategy Sharpe ratios and fix spec-code inconsistencies. The audit confirmed carry IC ~ 0 is NOT a code bug, but identified 5 implementation gaps that inflate costs, suppress signal, and break design-spec contracts.

**Architecture:** Targeted edits to existing modules — no new files. Weekly rebalancing goes into `backtest/__main__.py`, XSMOM z-score normalization into `factors/__main__.py`, calendar spread thresholds into `configs/strategy.yaml` + `backtest/__main__.py`, and the carry contamination finding is documented in `PROJECT_TRACKER.md`. An IS/OOS evaluation split is added via a new `evaluation/metrics.py` module.

**Tech Stack:** Python 3.12, pandas, numpy, pytest, ruff, mypy.

---

## What's Already Done

- Phase 1-4 complete. 343 tests passing. All 7 strategies backtested.
- Audit completed: carry IC ~ 0 is genuine (not a bug). TSMOM Sharpe +0.15 is the only positive strategy.
- Audit identified 5 high-leverage fixes, ranked by expected Sharpe impact.

## File Structure

### Files to Modify

```
src/commodity_curve_factors/
  backtest/__main__.py          # Task 1: weekly rebalancing
  factors/__main__.py           # Task 2: XSMOM z-score normalization  
  factors/momentum_xs.py        # Task 2: add z-score option to xsmom_signal
  signals/calendar_spreads.py   # (no change — thresholds come from caller)
configs/
  strategy.yaml                 # Task 3: add calendar_spread thresholds

tests/unit/
  test_ranking.py               # Task 1: add weekly rebalance test
  test_momentum_xs.py           # Task 2: add z-score variant test
```

### Files to Create

```
src/commodity_curve_factors/
  evaluation/__init__.py        # Task 5
  evaluation/metrics.py         # Task 5: IS/OOS split + core metrics
tests/unit/
  test_eval_metrics.py          # Task 5
```

---

## Task 1: Enforce Weekly Rebalancing

The audit found `strategy.yaml` says `rebalance: "weekly"` but the backtest runner rebalances daily (calling `rank_and_select` on every row). This inflates turnover from ~0.02 (weekly) to ~0.10 (daily), adding ~1% annual cost drag that converts a near-zero-Sharpe strategy into a negative-Sharpe one.

**Files:**
- Modify: `src/commodity_curve_factors/backtest/__main__.py`
- Modify: `src/commodity_curve_factors/signals/ranking.py` (add a `resample_to_weekly` helper)
- Create: `tests/unit/test_weekly_rebalance.py`

- [ ] **Step 1: Write the failing test for weekly resampling**

```python
# tests/unit/test_weekly_rebalance.py
import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.signals.ranking import resample_weights_weekly


def test_resample_weights_weekly_holds_for_5_days() -> None:
    """Weekly weights should be constant Mon-Fri within each week."""
    dates = pd.bdate_range("2020-01-06", periods=10)  # 2 full weeks
    # Daily weights that change every day
    daily = pd.DataFrame(
        {
            "CL": [0.33, 0.33, 0.33, 0.33, 0.33, -0.33, -0.33, -0.33, -0.33, -0.33],
            "NG": [-0.33, -0.33, -0.33, -0.33, -0.33, 0.33, 0.33, 0.33, 0.33, 0.33],
        },
        index=dates,
    )
    weekly = resample_weights_weekly(daily, rebalance_day="friday")
    # Within the first week (Mon-Fri), all weights should be the Friday value
    first_week = weekly.iloc[:5]
    for col in first_week.columns:
        assert first_week[col].nunique() == 1, f"{col} changed within week 1"


def test_resample_weights_weekly_changes_only_on_rebalance() -> None:
    """Weight changes should only occur on the rebalance day boundary."""
    dates = pd.bdate_range("2020-01-06", periods=15)  # 3 weeks
    rng = np.random.default_rng(42)
    daily = pd.DataFrame(
        {"CL": rng.normal(0, 1, 15), "NG": rng.normal(0, 1, 15)},
        index=dates,
    )
    weekly = resample_weights_weekly(daily, rebalance_day="friday")
    # Weight changes only at weekly boundaries
    changes = weekly.diff().abs().sum(axis=1)
    # Monday-Thursday of any week (not the first row) should have zero change
    for i in range(1, len(changes)):
        if weekly.index[i].weekday() != 0:  # not Monday (where prior Friday applies)
            assert changes.iloc[i] == 0.0, f"Unexpected weight change on {weekly.index[i]}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n curve-factors pytest tests/unit/test_weekly_rebalance.py -v
```

Expected: FAIL with `ImportError: cannot import name 'resample_weights_weekly'`

- [ ] **Step 3: Implement `resample_weights_weekly` in ranking.py**

Add to `src/commodity_curve_factors/signals/ranking.py` after the existing `rank_and_select` function:

```python
def resample_weights_weekly(
    weights: pd.DataFrame,
    rebalance_day: str = "friday",
) -> pd.DataFrame:
    """Downsample daily weights to weekly rebalancing frequency.

    Takes the weight snapshot on the ``rebalance_day`` of each week and
    forward-fills it through the following week. This ensures positions
    are held for ~5 business days between rebalances, matching the
    ``rebalance: "weekly"`` spec in strategy.yaml.

    Parameters
    ----------
    weights : pd.DataFrame
        Daily portfolio weights (DatetimeIndex x commodity columns).
    rebalance_day : str
        Day of week to take the rebalancing snapshot. Default ``"friday"``.

    Returns
    -------
    pd.DataFrame
        Same index as *weights*, but values change only at weekly boundaries.
    """
    _WEEKDAY_MAP = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4,
    }
    target_weekday = _WEEKDAY_MAP[rebalance_day.lower()]

    # Keep only the rebalance-day rows, then forward-fill to all dates
    is_rebal = weights.index.weekday == target_weekday
    rebal_only = weights.loc[is_rebal]
    return rebal_only.reindex(weights.index, method="ffill")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n curve-factors pytest tests/unit/test_weekly_rebalance.py -v
```

Expected: PASS

- [ ] **Step 5: Wire weekly resampling into the backtest runner**

In `src/commodity_curve_factors/backtest/__main__.py`, add the import at the top:

```python
from commodity_curve_factors.signals.ranking import rank_and_select, resample_weights_weekly
```

Then for EACH strategy that has `rebalance: "weekly"` in strategy.yaml (strategies 1-5 and 7), add a resampling step between `rank_and_select` and `build_portfolio`. For example, Strategy 1 (lines 190-196) changes from:

```python
raw_w = rank_and_select(carry, long_n=xs_carry_cfg["long_n"], short_n=xs_carry_cfg["short_n"])
w = build_portfolio(raw_w, returns, strategy_cfg, universe_cfg)
```

To:

```python
raw_w = rank_and_select(carry, long_n=xs_carry_cfg["long_n"], short_n=xs_carry_cfg["short_n"])
rebal_day = strategy_cfg.get("execution", {}).get("rebalance_day", "friday")
raw_w = resample_weights_weekly(raw_w, rebalance_day=rebal_day)
w = build_portfolio(raw_w, returns, strategy_cfg, universe_cfg)
```

Apply the same pattern to Strategies 2, 3, 4, 5, and 7 (calendar spread). Strategy 6 (TSMOM) uses `threshold_signal` not `rank_and_select`, so apply the resampling to `raw_w_norm` before `build_portfolio`:

```python
raw_w_norm = resample_weights_weekly(raw_w_norm, rebalance_day=rebal_day)
```

- [ ] **Step 6: Run `make check`**

```bash
conda run -n curve-factors make check
```

Expected: all tests pass including the 2 new ones.

- [ ] **Step 7: Commit**

```bash
git add src/commodity_curve_factors/signals/ranking.py src/commodity_curve_factors/backtest/__main__.py tests/unit/test_weekly_rebalance.py
git commit -m "Enforce weekly rebalancing per strategy.yaml spec

Daily rebalancing was inflating turnover ~5x (0.10 vs ~0.02) and adding
~1% annual cost drag. Weekly resampling takes the Friday weight snapshot
and forward-fills it through the following week."
```

---

## Task 2: Normalize XSMOM to Z-Score Scale

The audit found XSMOM outputs values in [0, 1] (cross-sectional rank) while all other 9 factors are z-scored (mean ~ 0, std ~ 1). When `equal_weight_composite` takes the nanmean, XSMOM's contribution is underweighted by ~3x because its values cluster around 0.5 instead of 0.

**Files:**
- Modify: `src/commodity_curve_factors/factors/__main__.py`
- Create: `tests/unit/test_xsmom_normalization.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_xsmom_normalization.py
import numpy as np
import pandas as pd

from commodity_curve_factors.factors.transforms import expanding_zscore_df


def test_xsmom_zscore_has_unit_variance() -> None:
    """After z-scoring, the XSMOM factor should have std ~ 1.0, not ~ 0.3."""
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    rng = np.random.default_rng(42)
    # Simulate rank-like data in [0, 1]
    raw = pd.DataFrame(
        rng.uniform(0, 1, (500, 5)),
        index=dates,
        columns=["CL", "NG", "GC", "ZC", "KC"],
    )
    z = expanding_zscore_df(raw, min_periods=50)
    valid = z.iloc[50:]
    mean_std = valid.std().mean()
    assert 0.8 < mean_std < 1.2, f"Expected std ~ 1.0, got {mean_std:.2f}"
```

- [ ] **Step 2: Run test to verify it passes** (this test validates the z-score transform works on [0,1] data — it should already pass since expanding_zscore handles any input)

```bash
conda run -n curve-factors pytest tests/unit/test_xsmom_normalization.py -v
```

Expected: PASS

- [ ] **Step 3: Apply z-score to XSMOM in the factor runner**

In `src/commodity_curve_factors/factors/__main__.py`, find where `xsmom` is assigned (around line 80-85) and add the z-score normalization:

```python
    xsmom = xsmom_signal(prices)
    # XSMOM is in [0, 1] (cross-sectional rank). Z-score it so it has the same
    # scale as all other factors (mean ~ 0, std ~ 1) before compositing.
    xsmom = expanding_zscore_df(xsmom, min_periods=252)
```

Make sure `expanding_zscore_df` is already imported at the top of the file. If not, add:

```python
from commodity_curve_factors.factors.transforms import expanding_zscore_df
```

- [ ] **Step 4: Run `make check`**

```bash
conda run -n curve-factors make check
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/commodity_curve_factors/factors/__main__.py tests/unit/test_xsmom_normalization.py
git commit -m "Z-score XSMOM before compositing

XSMOM was in [0,1] while all other 9 factors are z-scored (mean~0, std~1).
The nanmean composite was underweighting XSMOM by ~3x. Now z-scored with
expanding_zscore_df(min_periods=252) before inclusion in factor_dict."
```

---

## Task 3: Lower Calendar Spread Thresholds + Make Configurable

The audit found the calendar spread strategy has zero turnover because the z-score thresholds (1.0 / -1.0) are too tight — only ~16% of days per commodity breach them, and across 13 commodities the joint probability of triggering is low.

**Files:**
- Modify: `configs/strategy.yaml`
- Modify: `src/commodity_curve_factors/backtest/__main__.py`
- Create: `tests/unit/test_calendar_spread_threshold.py`

- [ ] **Step 1: Add thresholds to strategy.yaml**

Edit `configs/strategy.yaml`, change the `calendar_spread:` section from:

```yaml
  calendar_spread:
    type: "calendar_spread"
    near_tenor: 1
    far_tenor: 6
    rebalance: "weekly"
```

To:

```yaml
  calendar_spread:
    type: "calendar_spread"
    near_tenor: 1
    far_tenor: 6
    rebalance: "weekly"
    long_threshold: 0.5
    short_threshold: -0.5
```

- [ ] **Step 2: Write a test for the lower thresholds**

```python
# tests/unit/test_calendar_spread_threshold.py
import numpy as np
import pandas as pd

from commodity_curve_factors.signals.calendar_spreads import calendar_spread_signal


def test_lower_thresholds_produce_more_trades() -> None:
    """Thresholds at 0.5/-0.5 should trigger more often than 1.0/-1.0."""
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    rng = np.random.default_rng(42)
    carry_z = pd.DataFrame(
        rng.normal(0, 1, (500, 3)),
        index=dates,
        columns=["CL", "NG", "GC"],
    )
    # Tight thresholds
    tight = calendar_spread_signal(carry_z, long_threshold=1.0, short_threshold=-1.0)
    tight_active = (tight != 0).any(axis=1).sum()

    # Loose thresholds
    loose = calendar_spread_signal(carry_z, long_threshold=0.5, short_threshold=-0.5)
    loose_active = (loose != 0).any(axis=1).sum()

    assert loose_active > tight_active, (
        f"Loose thresholds should produce more active days: {loose_active} vs {tight_active}"
    )
    # With std-normal carry_z and threshold 0.5, expect ~38% of days per commodity
    # to trigger, so most days should have at least one active spread
    assert loose_active > 200, f"Expected >200 active days with 0.5 threshold, got {loose_active}"
```

- [ ] **Step 3: Run test**

```bash
conda run -n curve-factors pytest tests/unit/test_calendar_spread_threshold.py -v
```

Expected: PASS (the function already accepts threshold parameters — no code change needed)

- [ ] **Step 4: Wire config thresholds into the backtest runner**

In `src/commodity_curve_factors/backtest/__main__.py`, find the calendar spread section (around line 274-288). Change:

```python
spread_signal = calendar_spread_signal(carry)
```

To:

```python
cs_cfg = strategy_cfg["strategies"].get("calendar_spread", {})
spread_signal = calendar_spread_signal(
    carry,
    long_threshold=cs_cfg.get("long_threshold", 0.5),
    short_threshold=cs_cfg.get("short_threshold", -0.5),
)
```

- [ ] **Step 5: Run `make check`**

```bash
conda run -n curve-factors make check
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add configs/strategy.yaml src/commodity_curve_factors/backtest/__main__.py tests/unit/test_calendar_spread_threshold.py
git commit -m "Lower calendar spread thresholds to 0.5/-0.5

Old thresholds (1.0/-1.0) were too tight, producing zero turnover.
Thresholds now configurable via strategy.yaml calendar_spread section.
At 0.5/-0.5, ~38% of days per commodity should trigger."
```

---

## Task 4: Document Carry Contamination Finding

The audit confirmed that carry IC ~ 0 at lag=1 is genuine — same-day curve-price correlation inflates lag=0 metrics but vanishes once execution lag is applied. This is a research finding, not a deficiency.

**Files:**
- Modify: `.claude/PROJECT_TRACKER.md`
- Modify: `.claude/design_spec.md`

- [ ] **Step 1: Add Technical Decision to PROJECT_TRACKER.md**

Append to the Technical Decisions section:

```markdown
### Carry contemporaneous contamination (2026-04-13, audit finding)

> **Finding**: Carry IC is ~ 0 at lag=1 in this 13-commodity universe over 2005-2024
> **Context**: The 10-reviewer audit traced the carry signal end-to-end and found no code bug. Carry = (F1M - F2M) / F2M * 12 is correct. The expanding z-score has no lookahead. The execution lag is correctly applied. However, the lag=0 IC (0.048) drops to 0.006 at lag=1 because F1M is mechanically responsive to same-day spot price moves — when CL spot rises, F1M rises more than F2M, increasing carry contemporaneously with the positive return. Once the 1-day execution lag removes this contamination, the residual cross-sectional carry signal has insufficient power for 13 assets.
> **Statistical justification**: With IC=0.006 and breadth=6 independent bets per day, the fundamental law predicts SR = 0.006 * sqrt(6 * 252) = 0.23 before costs. After daily rebalancing costs (~1% annual drag), expected Sharpe drops to near zero, consistent with observed results. Weekly rebalancing (Task 1 fix) reduces cost drag to ~0.25%, potentially lifting net Sharpe to ~+0.1.
> **Academic benchmark**: Carry Sharpe of 0.5-0.9 in the literature uses 20+ commodities over 30+ year samples. With 13 commodities and a 20-year window including the 2015-2020 supercycle bust, the carry premium is within historical noise bounds.
> **Impact**: The project's value shifts from "carry generates alpha" to "production infrastructure that honestly evaluates factor signals." TSMOM is the only robust surviving signal. Cross-sectional carry may emerge with longer history, more commodities, or monthly rebalancing.
```

- [ ] **Step 2: Update design_spec.md carry formula**

Find the carry formula in `.claude/design_spec.md` section on Curve Metrics and update from `(F1M - F3M) / F3M * 4` to `(F1M - F2M) / F2M * 12` to match the code and `curve.yaml`. This resolves the stale-spec finding from Reviewer 1.

- [ ] **Step 3: No commit needed** (`.claude/` is gitignored)

---

## Task 5: Add IS/OOS Evaluation Split

The audit found no code enforces the 2005-2017 IS / 2018-2024 OOS split despite the design spec mandating it. This is the first module of Phase 5.

**Files:**
- Create: `src/commodity_curve_factors/evaluation/__init__.py`
- Create: `src/commodity_curve_factors/evaluation/metrics.py`
- Create: `tests/unit/test_eval_metrics.py`

- [ ] **Step 1: Write failing tests for evaluation metrics**

```python
# tests/unit/test_eval_metrics.py
import numpy as np
import pandas as pd
import pytest

from commodity_curve_factors.evaluation.metrics import (
    sharpe_ratio,
    max_drawdown,
    cagr,
    annual_volatility,
    compute_all_metrics,
    split_is_oos,
)


def test_sharpe_ratio_positive_for_positive_mean() -> None:
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))
    assert sharpe_ratio(returns) > 0


def test_sharpe_ratio_zero_for_zero_mean() -> None:
    returns = pd.Series([0.01, -0.01, 0.01, -0.01] * 100)
    assert abs(sharpe_ratio(returns)) < 0.5


def test_max_drawdown_negative() -> None:
    returns = pd.Series([0.01, -0.05, 0.01, -0.03, 0.02])
    dd = max_drawdown(returns)
    assert dd < 0


def test_max_drawdown_zero_for_all_positive() -> None:
    returns = pd.Series([0.01] * 100)
    dd = max_drawdown(returns)
    assert dd == 0.0


def test_cagr_positive_for_growing_wealth() -> None:
    returns = pd.Series([0.001] * 252)
    assert cagr(returns) > 0


def test_annual_volatility_scales_by_sqrt_252() -> None:
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0, 0.01, 1000))
    vol = annual_volatility(returns)
    daily_std = returns.std()
    assert abs(vol - daily_std * np.sqrt(252)) < 0.01


def test_compute_all_metrics_returns_dict() -> None:
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.0005, 0.01, 500))
    m = compute_all_metrics(returns)
    assert "sharpe" in m
    assert "max_drawdown" in m
    assert "cagr" in m
    assert "volatility" in m


def test_split_is_oos_correct_dates() -> None:
    dates = pd.bdate_range("2005-01-03", "2024-12-30")
    returns = pd.Series(0.001, index=dates)
    is_ret, oos_ret = split_is_oos(returns)
    assert is_ret.index.max() <= pd.Timestamp("2017-12-31")
    assert oos_ret.index.min() >= pd.Timestamp("2018-01-01")
    assert len(is_ret) + len(oos_ret) == len(returns)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n curve-factors pytest tests/unit/test_eval_metrics.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Create `evaluation/__init__.py`**

```python
# src/commodity_curve_factors/evaluation/__init__.py
```

(empty file)

- [ ] **Step 4: Implement `evaluation/metrics.py`**

```python
"""Core performance metrics and IS/OOS splitting.

Usage:
    from commodity_curve_factors.evaluation.metrics import compute_all_metrics, split_is_oos
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR: int = 252
_IS_END: str = "2017-12-31"
_OOS_START: str = "2018-01-01"


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    rf : float
        Daily risk-free rate. Default 0.

    Returns
    -------
    float
        Annualized Sharpe = (mean - rf) / std * sqrt(252).
    """
    excess = returns - rf
    std = float(excess.std())
    if std == 0 or len(returns) == 0:
        return 0.0
    result: float = float(excess.mean()) / std * np.sqrt(_TRADING_DAYS_PER_YEAR)
    return result


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    excess = returns - rf
    downside = excess[excess < 0]
    if len(downside) == 0 or len(returns) == 0:
        return 0.0
    down_std = float(downside.std())
    if down_std == 0:
        return 0.0
    result: float = float(excess.mean()) / down_std * np.sqrt(_TRADING_DAYS_PER_YEAR)
    return result


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from peak to trough.

    Returns
    -------
    float
        Negative value (e.g. -0.25 for 25% drawdown). Returns 0.0 if no drawdown.
    """
    cumulative = np.exp(returns.cumsum())
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1
    dd: float = float(drawdown.min())
    return dd


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate."""
    n = len(returns)
    if n == 0:
        return 0.0
    years = n / _TRADING_DAYS_PER_YEAR
    cum = float(np.exp(returns.sum()))
    if cum <= 0 or years <= 0:
        return 0.0
    result: float = cum ** (1.0 / years) - 1.0
    return result


def annual_volatility(returns: pd.Series) -> float:
    """Annualized volatility."""
    result: float = float(returns.std()) * np.sqrt(_TRADING_DAYS_PER_YEAR)
    return result


def calmar_ratio(returns: pd.Series) -> float:
    """Calmar ratio = CAGR / abs(max_drawdown)."""
    dd = max_drawdown(returns)
    if dd == 0:
        return 0.0
    result: float = cagr(returns) / abs(dd)
    return result


def hit_rate(returns: pd.Series) -> float:
    """Fraction of positive-return days."""
    if len(returns) == 0:
        return 0.0
    result: float = float((returns > 0).mean())
    return result


def compute_all_metrics(returns: pd.Series, rf: float = 0.0) -> dict[str, float]:
    """Compute all core metrics for a return series.

    Returns
    -------
    dict[str, float]
        Keys: sharpe, sortino, calmar, max_drawdown, cagr, volatility, hit_rate.
    """
    return {
        "sharpe": sharpe_ratio(returns, rf),
        "sortino": sortino_ratio(returns, rf),
        "calmar": calmar_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "cagr": cagr(returns),
        "volatility": annual_volatility(returns),
        "hit_rate": hit_rate(returns),
    }


def split_is_oos(
    returns: pd.Series,
    is_end: str = _IS_END,
    oos_start: str = _OOS_START,
) -> tuple[pd.Series, pd.Series]:
    """Split returns into in-sample and out-of-sample periods.

    Parameters
    ----------
    returns : pd.Series
        Daily returns with DatetimeIndex.
    is_end : str
        Last date of IS period. Default "2017-12-31".
    oos_start : str
        First date of OOS period. Default "2018-01-01".

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (is_returns, oos_returns).
    """
    is_ret = returns.loc[:is_end]
    oos_ret = returns.loc[oos_start:]
    return is_ret, oos_ret
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
conda run -n curve-factors pytest tests/unit/test_eval_metrics.py -v
```

Expected: PASS

- [ ] **Step 6: Run `make check`**

```bash
conda run -n curve-factors make check
```

Expected: all tests pass, lint/format/typecheck clean.

- [ ] **Step 7: Commit**

```bash
git add src/commodity_curve_factors/evaluation/__init__.py src/commodity_curve_factors/evaluation/metrics.py tests/unit/test_eval_metrics.py
git commit -m "Add evaluation metrics with IS/OOS split

Core metrics: Sharpe, Sortino, Calmar, max drawdown, CAGR, volatility,
hit rate. split_is_oos enforces 2005-2017 IS / 2018-2024 OOS per spec."
```

---

## Post-Implementation: Re-run Backtests

After all 5 tasks are committed, re-run the full pipeline to measure the impact:

```bash
# Re-compute factors (XSMOM z-score fix)
python -m commodity_curve_factors.factors

# Re-run all 7 strategies (weekly rebal + calendar thresholds)
python -m commodity_curve_factors.backtest
```

Then inspect the new results:

```python
python -c "
import pandas as pd, numpy as np
from pathlib import Path

bt = Path('data/processed/backtest')
for f in sorted(bt.glob('*.parquet')):
    if f.stem in ('benchmarks', 'cost_sensitivity'):
        continue
    df = pd.read_parquet(f)
    net = df['net_return']
    sharpe = net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else 0
    print(f'{f.stem:25s}: Sharpe={sharpe:+.2f}  Cum={df[\"cumulative\"].iloc[-1]:.3f}  Turnover={df[\"turnover\"].mean():.3f}')
"
```

**Expected impact:**
- xs_carry: Sharpe improves from -0.47 toward ~-0.10 (weekly rebal reduces cost drag)
- multi_factor_ew/ic/regime: improve similarly
- calendar_spread: goes from 0.00 to some nonzero value (lower thresholds)
- tsmom: stays at ~+0.15 (already the best strategy, weekly rebal may slightly improve)
- Factor composites improve from XSMOM scaling fix
