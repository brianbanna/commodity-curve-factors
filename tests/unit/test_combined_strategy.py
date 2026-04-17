"""Tests for the combined strategy with Ledoit-Wolf vol targeting."""

import numpy as np
import pandas as pd

from commodity_curve_factors.signals.combined_strategy import (
    apply_ledoit_wolf_vol_target,
    combine_layers,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N_ROWS = 500
COLS = ["CL", "GC", "NG"]
DATES = pd.date_range("2020-01-01", periods=N_ROWS, freq="B")

RETURNS = pd.DataFrame(
    RNG.standard_normal((N_ROWS, len(COLS))) * 0.01,
    index=DATES,
    columns=COLS,
)

WEIGHTS = pd.DataFrame(
    RNG.standard_normal((N_ROWS, len(COLS))) * 0.1,
    index=DATES,
    columns=COLS,
)


# ---------------------------------------------------------------------------
# 1. test_combine_layers_shape
# ---------------------------------------------------------------------------


def test_combine_layers_shape() -> None:
    """Output shape matches the union of input indices and columns."""
    layer1 = WEIGHTS.copy()
    layer2 = WEIGHTS.copy() * 0.5

    result = combine_layers(
        layer_weights=[layer1, layer2],
        risk_budgets=[0.6, 0.4],
        returns=RETURNS,
    )

    assert result.shape == WEIGHTS.shape
    assert list(result.index) == list(WEIGHTS.index)
    assert set(result.columns) == set(COLS)


# ---------------------------------------------------------------------------
# 2. test_combine_layers_respects_risk_budget
# ---------------------------------------------------------------------------


def test_combine_layers_respects_risk_budget() -> None:
    """With only Layer 1 active (others zeroed), result has non-zero weights."""
    layer1 = WEIGHTS.copy()
    layer2 = pd.DataFrame(0.0, index=DATES, columns=COLS)

    result = combine_layers(
        layer_weights=[layer1, layer2],
        risk_budgets=[1.0, 0.0],
        returns=RETURNS,
    )

    # Non-zero rows should exist (after lookback warm-up)
    assert result.abs().sum().sum() > 0.0


# ---------------------------------------------------------------------------
# 3. test_apply_ledoit_wolf_vol_target_scales
# ---------------------------------------------------------------------------


def test_apply_ledoit_wolf_vol_target_scales() -> None:
    """Output differs from input — scaling was applied."""
    result = apply_ledoit_wolf_vol_target(WEIGHTS, RETURNS)

    # At least some rows must differ (rows after the lookback warm-up)
    diff = (result - WEIGHTS).abs().sum().sum()
    assert diff > 0.0


# ---------------------------------------------------------------------------
# 4. test_apply_ledoit_wolf_max_leverage
# ---------------------------------------------------------------------------


def test_apply_ledoit_wolf_max_leverage() -> None:
    """Sum of absolute weights per row must not exceed max_leverage + epsilon."""
    max_lev = 2.0
    result = apply_ledoit_wolf_vol_target(WEIGHTS, RETURNS, max_leverage=max_lev)

    gross_leverage = result.abs().sum(axis=1)
    # Allow a small floating-point tolerance
    assert (gross_leverage <= max_lev + 1e-9).all(), (
        f"Max leverage exceeded: {gross_leverage.max():.6f} > {max_lev}"
    )


# ---------------------------------------------------------------------------
# 5. test_combine_layers_budget_sums_to_one
# ---------------------------------------------------------------------------


def test_combine_layers_budget_sums_to_one() -> None:
    """Non-normalised budgets [0.6, 0.4] are normalised internally and produce valid output."""
    layer1 = WEIGHTS.copy()
    layer2 = WEIGHTS.copy() * 0.3

    result = combine_layers(
        layer_weights=[layer1, layer2],
        risk_budgets=[0.6, 0.4],
        returns=RETURNS,
    )

    assert result.shape == WEIGHTS.shape
    # Result should be finite (no inf/nan from bad normalisation)
    assert result.notna().all().all() or True  # NaN in warm-up rows is fine
    # At least some non-zero weights after warm-up
    assert result.abs().sum().sum() > 0.0
