"""TSI layer decomposition chart (Chart 09).

Reconstructs per-layer cumulative returns from saved signals and price data,
showing which layer of the Term Structure Intelligence strategy contributes what.

Call ``generate_all()`` to produce the chart in one shot.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from commodity_curve_factors.curves.builder import load_curves
from commodity_curve_factors.curves.convenience_yield import (
    compute_convenience_yield,
    estimate_storage_cost,
    monthly_convenience_yield,
)
from commodity_curve_factors.data.futures_loader import load_front_month_data
from commodity_curve_factors.data.macro_loader import load_macro_data
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
from commodity_curve_factors.utils.config import load_config
from commodity_curve_factors.utils.paths import DATA_PROCESSED
from commodity_curve_factors.visualization.style import (
    ACCENT,
    DOWN_COLOR,
    FG_COLOR,
    UP_COLOR,
    add_is_oos_divider,
    savefig,
)

logger = logging.getLogger(__name__)

_BACKTEST_DIR = DATA_PROCESSED / "backtest"
_FACTORS_DIR = DATA_PROCESSED / "factors"

LAYER_COLORS: dict[str, str] = {
    "Layer 1: Directional": ACCENT,
    "Layer 2: Transition": UP_COLOR,
    "Layer 3: Spreads": "#6894be",
    "Combined TSI": DOWN_COLOR,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_returns(futures: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build daily simple pct_change returns from front-month close prices.

    Parameters
    ----------
    futures : dict[str, pd.DataFrame]
        Keyed by commodity symbol; each value must have a ``Close`` column.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex × commodity columns.
    """
    prices = pd.DataFrame(
        {sym: df["Close"] for sym, df in futures.items() if "Close" in df.columns}
    )
    return prices.pct_change(fill_method=None)


def _layer_cumulative(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Compute cumulative return series for a set of weights.

    Parameters
    ----------
    weights : pd.DataFrame
        Daily position weights (dates x commodities).
    returns : pd.DataFrame
        Daily pct returns (dates x commodities).

    Returns
    -------
    pd.Series
        Cumulative wealth index (base = 1.0).
    """
    common_cols = weights.columns.intersection(returns.columns)
    w = weights[common_cols].reindex(returns.index).fillna(0.0)
    r = returns[common_cols].reindex(returns.index).fillna(0.0)
    layer_ret = (w * r).sum(axis=1)
    return (1.0 + layer_ret).cumprod()


# ---------------------------------------------------------------------------
# Chart 09 — TSI Layer Decomposition
# ---------------------------------------------------------------------------


def plot_tsi_layer_decomposition() -> Path:
    """Chart 09: Per-layer cumulative returns for the TSI 3-layer strategy.

    Reconstructs each layer independently from saved signals and price data.
    The Combined TSI line is loaded from the saved backtest result.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # ------------------------------------------------------------------
    # Combined TSI from saved backtest (always attempted first)
    # ------------------------------------------------------------------
    combined_ok = False
    try:
        tsi_df = pd.read_parquet(_BACKTEST_DIR / "tsi.parquet")
        tsi_df.index = pd.to_datetime(tsi_df.index)
        combined_cum = (1.0 + tsi_df["net_return"].fillna(0.0)).cumprod()
        ax.plot(
            combined_cum.index,
            combined_cum.values,
            color=LAYER_COLORS["Combined TSI"],
            linewidth=2.0,
            linestyle="--",
            label="Combined TSI",
            zorder=5,
        )
        combined_ok = True
        logger.info("Loaded combined TSI from saved backtest")
    except Exception:
        logger.warning("Could not load combined TSI backtest result")

    # ------------------------------------------------------------------
    # Load shared data needed for layer reconstruction
    # ------------------------------------------------------------------
    try:
        futures = load_front_month_data()
        if not futures:
            raise ValueError("No futures data — run 'make data' first")
        returns = _build_returns(futures)

        macro = load_macro_data()
        tsmom = pd.read_parquet(_FACTORS_DIR / "tsmom.parquet")

        strategy_cfg = load_config("strategy")
        tsi_cfg = strategy_cfg.get("tsi", {})

        curves = load_curves()
        if not curves:
            raise ValueError("No curve data — run 'make curves' first")

        rf_df = macro.get("dgs3mo")
        if rf_df is not None and not rf_df.empty:
            rf_col = "Close" if "Close" in rf_df.columns else rf_df.columns[0]
            rf_series = rf_df[rf_col].rename("rf")
        else:
            all_curve_dates = sorted(set().union(*(df.index for df in curves.values())))
            rf_series = pd.Series(2.0, index=pd.DatetimeIndex(all_curve_dates), name="rf")
            logger.warning("TSI viz: dgs3mo unavailable — using constant 2.0%% risk-free rate")

        storage_costs = estimate_storage_cost(curves, is_end="2017-12-31")
        daily_cy = compute_convenience_yield(curves, rf_series, storage_costs, tenor="F6M")
        monthly_cy = monthly_convenience_yield(daily_cy)

        shared_ok = True
    except Exception:
        logger.warning("Could not load shared data for TSI layer reconstruction", exc_info=True)
        shared_ok = False

    # ------------------------------------------------------------------
    # Layer 1: Directional
    # ------------------------------------------------------------------
    if shared_ok:
        try:
            dir_cfg = tsi_cfg.get("curve_directional", {})
            thresholds = dir_cfg.get("regime_thresholds")
            position_map = dir_cfg.get("position_map")
            trend_up = float(dir_cfg.get("trend_up_mult", 1.2))
            trend_down = float(dir_cfg.get("trend_down_mult", 0.7))

            layer1 = build_directional_weights(
                monthly_cy,
                tsmom,
                returns.index,
                thresholds=thresholds,
                position_map=position_map,
                trend_up_mult=trend_up,
                trend_down_mult=trend_down,
            )
            cum1 = _layer_cumulative(layer1, returns)
            ax.plot(
                cum1.index,
                cum1.values,
                color=LAYER_COLORS["Layer 1: Directional"],
                linewidth=1.2,
                linestyle="-",
                label="Layer 1: Directional",
                zorder=3,
            )
            logger.info("Layer 1 (Directional) computed successfully")
        except Exception:
            logger.warning("Layer 1 (Directional) failed", exc_info=True)

    # ------------------------------------------------------------------
    # Layer 2: Transition Momentum
    # ------------------------------------------------------------------
    if shared_ok:
        try:
            trans_cfg = tsi_cfg.get("curve_transition", {})
            lookback_days = int(trans_cfg.get("lookback_days", 63))
            threshold_std = float(trans_cfg.get("threshold_std", 0.5))

            transition = compute_transition_signal(monthly_cy, lookback=lookback_days)
            layer2 = transition_to_position(transition, tsmom, threshold=threshold_std)

            cum2 = _layer_cumulative(layer2, returns)
            ax.plot(
                cum2.index,
                cum2.values,
                color=LAYER_COLORS["Layer 2: Transition"],
                linewidth=1.2,
                linestyle="-",
                label="Layer 2: Transition",
                zorder=3,
            )
            logger.info("Layer 2 (Transition) computed successfully")
        except Exception:
            logger.warning("Layer 2 (Transition) failed", exc_info=True)

    # ------------------------------------------------------------------
    # Layer 3: Structural Spreads
    # ------------------------------------------------------------------
    if shared_ok:
        try:
            spread_cfg = tsi_cfg.get("structural_spreads", {})

            # Crack spread
            crack_cfg = spread_cfg.get("crack_spread", {})
            crack_threshold = float(crack_cfg.get("z_threshold", 1.5))
            cy_crack = compute_cy_crack(daily_cy)
            layer3_crack = crack_spread_signal(cy_crack, threshold=crack_threshold)

            # Inventory overlay on crack positions
            inv_cfg = spread_cfg.get("inventory_overlay", {})
            inv_amplification = float(inv_cfg.get("amplification", 1.5))
            cy_change_cl = (
                daily_cy["CL"].diff() if "CL" in daily_cy.columns else pd.Series(dtype=float)
            )
            inv_surprise_proxy = -cy_change_cl
            layer3_crack = inventory_overlay(
                layer3_crack,
                inventory_surprise=inv_surprise_proxy,
                cy_change=cy_change_cl,
                amplification=inv_amplification,
            )

            # Livestock spread
            ls_cfg = spread_cfg.get("livestock_spread", {})
            ls_threshold = float(ls_cfg.get("z_threshold", 1.5))
            ls_years = int(ls_cfg.get("seasonal_lookback_years", 5))
            lc_df = futures.get("LC", pd.DataFrame())
            lh_df = futures.get("LH", pd.DataFrame())
            lc_close = lc_df["Close"] if "Close" in lc_df.columns else pd.Series(dtype=float)
            lh_close = lh_df["Close"] if "Close" in lh_df.columns else pd.Series(dtype=float)

            if lc_close.empty or lh_close.empty:
                logger.warning("TSI viz: LC or LH unavailable — using crack only for Layer 3")
                layer3 = layer3_crack.reindex(returns.index).fillna(0.0)
            else:
                layer3_livestock = livestock_spread_signal(
                    lc_close,
                    lh_close,
                    seasonal_years=ls_years,
                    threshold=ls_threshold,
                )
                layer3 = pd.concat(
                    [
                        layer3_crack.reindex(returns.index),
                        layer3_livestock.reindex(returns.index),
                    ],
                    axis=1,
                ).fillna(0.0)

            cum3 = _layer_cumulative(layer3, returns)
            ax.plot(
                cum3.index,
                cum3.values,
                color=LAYER_COLORS["Layer 3: Spreads"],
                linewidth=1.2,
                linestyle="-",
                label="Layer 3: Spreads",
                zorder=3,
            )
            logger.info("Layer 3 (Spreads) computed successfully")
        except Exception:
            logger.warning("Layer 3 (Spreads) failed", exc_info=True)

    if not combined_ok and not shared_ok:
        logger.error("All TSI layer data sources failed — chart will be empty")

    # ------------------------------------------------------------------
    # Annotations and formatting
    # ------------------------------------------------------------------
    add_is_oos_divider(ax)
    ax.axhline(1.0, color=FG_COLOR, linewidth=0.4, alpha=0.3, linestyle=":")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}×"))
    ax.set_title("TSI Layer Decomposition — Cumulative Returns per Layer")
    ax.set_xlabel("")
    ax.set_ylabel("Wealth Index")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    return savefig(fig, "09_tsi_layer_decomposition")


# ---------------------------------------------------------------------------
# generate_all
# ---------------------------------------------------------------------------


def generate_all() -> list[Path]:
    """Generate the TSI layer decomposition chart.

    Catches exceptions so a failure does not crash the caller.

    Returns
    -------
    list[Path]
        Paths of successfully saved figures.
    """
    paths: list[Path] = []
    try:
        path = plot_tsi_layer_decomposition()
        paths.append(path)
        logger.info("Generated chart: plot_tsi_layer_decomposition → %s", path)
    except Exception:
        logger.exception("Failed to generate chart: plot_tsi_layer_decomposition")
    return paths


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from commodity_curve_factors.visualization.style import setup

    setup()
    generate_all()
