"""Run all 7 strategy backtests and cost sensitivity analysis.

Loads pre-computed factor parquets from ``data/processed/factors/``, constructs
portfolio weights for each strategy, runs the backtest engine, and saves results
to ``data/processed/backtest/``.

Usage:
    python -m commodity_curve_factors.backtest
    make backtest
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_curve_factors.backtest.benchmarks import (
    cash_benchmark,
    equal_weight_long,
    load_market_benchmarks,
)
from commodity_curve_factors.backtest.engine import run_backtest
from commodity_curve_factors.backtest.sensitivity import run_cost_sensitivity
from commodity_curve_factors.curves.builder import load_curves
from commodity_curve_factors.curves.convenience_yield import (
    compute_convenience_yield,
    estimate_storage_cost,
    monthly_convenience_yield,
)
from commodity_curve_factors.data.futures_loader import load_front_month_data
from commodity_curve_factors.data.macro_loader import load_macro_data
from commodity_curve_factors.signals.calendar_spreads import calendar_spread_signal
from commodity_curve_factors.signals.combined_strategy import combine_layers
from commodity_curve_factors.signals.curve_transition import (
    compute_transition_signal,
    transition_to_position,
)
from commodity_curve_factors.signals.directional import build_directional_weights
from commodity_curve_factors.signals.portfolio import build_portfolio
from commodity_curve_factors.signals.ranking import rank_and_select, resample_weights_weekly
from commodity_curve_factors.signals.spreads import (
    compute_cy_crack,
    crack_spread_signal,
    inventory_overlay,
    livestock_spread_signal,
)
from commodity_curve_factors.signals.threshold import threshold_signal
from commodity_curve_factors.utils.config import load_config
from commodity_curve_factors.utils.paths import DATA_PROCESSED

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_factor(name: str) -> pd.DataFrame:
    """Load a saved factor DataFrame from ``data/processed/factors/``.

    Parameters
    ----------
    name : str
        Factor file stem (e.g. ``"carry"``).

    Returns
    -------
    pd.DataFrame
        DatetimeIndex × commodity columns.
    """
    path = DATA_PROCESSED / "factors" / f"{name}.parquet"
    logger.info("Loading factor %s from %s", name, path)
    return pd.read_parquet(path)


def _build_returns(futures: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build daily log returns from front-month close prices.

    Parameters
    ----------
    futures : dict[str, pd.DataFrame]
        Keyed by commodity symbol; each value must have a ``Close`` column.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex × commodity columns, values = log(P_t / P_{t-1}).
    """
    prices = pd.DataFrame(
        {sym: df["Close"] for sym, df in futures.items() if "Close" in df.columns}
    )
    log_ret = np.log(prices / prices.shift(1))
    return pd.DataFrame(log_ret, index=prices.index, columns=prices.columns)


def _extract_vix(macro: dict[str, pd.DataFrame]) -> pd.Series:
    """Extract daily VIX Close series from the macro data dict.

    Returns an empty Series if VIX data is unavailable.

    Parameters
    ----------
    macro : dict[str, pd.DataFrame]
        Output of ``load_macro_data()``.

    Returns
    -------
    pd.Series
        DatetimeIndex of VIX close values.
    """
    vix_df = macro.get("vix")
    if vix_df is None or vix_df.empty:
        logger.warning(
            "VIX data not available — regime-conditioned strategy will use equal-weight fallback"
        )
        return pd.Series(dtype=float, name="vix")

    col = "Close" if "Close" in vix_df.columns else vix_df.columns[0]
    return vix_df[col].rename("vix")


def _sector_groups(universe_cfg: dict) -> dict[str, list[str]]:
    """Build a mapping of sector → list of commodity symbols.

    Parameters
    ----------
    universe_cfg : dict
        Universe YAML config (``commodities`` key).

    Returns
    -------
    dict[str, list[str]]
        E.g. ``{"energy": ["CL", "NG", ...], "metals": [...], ...}``.
    """
    groups: dict[str, list[str]] = {}
    for sym, info in universe_cfg.get("commodities", {}).items():
        if isinstance(info, dict) and "sector" in info:
            groups.setdefault(info["sector"], []).append(sym)
    return groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate all 7 strategy backtests and cost sensitivity."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    out_dir: Path = DATA_PROCESSED / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load configs
    # ------------------------------------------------------------------
    strategy_cfg = load_config("strategy")
    backtest_cfg = load_config("backtest")
    universe_cfg = load_config("universe")

    cost_config = backtest_cfg["costs"]
    sensitivity_bps: list[int] = backtest_cfg.get("sensitivity_cost_bps", [0, 2, 5, 8, 10, 15, 20])

    # ------------------------------------------------------------------
    # Load market data
    # ------------------------------------------------------------------
    logger.info("Loading front-month futures data")
    futures = load_front_month_data()
    if not futures:
        logger.error("No futures data available — run 'make data' first")
        return
    returns = _build_returns(futures)

    logger.info("Loading macro data")
    macro = load_macro_data()
    # VIX is available via _extract_vix(macro) if live composite recomputation is needed;
    # here we load pre-computed composite factors from parquet.

    # ------------------------------------------------------------------
    # Load pre-computed factors
    # ------------------------------------------------------------------
    logger.info("Loading factor files")
    carry = _load_factor("carry")
    tsmom = _load_factor("tsmom")
    composite_ew = _load_factor("composite_ew")
    composite_ic = _load_factor("composite_ic")
    composite_regime = _load_factor("composite_regime")

    # Shared execution config
    rebal_day: str = strategy_cfg.get("execution", {}).get("rebalance_day", "friday")

    # Strategy-level config shortcuts
    xs_carry_cfg = strategy_cfg["strategies"]["cross_sectional_carry"]
    mf_cfg = strategy_cfg["strategies"]["multi_factor"]
    ts_cfg = strategy_cfg["strategies"].get("time_series_carry", {})

    strategies: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Strategy 1: Cross-Sectional Carry
    # ------------------------------------------------------------------
    logger.info("Strategy 1: Cross-Sectional Carry")
    raw_w = rank_and_select(
        carry,
        long_n=xs_carry_cfg["long_n"],
        short_n=xs_carry_cfg["short_n"],
    )
    raw_w = resample_weights_weekly(raw_w, rebalance_day=rebal_day)
    w = build_portfolio(raw_w, returns, strategy_cfg, universe_cfg)
    strategies["xs_carry"] = run_backtest(w, returns, cost_config)

    # ------------------------------------------------------------------
    # Strategy 2: Multi-Factor Equal-Weight
    # ------------------------------------------------------------------
    logger.info("Strategy 2: Multi-Factor Equal-Weight")
    raw_w = rank_and_select(
        composite_ew,
        long_n=mf_cfg["long_n"],
        short_n=mf_cfg["short_n"],
    )
    raw_w = resample_weights_weekly(raw_w, rebalance_day=rebal_day)
    w = build_portfolio(raw_w, returns, strategy_cfg, universe_cfg)
    strategies["multi_factor_ew"] = run_backtest(w, returns, cost_config)

    # ------------------------------------------------------------------
    # Strategy 3: Multi-Factor IC-Weighted
    # ------------------------------------------------------------------
    logger.info("Strategy 3: Multi-Factor IC-Weighted")
    raw_w = rank_and_select(
        composite_ic,
        long_n=mf_cfg["long_n"],
        short_n=mf_cfg["short_n"],
    )
    raw_w = resample_weights_weekly(raw_w, rebalance_day=rebal_day)
    w = build_portfolio(raw_w, returns, strategy_cfg, universe_cfg)
    strategies["multi_factor_ic"] = run_backtest(w, returns, cost_config)

    # ------------------------------------------------------------------
    # Strategy 4: Regime-Conditioned
    # ------------------------------------------------------------------
    logger.info("Strategy 4: Regime-Conditioned")
    raw_w = rank_and_select(
        composite_regime,
        long_n=mf_cfg["long_n"],
        short_n=mf_cfg["short_n"],
    )
    raw_w = resample_weights_weekly(raw_w, rebalance_day=rebal_day)
    w = build_portfolio(raw_w, returns, strategy_cfg, universe_cfg)
    strategies["regime_conditioned"] = run_backtest(w, returns, cost_config)

    # ------------------------------------------------------------------
    # Strategy 5: Sector-Neutral Multi-Factor
    # ------------------------------------------------------------------
    logger.info("Strategy 5: Sector-Neutral Multi-Factor")
    groups = _sector_groups(universe_cfg)
    sector_weight_parts: list[pd.DataFrame] = []
    for sector, syms in groups.items():
        # Keep only symbols that have factor data
        available = [s for s in syms if s in composite_ew.columns]
        if len(available) < 2:
            logger.debug("Sector %s: fewer than 2 available symbols, skipping", sector)
            continue
        sector_scores = composite_ew[available]
        sw = rank_and_select(sector_scores, long_n=1, short_n=1)
        sector_weight_parts.append(sw)

    if sector_weight_parts:
        # Combine across sectors; fill missing symbols with 0
        raw_w = pd.concat(sector_weight_parts, axis=1).fillna(0.0)
        raw_w = resample_weights_weekly(raw_w, rebalance_day=rebal_day)
        w = build_portfolio(raw_w, returns, strategy_cfg, universe_cfg)
        strategies["sector_neutral"] = run_backtest(w, returns, cost_config)
    else:
        logger.warning("Strategy 5: no sector had enough symbols; skipping")

    # ------------------------------------------------------------------
    # Strategy 6: TSMOM
    # ------------------------------------------------------------------
    logger.info("Strategy 6: Time-Series Momentum (TSMOM)")
    threshold = float(ts_cfg.get("threshold_z", 0.0))
    raw_w = threshold_signal(tsmom, threshold=threshold)
    # Replace NaN with 0 before normalizing so NaN commodities are excluded
    raw_w_filled = raw_w.fillna(0.0)
    n_active = (raw_w_filled != 0).sum(axis=1).replace(0, 1)
    raw_w_norm = raw_w_filled.div(n_active, axis=0)
    raw_w_norm = resample_weights_weekly(raw_w_norm, rebalance_day=rebal_day)
    w = build_portfolio(raw_w_norm, returns, strategy_cfg, universe_cfg)
    strategies["tsmom"] = run_backtest(w, returns, cost_config)

    # ------------------------------------------------------------------
    # Strategy 7: Calendar Spread Carry
    # ------------------------------------------------------------------
    logger.info("Strategy 7: Calendar Spread Carry")
    cs_cfg = strategy_cfg["strategies"].get("calendar_spread", {})
    spread_signal = calendar_spread_signal(
        carry,
        long_threshold=cs_cfg.get("long_threshold", 0.5),
        short_threshold=cs_cfg.get("short_threshold", -0.5),
    )
    # spread_signal has MultiIndex columns (commodity, leg).
    # The directional signal is captured entirely by the front leg:
    #   backwardation → front = +1 (long front, short back)
    #   contango      → front = -1 (short front, long back)
    # front + back always sums to 0, so we use the front leg as the
    # per-commodity directional weight for the cross-sectional portfolio.
    commodities_in_signal = spread_signal.columns.get_level_values("commodity").unique()
    net_spread = pd.DataFrame(index=spread_signal.index)
    for sym in commodities_in_signal:
        front = spread_signal.get((sym, "front"), pd.Series(0.0, index=spread_signal.index))
        net_spread[sym] = front
    # Scale by 1 / N active spread positions
    n_active_spread = (net_spread != 0).sum(axis=1).replace(0, 1)
    raw_w_spread = net_spread.div(n_active_spread, axis=0)
    raw_w_spread = resample_weights_weekly(raw_w_spread, rebalance_day=rebal_day)
    w = build_portfolio(raw_w_spread, returns, strategy_cfg, universe_cfg)
    strategies["calendar_spread"] = run_backtest(w, returns, cost_config)

    # ------------------------------------------------------------------
    # Strategy 8: Term Structure Intelligence (TSI)
    # ------------------------------------------------------------------
    logger.info("Strategy 8: Term Structure Intelligence (TSI)")
    tsi_cfg = strategy_cfg.get("tsi", {})
    try:
        curves = load_curves()
        if not curves:
            raise ValueError("No curve data found — run 'make curves' first")

        # Risk-free rate
        rf_df = macro.get("dgs3mo")
        if rf_df is not None and not rf_df.empty:
            rf_col = "Close" if "Close" in rf_df.columns else rf_df.columns[0]
            rf_series = rf_df[rf_col].rename("rf")
        else:
            logger.warning("TSI: dgs3mo not available — using constant 2.0%% risk-free rate")
            all_curve_dates = sorted(set().union(*(df.index for df in curves.values())))
            rf_series = pd.Series(
                2.0,
                index=pd.DatetimeIndex(all_curve_dates),
                name="rf",
            )

        storage_costs = estimate_storage_cost(curves, is_end="2017-12-31")
        daily_cy = compute_convenience_yield(curves, rf_series, storage_costs, tenor="F6M")
        monthly_cy = monthly_convenience_yield(daily_cy)

        # --- Layer 1: Curve Directional (long-biased regime tilt) ---
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

        # --- Layer 2: Curve Transition ---
        trans_cfg = tsi_cfg.get("curve_transition", {})
        lookback_days = int(trans_cfg.get("lookback_days", 63))
        threshold_std = float(trans_cfg.get("threshold_std", 0.5))
        transition = compute_transition_signal(monthly_cy, lookback=lookback_days)
        layer2_raw = transition_to_position(transition, tsmom, threshold=threshold_std)
        layer2 = resample_weights_weekly(layer2_raw, rebalance_day=rebal_day)

        # --- Layer 3: Structural Spreads ---
        spread_cfg = tsi_cfg.get("structural_spreads", {})

        # Crack spread
        crack_cfg = spread_cfg.get("crack_spread", {})
        crack_threshold = float(crack_cfg.get("z_threshold", 1.5))
        cy_crack = compute_cy_crack(daily_cy)
        layer3_crack = crack_spread_signal(cy_crack, threshold=crack_threshold)

        # Inventory overlay on crack positions (CL component)
        inv_cfg = spread_cfg.get("inventory_overlay", {})
        inv_amplification = float(inv_cfg.get("amplification", 1.5))
        # Use CY change as proxy for inventory signal (no EIA data for non-energies)
        cy_change_cl = daily_cy["CL"].diff() if "CL" in daily_cy.columns else pd.Series(dtype=float)
        inv_surprise_proxy = -cy_change_cl  # positive CY change → draw (negative surprise)
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
            logger.warning("TSI: LC or LH price data unavailable — skipping livestock layer")
            layer3_livestock = pd.DataFrame(dtype=float)
        else:
            layer3_livestock = livestock_spread_signal(
                lc_close,
                lh_close,
                seasonal_years=ls_years,
                threshold=ls_threshold,
            )

        # Combine crack + livestock into layer3
        if layer3_livestock.empty:
            layer3 = layer3_crack.reindex(returns.index).fillna(0.0)
        else:
            layer3 = pd.concat(
                [
                    layer3_crack.reindex(returns.index),
                    layer3_livestock.reindex(returns.index),
                ],
                axis=1,
            ).fillna(0.0)

        # Combine all layers with risk budgets
        dir_budget = float(dir_cfg.get("risk_budget", 0.40))
        trans_budget = float(trans_cfg.get("risk_budget", 0.25))
        spread_budget = float(spread_cfg.get("risk_budget", 0.35))

        layer1_aligned = layer1.reindex(returns.index).fillna(0.0)
        layer2_aligned = layer2.reindex(returns.index).fillna(0.0)
        layer3_aligned = layer3.reindex(returns.index).fillna(0.0)

        tsi_raw = combine_layers(
            [layer1_aligned, layer2_aligned, layer3_aligned],
            [dir_budget, trans_budget, spread_budget],
            returns,
        )

        tsi_w = build_portfolio(tsi_raw, returns, strategy_cfg, universe_cfg)
        strategies["tsi"] = run_backtest(tsi_w, returns, cost_config)

    except Exception:
        logger.exception("Strategy 8 (TSI) failed — skipping")

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------
    logger.info("Computing benchmarks")
    ew_long = equal_weight_long(returns)
    cash = cash_benchmark(returns.index)
    market_bm = load_market_benchmarks(macro)

    bm_df = pd.DataFrame({"equal_weight_long": ew_long, "cash": cash})
    for key, series in market_bm.items():
        bm_df[key] = series
    bm_df.to_parquet(out_dir / "benchmarks.parquet")
    logger.info("Saved benchmarks: shape=%s", bm_df.shape)

    # ------------------------------------------------------------------
    # Save strategy results
    # ------------------------------------------------------------------
    for name, result in strategies.items():
        out_path = out_dir / f"{name}.parquet"
        result.to_parquet(out_path)
        cum_final = float(result["cumulative"].iloc[-1]) if not result.empty else float("nan")
        logger.info(
            "Saved %s → %s (shape=%s, final_cum=%.3f)",
            name,
            out_path,
            result.shape,
            cum_final,
        )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    logger.info("=== Backtest Summary ===")
    for name, result in strategies.items():
        if result.empty:
            logger.info("  %-22s: (empty)", name)
            continue
        net = result["net_return"]
        std = float(net.std())
        sharpe = net.mean() / std * np.sqrt(_TRADING_DAYS_PER_YEAR) if std > 0 else 0.0
        cum = float(result["cumulative"].iloc[-1])
        dd = float(result["drawdown"].min())
        logger.info(
            "  %-22s: Sharpe=%.2f, Cum=%.2f, MaxDD=%.1f%%",
            name,
            sharpe,
            cum,
            dd * 100,
        )

    # ------------------------------------------------------------------
    # Cost sensitivity (flagship: multi_factor_ew)
    # ------------------------------------------------------------------
    if "multi_factor_ew" in strategies:
        logger.info("Running cost sensitivity on multi_factor_ew flagship strategy")
        # Reconstruct constrained weights for the flagship (use composite_ew)
        raw_w_flagship = rank_and_select(
            composite_ew,
            long_n=mf_cfg["long_n"],
            short_n=mf_cfg["short_n"],
        )
        raw_w_flagship = resample_weights_weekly(raw_w_flagship, rebalance_day=rebal_day)
        w_flagship = build_portfolio(raw_w_flagship, returns, strategy_cfg, universe_cfg)

        sensitivity_df = run_cost_sensitivity(w_flagship, returns, sensitivity_bps)
        sens_path = out_dir / "cost_sensitivity.parquet"
        sensitivity_df.to_parquet(sens_path)
        logger.info("Saved cost sensitivity → %s (shape=%s)", sens_path, sensitivity_df.shape)
        logger.info("\n%s", sensitivity_df.to_string(index=False))
    else:
        logger.warning("multi_factor_ew strategy missing — skipping cost sensitivity")

    logger.info("Done — results in %s", out_dir)


if __name__ == "__main__":
    main()
