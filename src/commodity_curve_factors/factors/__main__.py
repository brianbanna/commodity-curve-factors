"""Compute all factor signals and save to data/processed/factors/.

Usage:
    python -m commodity_curve_factors.factors
    make factors
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_curve_factors.curves.builder import load_curves
from commodity_curve_factors.data.cftc_loader import load_cot_data
from commodity_curve_factors.data.futures_loader import load_front_month_data
from commodity_curve_factors.data.inventory_loader import load_inventory_data
from commodity_curve_factors.data.macro_loader import load_macro_data
from commodity_curve_factors.factors.carry import compute_carry_factor
from commodity_curve_factors.factors.combination import (
    equal_weight_composite,
    ic_weighted_composite,
    regime_conditioned_composite,
)
from commodity_curve_factors.factors.curvature import compute_curvature_factor
from commodity_curve_factors.factors.curve_momentum import compute_curve_momentum_factor
from commodity_curve_factors.factors.inventory import compute_all_inventory_surprises
from commodity_curve_factors.factors.macro import compute_macro_factor
from commodity_curve_factors.factors.momentum_ts import tsmom_signal
from commodity_curve_factors.factors.momentum_xs import xsmom_signal
from commodity_curve_factors.factors.positioning import compute_positioning_factor
from commodity_curve_factors.factors.slope import compute_slope_factor
from commodity_curve_factors.factors.volatility import vol_regime_ratio
from commodity_curve_factors.utils.config import load_config
from commodity_curve_factors.utils.paths import DATA_PROCESSED

logger = logging.getLogger(__name__)


def main() -> None:
    """Compute all 10 factor signals and 3 composite signals; save to Parquet."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_dir: Path = DATA_PROCESSED / "factors"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load all data sources ---
    logger.info("Loading data sources")
    curves = load_curves()
    futures = load_front_month_data()
    macro_data = load_macro_data()
    inventory_data = load_inventory_data()
    cot = load_cot_data()
    universe = load_config("universe")
    factors_config = load_config("factors")
    strategy_config = load_config("strategy")

    all_commodities: list[str] = list(universe["commodities"].keys())

    # Build price DataFrame from front-month data (Close column)
    prices = pd.DataFrame(
        {sym: df["Close"] for sym, df in futures.items() if "Close" in df.columns}
    )

    # Compute daily log returns
    returns: pd.DataFrame = pd.DataFrame(
        np.log(prices / prices.shift(1)),
        index=prices.index,
        columns=prices.columns,
    )

    # --- Curve factors ---
    logger.info("Computing curve factors")
    carry = compute_carry_factor(curves)
    slope = compute_slope_factor(curves)
    curvature = compute_curvature_factor(curves)
    curve_mom = compute_curve_momentum_factor(curves)

    # --- Momentum factors ---
    logger.info("Computing momentum factors")
    tsmom = tsmom_signal(prices)
    xsmom = xsmom_signal(prices)

    # --- Inventory ---
    logger.info("Computing inventory surprise")
    commodity_inventory_map: dict[str, str] = {
        "CL": "crude_stocks",
        "NG": "natural_gas_storage",
        "HO": "distillate_stocks",
        "RB": "gasoline_stocks",
    }
    inventory_years: int = factors_config.get("inventory", {}).get("surprise_seasonal_years", 5)
    inventory = compute_all_inventory_surprises(
        inventory_data,
        commodity_inventory_map,
        all_commodities,
        years=inventory_years,
    )

    # --- Positioning ---
    logger.info("Computing positioning factor")
    positioning = compute_positioning_factor(cot, all_commodities=all_commodities)

    # --- Macro ---
    logger.info("Computing macro factor")
    macro = compute_macro_factor(returns, macro_data)

    # --- Volatility ---
    logger.info("Computing volatility regime")
    vol = vol_regime_ratio(returns)

    # --- Save individual factors ---
    factor_dict: dict[str, pd.DataFrame] = {
        "carry": carry,
        "slope": slope,
        "curvature": curvature,
        "curve_momentum": curve_mom,
        "tsmom": tsmom,
        "xsmom": xsmom,
        "inventory": inventory,
        "positioning": positioning,
        "macro": macro,
        "volatility": vol,
    }

    for name, df in factor_dict.items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path)
        logger.info("Saved %s: shape=%s", name, df.shape)

    # --- Composites ---
    logger.info("Computing composite factors")

    ew = equal_weight_composite(factor_dict)
    ew.to_parquet(out_dir / "composite_ew.parquet")
    logger.info("Saved composite_ew: shape=%s", ew.shape)

    # IC-weighted: forward returns = next-day return, shifted by -1.
    # Row t of fwd_returns holds the return from day t to day t+1, so
    # when weighting factor scores at day t we use the contemporaneous
    # forward return without any future information leaking back.
    fwd_returns = returns.shift(-1)
    ic_cfg = factors_config.get("combination", {}).get("ic_weighted", {})
    ic = ic_weighted_composite(
        factor_dict,
        fwd_returns,
        lookback=ic_cfg.get("lookback_days", 252),
        min_observations=ic_cfg.get("min_observations", 60),
    )
    ic.to_parquet(out_dir / "composite_ic.parquet")
    logger.info("Saved composite_ic: shape=%s", ic.shape)

    # Regime-conditioned: requires VIX data
    vix_df = macro_data.get("vix", pd.DataFrame())
    if not vix_df.empty:
        # VIX is stored from yfinance with a Close column
        if "Close" in vix_df.columns:
            vix_close = vix_df["Close"]
        else:
            vix_close = vix_df.iloc[:, 0]
        regime_cfg = strategy_config.get("regime_conditioned", {})
        regime = regime_conditioned_composite(
            factor_dict,
            vix_close,
            vix_thresholds=regime_cfg.get("vix_thresholds", [15, 25]),
            weights_by_regime=regime_cfg.get("weights_by_regime"),
        )
        regime.to_parquet(out_dir / "composite_regime.parquet")
        logger.info("Saved composite_regime: shape=%s", regime.shape)
    else:
        logger.warning("VIX data not found — skipping regime-conditioned composite")

    logger.info("All factors saved to %s", out_dir)


if __name__ == "__main__":
    main()
