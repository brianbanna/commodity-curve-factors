"""Evaluation report: IS/OOS metrics for all strategies + benchmarks."""

import logging

import numpy as np
import pandas as pd

from commodity_curve_factors.evaluation.bootstrap import bootstrap_sharpe_ci
from commodity_curve_factors.evaluation.metrics import (
    compute_all_metrics,
    split_is_oos,
)
from commodity_curve_factors.evaluation.stress import historical_stress_test
from commodity_curve_factors.utils.paths import DATA_PROCESSED, RESULTS

logger = logging.getLogger(__name__)

STRATEGY_NAMES = [
    "xs_carry",
    "multi_factor_ew",
    "multi_factor_ic",
    "regime_conditioned",
    "sector_neutral",
    "tsmom",
    "calendar_spread",
]


def _load_strategy_returns(name: str) -> pd.Series:
    """Load net returns for a strategy from backtest output."""
    path = DATA_PROCESSED / "backtest" / f"{name}.parquet"
    df = pd.read_parquet(path)
    return df["net_return"]


def _load_benchmark_returns() -> dict[str, pd.Series]:
    """Load benchmark return series."""
    path = DATA_PROCESSED / "backtest" / "benchmarks.parquet"
    df = pd.read_parquet(path)
    benchmarks = {}
    for col in df.columns:
        benchmarks[col] = df[col].dropna()
    return benchmarks


def build_performance_table() -> pd.DataFrame:
    """Build IS/OOS performance table for all strategies and benchmarks.

    Returns
    -------
    pd.DataFrame
        Rows: strategies + benchmarks. Columns include IS and OOS metrics,
        bootstrap CI, turnover.
    """
    rows = []

    for name in STRATEGY_NAMES:
        try:
            net_ret = _load_strategy_returns(name)
        except FileNotFoundError:
            logger.warning("Strategy %s not found, skipping", name)
            continue

        is_ret, oos_ret = split_is_oos(net_ret)
        is_metrics = compute_all_metrics(is_ret)
        oos_metrics = compute_all_metrics(oos_ret)

        # Bootstrap CI on full sample
        point, ci_lo, ci_hi = bootstrap_sharpe_ci(net_ret, n_samples=10000, seed=42)

        # Load turnover
        path = DATA_PROCESSED / "backtest" / f"{name}.parquet"
        df = pd.read_parquet(path)
        mean_turnover = float(df["turnover"].mean()) if "turnover" in df.columns else np.nan

        rows.append({
            "strategy": name,
            "is_sharpe": is_metrics["sharpe"],
            "is_sortino": is_metrics["sortino"],
            "is_cagr": is_metrics["cagr"],
            "is_max_dd": is_metrics["max_drawdown"],
            "is_vol": is_metrics["volatility"],
            "oos_sharpe": oos_metrics["sharpe"],
            "oos_sortino": oos_metrics["sortino"],
            "oos_cagr": oos_metrics["cagr"],
            "oos_max_dd": oos_metrics["max_drawdown"],
            "oos_vol": oos_metrics["volatility"],
            "full_sharpe": point,
            "sharpe_ci_lo": ci_lo,
            "sharpe_ci_hi": ci_hi,
            "turnover": mean_turnover,
        })

    # Benchmarks
    benchmarks = _load_benchmark_returns()
    for bm_name, bm_ret in benchmarks.items():
        is_ret, oos_ret = split_is_oos(bm_ret)
        is_metrics = compute_all_metrics(is_ret)
        oos_metrics = compute_all_metrics(oos_ret)
        point, ci_lo, ci_hi = bootstrap_sharpe_ci(bm_ret, n_samples=10000, seed=42)

        rows.append({
            "strategy": f"BM_{bm_name}",
            "is_sharpe": is_metrics["sharpe"],
            "is_sortino": is_metrics["sortino"],
            "is_cagr": is_metrics["cagr"],
            "is_max_dd": is_metrics["max_drawdown"],
            "is_vol": is_metrics["volatility"],
            "oos_sharpe": oos_metrics["sharpe"],
            "oos_sortino": oos_metrics["sortino"],
            "oos_cagr": oos_metrics["cagr"],
            "oos_max_dd": oos_metrics["max_drawdown"],
            "oos_vol": oos_metrics["volatility"],
            "full_sharpe": point,
            "sharpe_ci_lo": ci_lo,
            "sharpe_ci_hi": ci_hi,
            "turnover": np.nan,
        })

    result = pd.DataFrame(rows)
    logger.info("build_performance_table: %d strategies + benchmarks", len(result))
    return result


def build_stress_table() -> pd.DataFrame:
    """Run stress tests across all strategies.

    Returns
    -------
    pd.DataFrame
        Stress test results for each strategy x period.
    """
    all_rows = []
    for name in STRATEGY_NAMES:
        try:
            net_ret = _load_strategy_returns(name)
        except FileNotFoundError:
            continue
        stress = historical_stress_test(net_ret)
        stress.insert(0, "strategy", name)
        all_rows.append(stress)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def main() -> None:
    """Run full evaluation report and save results."""
    tables_dir = RESULTS / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building performance table")
    perf = build_performance_table()
    perf.to_parquet(tables_dir / "performance_summary.parquet")
    logger.info("Saved performance_summary.parquet")

    # Print performance table
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    logger.info("\n=== IS/OOS Performance ===\n%s", perf.to_string(index=False, float_format="%.3f"))

    logger.info("Building stress test table")
    stress = build_stress_table()
    stress.to_parquet(tables_dir / "stress_tests.parquet")
    logger.info("Saved stress_tests.parquet")
    logger.info("\n=== Stress Tests ===\n%s", stress.to_string(index=False, float_format="%.3f"))

    logger.info("Done — results in %s", tables_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
