"""Build and save curves for all commodities.

Usage:
    python -m commodity_curve_factors.curves
"""

import logging

from commodity_curve_factors.curves.builder import build_all_curves, save_curves
from commodity_curve_factors.curves.metrics import compute_all_metrics
from commodity_curve_factors.data.wrds_loader import load_all_contracts
from commodity_curve_factors.utils.config import load_config
from commodity_curve_factors.utils.paths import DATA_PROCESSED


def main() -> None:
    """Entry point: build curves, save them, then compute and save metrics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Loading WRDS contracts for all commodities")
    contracts = load_all_contracts()

    logger.info("Loading curve config")
    curve_config = load_config("curve")

    logger.info("Building curves")
    curves = build_all_curves(contracts, curve_config)

    logger.info("Saving curves to %s", DATA_PROCESSED / "curves")
    save_curves(curves)

    logger.info("Computing metrics")
    metrics = compute_all_metrics(curves)
    metrics_dir = DATA_PROCESSED / "curve_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for name, df in metrics.items():
        df.to_parquet(metrics_dir / f"{name}.parquet")
        logger.info("Saved %s: shape=%s", name, df.shape)

    logger.info("Done")


if __name__ == "__main__":
    main()
