"""
End-to-end pipeline: data -> curves -> factors -> signals -> backtest -> evaluate.

Usage:
    python scripts/run_pipeline.py
    make run
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    """Run the full pipeline."""
    print("=" * 60)
    print("Commodity Curve Factors — Full Pipeline")
    print("=" * 60)

    # Each step will be implemented as the corresponding module is built.
    # Uncomment steps as they become available.

    # Step 1: Data
    # print("\n[1/6] Downloading and processing data...")
    # from commodity_curve_factors.data import futures_loader
    # futures_loader.main()

    # Step 2: Curves
    # print("\n[2/6] Building term structures...")
    # from commodity_curve_factors.curves import builder
    # builder.main()

    # Step 3: Factors
    # print("\n[3/6] Computing factor signals...")
    # from commodity_curve_factors.factors import combination
    # combination.main()

    # Step 4: Signals
    # print("\n[4/6] Generating portfolio signals...")
    # from commodity_curve_factors.signals import portfolio
    # portfolio.main()

    # Step 5: Backtest
    # print("\n[5/6] Running backtests...")
    # from commodity_curve_factors.backtest import engine
    # engine.main()

    # Step 6: Evaluate
    # print("\n[6/6] Computing performance metrics...")
    # from commodity_curve_factors.evaluation import report
    # report.main()

    print("\n" + "=" * 60)
    print("Pipeline complete. Results saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
