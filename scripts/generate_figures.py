"""
Generate all publication-quality figures from results.

Usage:
    python scripts/generate_figures.py
    make figures
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    """Generate all figures."""
    print("=" * 60)
    print("Commodity Curve Factors — Figure Generation")
    print("=" * 60)

    output_dir = Path(__file__).resolve().parent.parent / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Each visualization module will be called as it is implemented.
    # Uncomment steps as they become available.

    # print("\n[1/4] Generating curve visualizations...")
    # from commodity_curve_factors.visualization import curves
    # curves.generate_all(output_dir)

    # print("\n[2/4] Generating factor visualizations...")
    # from commodity_curve_factors.visualization import factors
    # factors.generate_all(output_dir)

    # print("\n[3/4] Generating performance visualizations...")
    # from commodity_curve_factors.visualization import performance
    # performance.generate_all(output_dir)

    # print("\n[4/4] Generating tearsheet...")
    # from commodity_curve_factors.visualization import tearsheet
    # tearsheet.generate(output_dir)

    print(f"\nAll figures saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
