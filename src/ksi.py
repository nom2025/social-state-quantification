"""
Kernel Stress Index (KSI) - Reference Implementation

Computes the KSI, a unified state quantity that integrates heterogeneous
public statistics into a single continuous indicator of household-level
socioeconomic pressure.

Algorithm (from the paper):
  1. Convert each indicator into a z-score (60-month rolling window)
  2. Align indicator polarity toward "stress-increasing" direction
  3. Apply LOCF to harmonize monthly data
  4. Aggregate via vector summation to produce a unified state quantity

Usage:
  python src/ksi.py --input data/sample_data.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    DEFAULT_INDICATORS,
    align_polarity,
    locf,
    rolling_zscore,
)


def compute_ksi(
    df: pd.DataFrame,
    indicators: list | None = None,
    window: int = 60,
) -> pd.DataFrame:
    """
    Compute the Kernel Stress Index from a DataFrame of monthly indicators.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'date' column and indicator columns.
    indicators : list of (column_name, invert) tuples, optional
        Indicator configuration. Defaults to DEFAULT_INDICATORS.
    window : int
        Rolling window for z-score normalization (default: 60 months).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, KSI, and individual z-scores.
    """
    if indicators is None:
        indicators = DEFAULT_INDICATORS

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    z_scores = pd.DataFrame({"date": df["date"]})

    available = []
    for col, invert in indicators:
        if col not in df.columns:
            print(f"  Warning: '{col}' not found in data, skipping.")
            continue

        series = df[col].astype(float)
        series = locf(series)
        z = rolling_zscore(series, window=window)
        z = align_polarity(z, invert=invert)
        z_scores[f"z_{col}"] = z
        available.append(f"z_{col}")

    if not available:
        raise ValueError("No valid indicators found in the data.")

    # Vector summation: KSI = sum of all z-scores
    z_scores["KSI"] = z_scores[available].sum(axis=1)

    print(f"  KSI computed from {len(available)} indicators.")
    return z_scores


def plot_ksi(result: pd.DataFrame, output_path: str | None = None) -> None:
    """
    Plot the KSI time series.

    Parameters
    ----------
    result : pd.DataFrame
        Output from compute_ksi().
    output_path : str, optional
        If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(result["date"], result["KSI"], color="#d32f2f", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(
        result["date"],
        result["KSI"],
        0,
        where=result["KSI"] > 0,
        alpha=0.3,
        color="#d32f2f",
        label="Stress > 0",
    )
    ax.fill_between(
        result["date"],
        result["KSI"],
        0,
        where=result["KSI"] <= 0,
        alpha=0.3,
        color="#1976d2",
        label="Stress <= 0",
    )

    ax.set_title("Kernel Stress Index (KSI)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("KSI (aggregated z-score)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"  Figure saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compute the Kernel Stress Index (KSI)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/sample_data.csv",
        help="Path to input CSV file (default: data/sample_data.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the KSI plot (optional)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Rolling window size in months (default: 60)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    print("Computing KSI...")
    result = compute_ksi(df, window=args.window)

    # Display summary
    print("\n--- KSI Summary ---")
    print(f"  Period: {result['date'].min()} to {result['date'].max()}")
    print(f"  Max KSI: {result['KSI'].max():.3f}")
    print(f"  Min KSI: {result['KSI'].min():.3f}")
    print(f"  Latest KSI: {result['KSI'].iloc[-1]:.3f}")

    # Save results
    output_csv = input_path.parent / "ksi_results.csv"
    result.to_csv(output_csv, index=False)
    print(f"\n  Results saved to {output_csv}")

    # Plot
    plot_ksi(result, args.output)


if __name__ == "__main__":
    main()
