"""
Demo: How to use the KSI reference implementation.

This script demonstrates the basic usage of the KSI calculation
using the included sample data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from ksi import compute_ksi, plot_ksi


def main():
    # Load sample data
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "sample_data.csv"
    print(f"Loading sample data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print()

    # Compute KSI with default settings (60-month window)
    result = compute_ksi(df, window=60)

    # Show results
    print("\n--- Results ---")
    print(result[["date", "KSI"]].tail(12).to_string(index=False))

    # Plot
    plot_ksi(result)


if __name__ == "__main__":
    main()
