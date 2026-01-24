#!/usr/bin/env python3
"""
Merge test coverage features with Calcite software metrics.

This script:
1. Loads all coverage CSV files (versions 1.1.0-1.15.0)
2. Loads Calcite software metrics Excel file
3. Joins them on version + filename
4. Optionally filters to top-N features from previous clustering
5. Saves the merged dataset

Usage:
    python3 merge_coverage.py                    # Full merge (all features)
    python3 merge_coverage.py --top-features 30  # Top 30 + coverage only
"""

import argparse
import glob
import json
import os

import pandas as pd


# Configuration
SM_FILE = "data/All Calcite 1.0.0-1.15.0 software metrics.xlsx"
SM_SHEET = "All SM"
SM_HEADER_ROW = 9
COVERAGE_PATTERN = "data/Coverage-Calcite-*-filename.csv"
CALCITE_RESULTS = "results/calcite/dbscan/results.json"

# Columns to keep from coverage files
COVERAGE_COLS = ["COV_INSTRUCTION", "COV_BRANCH", "COV_LINE", "COV_COMPLEXITY", "COV_METHOD"]

# Metadata columns to always keep
META_COLS = ["Calcite version", "file", "Bug"]


def load_coverage_data() -> pd.DataFrame:
    """Load all coverage CSV files and combine them."""
    all_coverage = []

    for csv_file in sorted(glob.glob(COVERAGE_PATTERN)):
        # Extract version from filename (e.g., "Coverage-Calcite-1.1.0-filename.csv" -> "1.1.0")
        basename = os.path.basename(csv_file)
        version = basename.split("-")[2]

        print(f"  Loading {basename}...")
        df = pd.read_csv(csv_file)
        df["version"] = version

        # Rename filename column for clarity
        df = df.rename(columns={"filename": "file"})

        all_coverage.append(df)

    combined = pd.concat(all_coverage, ignore_index=True)
    print(f"  Total coverage rows: {len(combined)}")

    return combined


def load_sm_data() -> pd.DataFrame:
    """Load Calcite software metrics, excluding version 1.0.0."""
    print(f"  Loading {SM_FILE}...")
    df = pd.read_excel(SM_FILE, sheet_name=SM_SHEET, header=SM_HEADER_ROW)

    # Remove rows with NaN version
    df = df.dropna(subset=["Calcite version"])

    # Exclude version 1.0.0 (no coverage data available)
    original_count = len(df)
    df = df[df["Calcite version"] != "1.0.0"]
    excluded = original_count - len(df)

    print(f"  Total SM rows: {original_count}")
    print(f"  Excluded 1.0.0 rows: {excluded}")
    print(f"  Remaining rows: {len(df)}")

    return df


def merge_data(sm_df: pd.DataFrame, cov_df: pd.DataFrame) -> pd.DataFrame:
    """Merge SM and coverage data on version + file."""
    # Prepare coverage data for merge
    cov_subset = cov_df[["version", "file"] + COVERAGE_COLS].copy()

    # Merge on version and file
    merged = pd.merge(
        sm_df,
        cov_subset,
        left_on=["Calcite version", "file"],
        right_on=["version", "file"],
        how="left"
    )

    # Drop the duplicate version column from coverage
    merged = merged.drop(columns=["version"])

    # Check for unmatched rows
    unmatched = merged[COVERAGE_COLS[0]].isna().sum()
    if unmatched > 0:
        print(f"  WARNING: {unmatched} rows did not match coverage data")
    else:
        print(f"  All rows matched successfully!")

    return merged


def get_top_features(n: int) -> list[str]:
    """Load top N features from previous Calcite clustering results."""
    with open(CALCITE_RESULTS) as f:
        results = json.load(f)

    top_features = [
        feat["feature"] for feat in results["feature_relevance"][:n]
    ]
    return top_features


def main():
    parser = argparse.ArgumentParser(
        description="Merge coverage features with Calcite software metrics"
    )
    parser.add_argument(
        "--top-features", "-t",
        type=int,
        default=None,
        help="Only keep top N features from previous clustering (default: all)",
    )
    args = parser.parse_args()

    # Determine output file based on mode
    if args.top_features:
        output_file = f"data/Calcite-top{args.top_features}-with-coverage.csv"
    else:
        output_file = "data/Calcite-with-coverage.csv"

    print("=" * 60)
    print("Merging Coverage Data with Calcite Software Metrics")
    if args.top_features:
        print(f"  Mode: Top {args.top_features} features + coverage")
    else:
        print("  Mode: All features + coverage")
    print("=" * 60)

    print("\nStep 1: Loading coverage data...")
    cov_df = load_coverage_data()

    print("\nStep 2: Loading software metrics...")
    sm_df = load_sm_data()

    print("\nStep 3: Merging datasets...")
    merged_df = merge_data(sm_df, cov_df)

    # Filter to top features if requested
    if args.top_features:
        print(f"\nStep 3b: Filtering to top {args.top_features} features...")
        top_features = get_top_features(args.top_features)
        print(f"  Top features: {top_features[:5]}... (and {len(top_features)-5} more)")

        # Keep only meta columns + top features + coverage
        cols_to_keep = META_COLS + top_features + COVERAGE_COLS

        # Check which columns exist in merged_df
        existing_cols = [c for c in cols_to_keep if c in merged_df.columns]
        missing_cols = [c for c in cols_to_keep if c not in merged_df.columns]

        if missing_cols:
            print(f"  WARNING: Missing columns: {missing_cols[:5]}...")

        merged_df = merged_df[existing_cols]
        print(f"  Filtered to {len(existing_cols)} columns")

    print(f"\nStep 4: Saving to {output_file}...")
    # Check coverage column statistics
    print("\nCoverage feature statistics:")
    for col in COVERAGE_COLS:
        if col in merged_df.columns:
            mean_val = merged_df[col].mean()
            non_zero = (merged_df[col] > 0).sum()
            print(f"  {col}: mean={mean_val:.4f}, non-zero={non_zero} ({non_zero/len(merged_df)*100:.1f}%)")

    merged_df.to_csv(output_file, index=False)
    print(f"\nMerged dataset saved: {output_file}")
    print(f"  Rows: {len(merged_df)}")
    print(f"  Columns: {len(merged_df.columns)}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
