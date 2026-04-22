#!/usr/bin/env python3
"""
Create datasets with 170 effort-related metrics for clustering experiments.

This script creates:
1. Calcite-effort170-cov-only.csv: 170 effort + 5 coverage features
2. Calcite-top30-sm-effort170-cov.csv: 30 SM + 170 effort + 5 coverage features
"""

import pandas as pd
from pathlib import Path
from glob import glob
import json

# Configuration
EFFORT_DATA_FILE = "effort_data/All Calcite 1.0.0-1.15.0 effort-related metrics.xlsx"
EFFORT_SHEET = "Calcite_All"
EFFORT_HEADER_ROW = 9

COVERAGE_FEATURES = [
    "COV_INSTRUCTION",
    "COV_BRANCH",
    "COV_LINE",
    "COV_COMPLEXITY",
    "COV_METHOD",
]

METADATA_COLS = ["Calcite version", "ID", "file", "Version-ID", "Bug"]


def get_top30_sm_features(results_path: str = "results/calcite/dbscan/results.json") -> list:
    """Get top 30 SM features from clustering results."""
    with open(results_path) as f:
        results = json.load(f)

    feature_relevance = results["feature_relevance"]
    sm_features = [f for f in feature_relevance if f["feature"].startswith("SM_")]
    sorted_features = sorted(sm_features, key=lambda x: x["relevance"], reverse=True)
    return [f["feature"] for f in sorted_features[:30]]


def load_coverage_data() -> pd.DataFrame:
    """Load and merge all coverage CSV files."""
    coverage_dfs = []
    for cov_file in sorted(Path("data").glob("Coverage-Calcite-*.csv")):
        version = cov_file.stem.replace("Coverage-Calcite-", "").replace("-filename", "")
        df = pd.read_csv(cov_file)
        df["version"] = version
        coverage_dfs.append(df)

    coverage_df = pd.concat(coverage_dfs, ignore_index=True)
    cov_merge = coverage_df[["filename", "version"] + COVERAGE_FEATURES].copy()
    cov_merge = cov_merge.rename(columns={"filename": "file", "version": "Calcite version"})
    return cov_merge


def create_effort170_cov_only():
    """Create dataset with 170 effort + 5 coverage features."""
    output_file = "data/Calcite-effort170-cov-only.csv"

    print("Creating 170 effort + 5 coverage dataset...")

    # Load effort data (170 features)
    print(f"  Loading effort data from {EFFORT_DATA_FILE}...")
    effort_df = pd.read_excel(EFFORT_DATA_FILE, sheet_name=EFFORT_SHEET, header=EFFORT_HEADER_ROW)
    print(f"    Shape: {effort_df.shape}")

    # Filter out version 1.0.0 (no coverage data)
    original_count = len(effort_df)
    effort_df = effort_df[effort_df["Calcite version"] != "1.0.0"]
    print(f"    Filtered to v1.1+: {len(effort_df)} rows (removed {original_count - len(effort_df)})")

    # Load coverage data
    print("  Loading coverage data...")
    cov_df = load_coverage_data()
    print(f"    Coverage rows: {len(cov_df)}")

    # Merge effort and coverage
    print("  Merging datasets...")
    merged = effort_df.merge(cov_df, on=["file", "Calcite version"], how="inner")
    print(f"    Merged shape: {merged.shape}")

    # Save
    merged.to_csv(output_file, index=False)
    print(f"  Output: {output_file}")

    # Summary
    effort_cols = [c for c in merged.columns if not c.startswith("COV_") and c not in METADATA_COLS]
    cov_cols = [c for c in merged.columns if c.startswith("COV_")]
    print(f"  Features: {len(effort_cols)} effort + {len(cov_cols)} coverage = {len(effort_cols) + len(cov_cols)} total")

    return output_file


def create_top30_sm_effort170_cov():
    """Create dataset with 30 SM + 170 effort + 5 coverage features."""
    output_file = "data/Calcite-top30-sm-effort170-cov.csv"

    print("\nCreating 30 SM + 170 effort + 5 coverage dataset...")

    # Get top 30 SM features
    top_sm = get_top30_sm_features()
    print(f"  Selected {len(top_sm)} SM features")

    # Load SM data
    print("  Loading SM data...")
    sm_df = pd.read_csv("data/Calcite-SM-only.csv")
    available_sm = [f for f in top_sm if f in sm_df.columns]
    sm_subset = sm_df[METADATA_COLS + available_sm].copy()
    print(f"    SM shape: {sm_subset.shape}")

    # Filter to v1.1+ (for coverage data)
    sm_subset = sm_subset[sm_subset["Calcite version"] != "1.0.0"]
    print(f"    After v1.1+ filter: {len(sm_subset)} rows")

    # Load effort data
    print("  Loading effort data...")
    effort_df = pd.read_excel(EFFORT_DATA_FILE, sheet_name=EFFORT_SHEET, header=EFFORT_HEADER_ROW)
    effort_df = effort_df[effort_df["Calcite version"] != "1.0.0"]

    # Get effort columns (exclude metadata)
    effort_cols = [c for c in effort_df.columns if c not in METADATA_COLS]
    effort_merge = effort_df[["Version-ID"] + effort_cols].copy()
    print(f"    Effort features: {len(effort_cols)}")

    # Load coverage data
    print("  Loading coverage data...")
    cov_df = load_coverage_data()

    # Merge all
    print("  Merging datasets...")
    merged = sm_subset.merge(cov_df, on=["file", "Calcite version"], how="inner")
    print(f"    After coverage merge: {merged.shape}")
    merged = merged.merge(effort_merge, on="Version-ID", how="inner")
    print(f"    After effort merge: {merged.shape}")

    # Save
    merged.to_csv(output_file, index=False)
    print(f"  Output: {output_file}")

    # Summary
    sm_count = len([c for c in merged.columns if c.startswith("SM_")])
    cov_count = len([c for c in merged.columns if c.startswith("COV_")])
    total_features = len(merged.columns) - len(METADATA_COLS)
    effort_count = total_features - sm_count - cov_count
    print(f"  Features: {sm_count} SM + {effort_count} effort + {cov_count} coverage = {total_features} total")

    return output_file


def main():
    # Create both datasets
    create_effort170_cov_only()
    create_top30_sm_effort170_cov()

    print("\nDatasets created. Add to config.py:")
    print("""
    "calcite-effort170-cov-only": {
        "file": "data/Calcite-effort170-cov-only.csv",
        "sheet": None,
        "header_row": 0,
        "feature_name_row": None,
        "label_column": "Bug",
        "description": "Calcite: 170 effort + 5 coverage features (175 total)",
    },
    "calcite-top30-sm-effort170-cov": {
        "file": "data/Calcite-top30-sm-effort170-cov.csv",
        "sheet": None,
        "header_row": 0,
        "feature_name_row": None,
        "label_column": "Bug",
        "description": "Calcite: 30 SM + 170 effort + 5 coverage (205 features)",
    },
""")


if __name__ == "__main__":
    main()
