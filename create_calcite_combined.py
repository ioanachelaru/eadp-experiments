#!/usr/bin/env python3
"""
Create combined Calcite dataset with:
- Top 30 SM features (by clustering relevance)
- 5 coverage features
- 26 effort-based features

Total: 61 features + metadata columns
"""

import json
import pandas as pd
from pathlib import Path


def get_top_30_sm_features():
    """Extract top 30 SM feature names from clustering results."""
    results_file = Path("results/calcite/dbscan/results.json")
    with open(results_file) as f:
        data = json.load(f)

    # Get feature relevance data
    feature_relevance = data["feature_relevance"]

    # Filter for SM_* features only, sort by relevance, then take top 30
    sm_features = [f for f in feature_relevance if f["feature"].startswith("SM_")]
    sorted_features = sorted(sm_features, key=lambda x: x["relevance"], reverse=True)
    top_30 = [f["feature"] for f in sorted_features[:30]]

    print(f"Top 30 SM features:")
    for i, feat in enumerate(top_30, 1):
        print(f"  {i}. {feat}")

    return top_30


def get_26_effort_features():
    """Get the 26 effort features common to Ant-Ivy and Calcite."""
    # These are from the "26 Metrics (Ant int Calcite)" sheet
    effort_features = [
        "CHANGE_TYPE_computation",
        "CHANGE_TYPE_data",
        "CHANGE_TYPE_other",
        "HASSAN_edhcm",
        "HASSAN_hcm",
        "HASSAN_ldhcm",
        "HASSAN_lgdhcm",
        "HASSAN_whcm",
        "ISSUE_major_bug",
        "ISSUE_major_improvement",
        "MOSER_authors",
        "MOSER_bugfix",
        "MOSER_revisions",
        "MOSER_sum_lines_deleted",
        "MOSER_weighted_age",
        "PMD_arp",
        "PMD_cis",
        "PMD_odpl",
        "PMD_rule_type_basic rules",
        "PMD_rule_type_controversial rules",
        "PMD_rule_type_design rules",
        "PMD_rule_type_string and stringbuffer rules",
        "PMD_rule_type_unnecessary and unused code rules",
        "PMD_severity_critical",
        "PMD_severity_major",
        "PMD_severity_minor",
    ]
    print(f"\n26 effort features:")
    for i, feat in enumerate(effort_features, 1):
        print(f"  {i}. {feat}")
    return effort_features


def load_sm_data(top_30_features):
    """Load SM data and extract top 30 features."""
    print("\nLoading SM data...")
    sm_df = pd.read_csv("data/Calcite-SM-only.csv")
    print(f"  SM data shape: {sm_df.shape}")

    # Metadata columns
    metadata_cols = ["Calcite version", "ID", "file", "Version-ID", "Bug"]

    # Check which top 30 features exist
    available = [f for f in top_30_features if f in sm_df.columns]
    missing = [f for f in top_30_features if f not in sm_df.columns]

    if missing:
        print(f"  Warning: {len(missing)} top-30 features not found in SM data:")
        for f in missing:
            print(f"    - {f}")

    # Select metadata + top 30 features
    cols_to_keep = metadata_cols + available
    sm_subset = sm_df[cols_to_keep].copy()
    print(f"  Selected {len(available)} SM features + {len(metadata_cols)} metadata cols")

    return sm_subset


def load_coverage_data():
    """Load and combine coverage data from all version files."""
    print("\nLoading coverage data...")
    coverage_files = sorted(Path("data").glob("Coverage-Calcite-*.csv"))

    coverage_dfs = []
    for cov_file in coverage_files:
        # Extract version from filename like "Coverage-Calcite-1.1.0-filename.csv"
        version = cov_file.stem.replace("Coverage-Calcite-", "").replace("-filename", "")

        df = pd.read_csv(cov_file)
        df["version"] = version
        coverage_dfs.append(df)
        print(f"  Loaded {cov_file.name}: {len(df)} rows")

    coverage_df = pd.concat(coverage_dfs, ignore_index=True)
    print(f"  Combined coverage data: {coverage_df.shape}")

    # Rename columns to have COV_ prefix for clarity (already have it)
    # Create merge key: file + version
    coverage_df["Version-ID"] = coverage_df["version"] + "-" + coverage_df.index.astype(str)

    return coverage_df


def load_effort_data(effort_features):
    """Load effort data from Excel file."""
    print("\nLoading effort data...")
    effort_file = "effort_data/All Calcite 1.0.0-1.15.0 effort-related metrics.xlsx"

    # Read with header at row 9 (0-indexed)
    effort_df = pd.read_excel(effort_file, sheet_name="Calcite_Correlation", header=9)
    print(f"  Effort data shape: {effort_df.shape}")

    # Check which effort features exist
    available = [f for f in effort_features if f in effort_df.columns]
    missing = [f for f in effort_features if f not in effort_df.columns]

    if missing:
        print(f"  Warning: {len(missing)} effort features not found:")
        for f in missing:
            print(f"    - {f}")

    # Keep metadata + available effort features
    metadata_cols = ["Calcite version", "ID", "file", "Version-ID", "Bug"]
    cols_to_keep = metadata_cols + available
    effort_subset = effort_df[cols_to_keep].copy()
    print(f"  Selected {len(available)} effort features")

    return effort_subset


def merge_datasets(sm_df, coverage_df, effort_df, top_30_features, effort_features):
    """Merge SM, coverage, and effort data."""
    print("\nMerging datasets...")

    # Start with SM data
    merged = sm_df.copy()
    print(f"  Starting with SM data: {merged.shape}")

    # Merge coverage data on file and version
    # Coverage has 'filename' and 'version', SM has 'file' and 'Calcite version'
    coverage_cols = ["COV_INSTRUCTION", "COV_BRANCH", "COV_LINE", "COV_COMPLEXITY", "COV_METHOD"]

    # Prepare coverage for merge
    cov_merge = coverage_df[["filename", "version"] + coverage_cols].copy()
    cov_merge = cov_merge.rename(columns={"filename": "file", "version": "Calcite version"})

    # Merge coverage
    merged = merged.merge(cov_merge, on=["file", "Calcite version"], how="inner")
    print(f"  After merging coverage: {merged.shape}")

    # Merge effort data on Version-ID
    effort_cols = [f for f in effort_features if f in effort_df.columns]
    effort_merge = effort_df[["Version-ID"] + effort_cols].copy()

    merged = merged.merge(effort_merge, on="Version-ID", how="inner")
    print(f"  After merging effort: {merged.shape}")

    return merged


def main():
    print("=" * 60)
    print("Creating combined Calcite dataset")
    print("Top 30 SM + 5 Coverage + 26 Effort features")
    print("=" * 60)

    # Get feature lists
    top_30_sm = get_top_30_sm_features()
    effort_26 = get_26_effort_features()
    coverage_5 = ["COV_INSTRUCTION", "COV_BRANCH", "COV_LINE", "COV_COMPLEXITY", "COV_METHOD"]

    # Load data
    sm_df = load_sm_data(top_30_sm)
    coverage_df = load_coverage_data()
    effort_df = load_effort_data(effort_26)

    # Merge
    combined = merge_datasets(sm_df, coverage_df, effort_df, top_30_sm, effort_26)

    # Verify feature counts
    metadata_cols = ["Calcite version", "ID", "file", "Version-ID", "Bug"]
    feature_cols = [c for c in combined.columns if c not in metadata_cols]

    sm_feat_count = len([c for c in feature_cols if c.startswith("SM_")])
    cov_feat_count = len([c for c in feature_cols if c.startswith("COV_")])
    effort_feat_count = len([c for c in feature_cols if not c.startswith("SM_") and not c.startswith("COV_")])

    print("\n" + "=" * 60)
    print("Final dataset summary:")
    print(f"  Total rows: {len(combined)}")
    print(f"  Metadata columns: {len(metadata_cols)}")
    print(f"  SM features: {sm_feat_count}")
    print(f"  Coverage features: {cov_feat_count}")
    print(f"  Effort features: {effort_feat_count}")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Total columns: {len(combined.columns)}")

    # Bug distribution
    print(f"\nBug distribution:")
    print(combined["Bug"].value_counts())

    # Save
    output_file = "data/Calcite-top30-sm-cov-effort.csv"
    combined.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")

    # List all feature columns
    print("\nFeature columns:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")


if __name__ == "__main__":
    main()
