"""
Data preparation functions for defect prediction experiments.

Consolidates all data preparation logic:
- Extract SM features from raw Excel files
- Merge coverage data
- Create combined datasets
- Create subset datasets (top-30 SM, effort-only, etc.)
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import (
    RAW_DATA,
    COVERAGE_FEATURES,
    EFFORT_FEATURES_26,
    EFFORT_PREFIXES,
    METADATA_COLUMNS,
)


def extract_sm_features(dataset: str, output_path: Optional[str] = None) -> str:
    """
    Extract SM_* features from raw Excel data.

    Args:
        dataset: Either "calcite" or "ant-ivy"
        output_path: Optional custom output path

    Returns:
        Path to the output CSV file
    """
    if dataset == "calcite":
        input_file = RAW_DATA["calcite_sm"]
        sheet_name = "All SM"
        header_row = 9
        metadata_cols = METADATA_COLUMNS["calcite"]
        default_output = "data/Calcite-SM-only.csv"
    elif dataset == "ant-ivy":
        input_file = RAW_DATA["ant_ivy_sm"]
        sheet_name = "ant-ivy-all versions"
        header_row = 8
        feature_name_row = 7
        metadata_cols = METADATA_COLUMNS["ant_ivy"]
        default_output = "data/Ant-Ivy-SM-only.csv"
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'calcite' or 'ant-ivy'")

    output_file = output_path or default_output
    print(f"Extracting SM features from {dataset}...")
    print(f"  Input: {input_file}")

    if dataset == "ant-ivy":
        # Ant-Ivy has feature names in a different row
        df_names = pd.read_excel(
            input_file, sheet_name=sheet_name, header=feature_name_row, nrows=0
        )
        feature_names = list(df_names.columns)
        df = pd.read_excel(input_file, sheet_name=sheet_name, header=header_row)
        data_header_names = list(df.columns)

        # Combine: keep metadata names from header row, use feature names from feature row
        num_metadata_cols = 5
        combined_names = data_header_names[:num_metadata_cols] + feature_names[num_metadata_cols:]
        if len(combined_names) == len(df.columns):
            df.columns = combined_names
    else:
        df = pd.read_excel(input_file, sheet_name=sheet_name, header=header_row)

    print(f"  Total columns: {len(df.columns)}")
    print(f"  Total rows: {len(df)}")

    # Find all SM_* columns
    sm_columns = [col for col in df.columns if str(col).startswith("SM_")]
    print(f"  Found {len(sm_columns)} SM_* columns")

    # Check which metadata columns exist
    available_metadata = [col for col in metadata_cols if col in df.columns]
    if len(available_metadata) < len(metadata_cols):
        missing = set(metadata_cols) - set(available_metadata)
        print(f"  Warning: Missing metadata columns: {missing}")

    # Select columns to keep
    columns_to_keep = available_metadata + sm_columns
    df_subset = df[columns_to_keep]

    # Save to CSV
    df_subset.to_csv(output_file, index=False)
    print(f"  Output: {output_file}")
    print(f"  Shape: {df_subset.shape}")

    return output_file


def merge_coverage_data(output_path: Optional[str] = None) -> str:
    """
    Merge coverage CSV files with SM data.

    Args:
        output_path: Optional custom output path

    Returns:
        Path to the output CSV file
    """
    from glob import glob
    import os

    output_file = output_path or "data/Calcite-with-coverage.csv"
    coverage_pattern = RAW_DATA["coverage_pattern"]
    sm_file = RAW_DATA["calcite_sm"]

    print("Merging coverage data with Calcite software metrics...")

    # Load all coverage CSV files
    print("  Loading coverage data...")
    all_coverage = []
    for csv_file in sorted(glob(coverage_pattern)):
        basename = os.path.basename(csv_file)
        version = basename.split("-")[2]
        df = pd.read_csv(csv_file)
        df["version"] = version
        df = df.rename(columns={"filename": "file"})
        all_coverage.append(df)
        print(f"    {basename}: {len(df)} rows")

    coverage_df = pd.concat(all_coverage, ignore_index=True)
    print(f"  Total coverage rows: {len(coverage_df)}")

    # Load SM data, excluding version 1.0.0 (no coverage data)
    print(f"  Loading SM data from {sm_file}...")
    sm_df = pd.read_excel(sm_file, sheet_name="All SM", header=9)
    sm_df = sm_df.dropna(subset=["Calcite version"])
    original_count = len(sm_df)
    sm_df = sm_df[sm_df["Calcite version"] != "1.0.0"]
    print(f"  SM rows: {original_count} -> {len(sm_df)} (excluded 1.0.0)")

    # Merge on version and file
    cov_subset = coverage_df[["version", "file"] + COVERAGE_FEATURES].copy()
    merged = pd.merge(
        sm_df,
        cov_subset,
        left_on=["Calcite version", "file"],
        right_on=["version", "file"],
        how="left"
    )
    merged = merged.drop(columns=["version"])

    # Check for unmatched rows
    unmatched = merged[COVERAGE_FEATURES[0]].isna().sum()
    if unmatched > 0:
        print(f"  WARNING: {unmatched} rows did not match coverage data")

    # Save
    merged.to_csv(output_file, index=False)
    print(f"  Output: {output_file}")
    print(f"  Shape: {merged.shape}")

    return output_file


def get_top_n_sm_features(results_path: str, n: int = 30) -> list:
    """
    Get top N SM features from clustering results.

    Args:
        results_path: Path to clustering results JSON
        n: Number of top features to return

    Returns:
        List of top N SM feature names
    """
    with open(results_path) as f:
        results = json.load(f)

    feature_relevance = results["feature_relevance"]
    sm_features = [f for f in feature_relevance if f["feature"].startswith("SM_")]
    sorted_features = sorted(sm_features, key=lambda x: x["relevance"], reverse=True)
    return [f["feature"] for f in sorted_features[:n]]


def create_combined_dataset(
    top_n_sm: int = 30,
    results_path: str = "results/calcite/dbscan/results.json",
    output_path: Optional[str] = None,
) -> str:
    """
    Create combined dataset with top SM + effort + coverage features.

    Args:
        top_n_sm: Number of top SM features to include
        results_path: Path to clustering results for feature selection
        output_path: Optional custom output path

    Returns:
        Path to the output CSV file
    """
    output_file = output_path or f"data/Calcite-top{top_n_sm}-sm-cov-effort.csv"
    metadata_cols = METADATA_COLUMNS["calcite"]

    print(f"Creating combined dataset (top-{top_n_sm} SM + effort + coverage)...")

    # Get top SM features
    top_sm = get_top_n_sm_features(results_path, top_n_sm)
    print(f"  Top {len(top_sm)} SM features selected")

    # Load SM data
    print("  Loading SM data...")
    sm_df = pd.read_csv("data/Calcite-SM-only.csv")
    available_sm = [f for f in top_sm if f in sm_df.columns]
    sm_subset = sm_df[metadata_cols + available_sm].copy()
    print(f"    Shape: {sm_subset.shape}")

    # Load coverage data
    print("  Loading coverage data...")
    from glob import glob
    coverage_dfs = []
    for cov_file in sorted(Path("data").glob("Coverage-Calcite-*.csv")):
        version = cov_file.stem.replace("Coverage-Calcite-", "").replace("-filename", "")
        df = pd.read_csv(cov_file)
        df["version"] = version
        coverage_dfs.append(df)

    coverage_df = pd.concat(coverage_dfs, ignore_index=True)
    cov_merge = coverage_df[["filename", "version"] + COVERAGE_FEATURES].copy()
    cov_merge = cov_merge.rename(columns={"filename": "file", "version": "Calcite version"})

    # Load effort data
    print("  Loading effort data...")
    effort_file = RAW_DATA["effort_data"]
    effort_df = pd.read_excel(effort_file, sheet_name="Calcite_Correlation", header=9)
    available_effort = [f for f in EFFORT_FEATURES_26 if f in effort_df.columns]
    effort_merge = effort_df[["Version-ID"] + available_effort].copy()

    # Merge all
    print("  Merging datasets...")
    merged = sm_subset.merge(cov_merge, on=["file", "Calcite version"], how="inner")
    print(f"    After coverage merge: {merged.shape}")
    merged = merged.merge(effort_merge, on="Version-ID", how="inner")
    print(f"    After effort merge: {merged.shape}")

    # Save
    merged.to_csv(output_file, index=False)
    print(f"  Output: {output_file}")

    # Summary
    feature_cols = [c for c in merged.columns if c not in metadata_cols]
    sm_count = len([c for c in feature_cols if c.startswith("SM_")])
    cov_count = len([c for c in feature_cols if c.startswith("COV_")])
    effort_count = len(feature_cols) - sm_count - cov_count
    print(f"  Features: {sm_count} SM + {cov_count} coverage + {effort_count} effort = {len(feature_cols)} total")

    return output_file


def create_top30_sm_dataset(
    results_path: str = "results/calcite/dbscan/results.json",
    output_path: Optional[str] = None,
    v1_1_plus: bool = False,
) -> str:
    """
    Create dataset with only top-30 SM features.

    Args:
        results_path: Path to clustering results for feature selection
        output_path: Optional custom output path
        v1_1_plus: If True, filter to versions 1.1.0+ only

    Returns:
        Path to the output CSV file
    """
    if v1_1_plus:
        default_output = "data/Calcite-top30-sm-only-v1.1+.csv"
    else:
        default_output = "data/Calcite-top30-sm-only.csv"
    output_file = output_path or default_output
    metadata_cols = METADATA_COLUMNS["calcite"]

    print("Creating top-30 SM-only dataset...")

    # Get top 30 SM features
    top_sm = get_top_n_sm_features(results_path, 30)
    print(f"  Selected {len(top_sm)} SM features")

    # Load SM data
    df = pd.read_csv("data/Calcite-SM-only.csv")
    available = [f for f in top_sm if f in df.columns]
    df_filtered = df[metadata_cols + available]

    # Filter to v1.1+ if requested
    if v1_1_plus:
        df_filtered = df_filtered[df_filtered["Calcite version"] != "1.0.0"]
        print(f"  Filtered to v1.1+: {len(df_filtered)} rows")

    # Save
    df_filtered.to_csv(output_file, index=False)
    print(f"  Output: {output_file}")
    print(f"  Shape: {df_filtered.shape}")

    return output_file


def create_effort_only_dataset(output_path: Optional[str] = None) -> str:
    """
    Create dataset with effort + coverage features only (no SM).

    Args:
        output_path: Optional custom output path

    Returns:
        Path to the output CSV file
    """
    output_file = output_path or "data/Calcite-effort-cov-only.csv"
    metadata_cols = METADATA_COLUMNS["calcite"]

    print("Creating effort + coverage only dataset...")

    # Load combined dataset
    df = pd.read_csv("data/Calcite-top30-sm-cov-effort.csv")
    print(f"  Input shape: {df.shape}")

    # Identify effort and coverage columns
    effort_cols = [col for col in df.columns if col.startswith(EFFORT_PREFIXES)]
    coverage_cols = [col for col in df.columns if col.startswith("COV_")]

    print(f"  Effort features: {len(effort_cols)}")
    print(f"  Coverage features: {len(coverage_cols)}")

    # Select columns
    selected_cols = metadata_cols + effort_cols + coverage_cols
    df_filtered = df[selected_cols]

    # Save
    df_filtered.to_csv(output_file, index=False)
    print(f"  Output: {output_file}")
    print(f"  Shape: {df_filtered.shape}")

    return output_file


def run_prepare_action(action: str, dataset: str = "calcite", **kwargs) -> str:
    """
    Run a data preparation action.

    Args:
        action: One of "extract-sm", "merge-coverage", "create-combined",
                "create-top30-sm", "create-effort-only"
        dataset: Dataset to use (for extract-sm)
        **kwargs: Additional arguments passed to the specific function

    Returns:
        Path to the output file
    """
    if action == "extract-sm":
        return extract_sm_features(dataset, **kwargs)
    elif action == "merge-coverage":
        return merge_coverage_data(**kwargs)
    elif action == "create-combined":
        return create_combined_dataset(**kwargs)
    elif action == "create-top30-sm":
        return create_top30_sm_dataset(**kwargs)
    elif action == "create-effort-only":
        return create_effort_only_dataset(**kwargs)
    else:
        raise ValueError(
            f"Unknown action: {action}. "
            f"Use: extract-sm, merge-coverage, create-combined, create-top30-sm, create-effort-only"
        )
