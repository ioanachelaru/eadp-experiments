#!/usr/bin/env python3
"""
Compare clustering feature rankings with effort_data feature groups.

Usage:
    python compare_features.py --dataset ant-ivy --algorithm dbscan
"""

import argparse
import json
import os
from collections import defaultdict

import pandas as pd


EFFORT_DATA_FILE = "effort_data/All Calcite 1.0.0-1.15.0 effort-related metrics.xlsx"

# Sheets and their configurations
EFFORT_SHEETS = {
    "26_common": {
        "sheet": "26 Metrics (Ant int Calcite)",
        "description": "26 metrics common to Ant & Calcite (corr>=0.1)",
    },
    "ant_all": {
        "sheet": "Ant_All",
        "header_row": 9,
        "exclude_cols": ["Ant version", "ID", "file", "Bug"],
        "description": "149 Ant effort-data features (all CHANGE, HASSAN, ISSUE, MOSER, PMD)",
    },
    "calcite_all": {
        "sheet": "Calcite_All",
        "header_row": 9,
        "exclude_cols": ["Calcite version", "ID", "file", "Bug", "Version-ID"],
        "description": "170 Calcite effort-data features (all CHANGE, HASSAN, ISSUE, MOSER, PMD)",
    },
}


def load_clustering_results(results_path: str) -> tuple[list[dict], dict]:
    """
    Load feature relevance rankings from clustering results.

    Returns:
        Tuple of (feature_rankings, metadata)
    """
    with open(results_path, "r") as f:
        results = json.load(f)

    feature_relevance = results.get("feature_relevance", [])
    metadata = {
        "dataset": results.get("dataset"),
        "algorithm": results.get("algorithm"),
        "n_samples": results.get("n_samples"),
        "n_features": results.get("n_features"),
    }

    return feature_relevance, metadata


def load_effort_data_features(xlsx_path: str, dataset: str) -> dict[str, list[str]]:
    """
    Load feature lists from effort_data Excel sheets.

    Args:
        xlsx_path: Path to effort_data Excel file
        dataset: Dataset name ("ant-ivy" or "calcite")

    Returns:
        Dictionary mapping group name to list of feature names
    """
    xl = pd.ExcelFile(xlsx_path)
    features = {}

    # Load 26 common metrics
    df_26 = pd.read_excel(xl, sheet_name="26 Metrics (Ant int Calcite)", header=None)
    common_features = []
    for col in df_26.columns:
        for val in df_26[col]:
            if pd.notna(val) and "Metrics with corr" not in str(val):
                common_features.append(str(val))
    features["26_common"] = common_features

    # Load dataset-specific features
    if dataset == "ant-ivy":
        config = EFFORT_SHEETS["ant_all"]
        df = pd.read_excel(xl, sheet_name=config["sheet"], header=config["header_row"])
        exclude = set(config["exclude_cols"])
        dataset_features = [
            str(c) for c in df.columns
            if "Unnamed" not in str(c) and str(c) not in exclude
        ]
        features["ant_all"] = dataset_features
    elif dataset == "calcite":
        config = EFFORT_SHEETS["calcite_all"]
        df = pd.read_excel(xl, sheet_name=config["sheet"], header=config["header_row"])
        exclude = set(config["exclude_cols"])
        dataset_features = [
            str(c) for c in df.columns
            if "Unnamed" not in str(c) and str(c) not in exclude
        ]
        features["calcite_all"] = dataset_features

    return features


def get_feature_category(feature_name: str) -> str:
    """Extract category prefix from feature name."""
    prefixes = ["PMD_", "HASSAN_", "MOSER_", "ISSUE_", "CHANGE_TYPE_", "SM_", "AST_"]
    for prefix in prefixes:
        if feature_name.startswith(prefix):
            return prefix.rstrip("_")
    return "OTHER"


def find_feature_ranks(
    clustering_features: list[dict],
    effort_features: list[str],
) -> dict[str, int | None]:
    """
    Find the rank of each effort_data feature in clustering results.

    Returns:
        Dictionary mapping feature name to rank (1-indexed) or None if not found
    """
    # Build name -> rank mapping
    name_to_rank = {}
    for i, item in enumerate(clustering_features):
        name = item.get("feature", "")
        name_to_rank[name] = i + 1

    # Find ranks for effort features
    ranks = {}
    for feature in effort_features:
        ranks[feature] = name_to_rank.get(feature)

    return ranks


def compute_statistics(ranks: dict[str, int | None], total_features: int) -> dict:
    """
    Compute comparison statistics.

    Returns:
        Dictionary with statistics
    """
    found_ranks = {k: v for k, v in ranks.items() if v is not None}
    not_found = [k for k, v in ranks.items() if v is None]

    stats = {
        "total_effort_features": len(ranks),
        "found": len(found_ranks),
        "not_found": len(not_found),
        "not_found_features": not_found,
        "found_pct": len(found_ranks) / len(ranks) * 100 if ranks else 0,
    }

    if found_ranks:
        rank_values = list(found_ranks.values())
        stats["min_rank"] = min(rank_values)
        stats["max_rank"] = max(rank_values)
        stats["median_rank"] = sorted(rank_values)[len(rank_values) // 2]
        stats["avg_rank"] = sum(rank_values) / len(rank_values)

        # Count features in top N
        for n in [50, 100, 500, 1000]:
            in_top_n = sum(1 for r in rank_values if r <= n)
            stats[f"in_top_{n}"] = in_top_n
            stats[f"in_top_{n}_pct"] = in_top_n / len(ranks) * 100

        # Category breakdown
        category_ranks = defaultdict(list)
        for feature, rank in found_ranks.items():
            cat = get_feature_category(feature)
            category_ranks[cat].append(rank)

        stats["by_category"] = {}
        for cat, cat_ranks in sorted(category_ranks.items()):
            stats["by_category"][cat] = {
                "count": len(cat_ranks),
                "avg_rank": sum(cat_ranks) / len(cat_ranks),
                "min_rank": min(cat_ranks),
                "in_top_100": sum(1 for r in cat_ranks if r <= 100),
            }
    else:
        stats["min_rank"] = None
        stats["max_rank"] = None
        stats["median_rank"] = None
        stats["avg_rank"] = None
        for n in [50, 100, 500, 1000]:
            stats[f"in_top_{n}"] = 0
            stats[f"in_top_{n}_pct"] = 0.0
        stats["by_category"] = {}

    return stats


def generate_report(
    metadata: dict,
    effort_features: dict[str, list[str]],
    all_ranks: dict[str, dict],
    all_stats: dict[str, dict],
    output_path: str,
) -> None:
    """Generate and save comparison report."""
    lines = []
    lines.append("=" * 70)
    lines.append("FEATURE COMPARISON REPORT: Clustering vs Effort-Data")
    lines.append("=" * 70)
    lines.append(f"Dataset: {metadata['dataset']}")
    lines.append(f"Algorithm: {metadata['algorithm'].upper()}")
    lines.append(f"Total clustering features: {metadata['n_features']}")
    lines.append("")

    for group_name, features in effort_features.items():
        stats = all_stats[group_name]
        ranks = all_ranks[group_name]

        lines.append("-" * 70)
        lines.append(f"FEATURE GROUP: {group_name}")
        lines.append(f"Description: {EFFORT_SHEETS.get(group_name, {}).get('description', group_name)}")
        lines.append(f"Total features: {len(features)}")
        lines.append("-" * 70)
        lines.append("")

        lines.append("OVERLAP SUMMARY")
        lines.append(f"  Features found in clustering: {stats['found']}/{len(features)} ({stats['found_pct']:.1f}%)")
        if stats["not_found_features"]:
            lines.append(f"  Not found: {', '.join(stats['not_found_features'][:5])}" +
                        (f"... (+{len(stats['not_found_features'])-5} more)" if len(stats['not_found_features']) > 5 else ""))
        lines.append("")

        if stats["found"] > 0:
            lines.append("RANK DISTRIBUTION")
            lines.append(f"  Best rank: {stats['min_rank']}")
            lines.append(f"  Worst rank: {stats['max_rank']}")
            lines.append(f"  Median rank: {stats['median_rank']}")
            lines.append(f"  Average rank: {stats['avg_rank']:.1f}")
            lines.append("")

            lines.append("TOP N COVERAGE")
            for n in [50, 100, 500, 1000]:
                lines.append(f"  In top {n:4d}: {stats[f'in_top_{n}']:3d} ({stats[f'in_top_{n}_pct']:5.1f}%)")
            lines.append("")

            if stats["by_category"]:
                lines.append("CATEGORY BREAKDOWN")
                for cat, cat_stats in sorted(stats["by_category"].items()):
                    lines.append(
                        f"  {cat:15s}: {cat_stats['count']:2d} features, "
                        f"avg rank {cat_stats['avg_rank']:6.1f}, "
                        f"best {cat_stats['min_rank']:4d}, "
                        f"{cat_stats['in_top_100']} in top 100"
                    )
                lines.append("")

            lines.append("TOP EFFORT-DATA FEATURES BY CLUSTERING RANK")
            found_ranks = [(f, r) for f, r in ranks.items() if r is not None]
            found_ranks.sort(key=lambda x: x[1])
            for i, (feature, rank) in enumerate(found_ranks[:20], 1):
                cat = get_feature_category(feature)
                lines.append(f"  {i:2d}. {feature:50s} rank {rank:4d} [{cat}]")
            lines.append("")

    lines.append("=" * 70)

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    # Also print to console
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Compare clustering feature rankings with effort_data feature groups"
    )
    parser.add_argument(
        "--dataset", "-d",
        default="ant-ivy",
        help="Dataset name (default: ant-ivy)",
    )
    parser.add_argument(
        "--algorithm", "-a",
        default="dbscan",
        help="Clustering algorithm (default: dbscan)",
    )
    parser.add_argument(
        "--effort-data",
        default=EFFORT_DATA_FILE,
        help=f"Path to effort_data Excel file (default: {EFFORT_DATA_FILE})",
    )

    args = parser.parse_args()

    # Build paths
    results_path = os.path.join("results", args.dataset, args.algorithm, "results.json")
    output_path = os.path.join("results", args.dataset, args.algorithm, "comparison_report.txt")

    # Check files exist
    if not os.path.exists(results_path):
        print(f"Error: Results file not found: {results_path}")
        return 1

    if not os.path.exists(args.effort_data):
        print(f"Error: Effort data file not found: {args.effort_data}")
        return 1

    # Load data
    print(f"Loading clustering results from {results_path}...")
    clustering_features, metadata = load_clustering_results(results_path)

    print(f"Loading effort_data features from {args.effort_data}...")
    effort_features = load_effort_data_features(args.effort_data, args.dataset)

    # Compute comparisons for each group
    all_ranks = {}
    all_stats = {}

    for group_name, features in effort_features.items():
        print(f"Comparing {group_name} ({len(features)} features)...")
        ranks = find_feature_ranks(clustering_features, features)
        stats = compute_statistics(ranks, metadata["n_features"])
        all_ranks[group_name] = ranks
        all_stats[group_name] = stats

    # Generate report
    print(f"\nGenerating report to {output_path}...\n")
    generate_report(metadata, effort_features, all_ranks, all_stats, output_path)

    print(f"\nReport saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
