#!/usr/bin/env python3
"""
Main entry point for clustering analysis experiments.

Usage:
    python run_clustering.py --dataset ant-ivy --algorithm dbscan
    python run_clustering.py --dataset calcite --algorithm kmeans --k 3
    python run_clustering.py --dataset ant-ivy --algorithm dbscan --no-outliers
"""

import argparse
import json
import os
import sys

from src.config import DATASETS, CLUSTERING_DEFAULTS, PREPROCESSING, OUTPUT
from src.data_utils import load_dataset, preprocess_features
from src.clustering import run_kmeans, run_dbscan, get_cluster_centers
from src.metrics import compute_internal_metrics, compute_external_metrics, compute_cluster_stats
from src.feature_analysis import compute_feature_relevance, save_feature_rankings
from src.plotting import (
    plot_clusters_pca,
    plot_feature_relevance,
    plot_k_distance,
    plot_metrics_comparison,
)


def run_analysis(
    dataset_name: str,
    algorithm: str,
    n_clusters: int = None,
    eps: float = None,
    min_samples: int = None,
    remove_outliers: bool = False,
    z_threshold: float = None,
) -> dict:
    """
    Run complete clustering analysis on a dataset.

    Args:
        dataset_name: Name of the dataset to analyze
        algorithm: Clustering algorithm ("kmeans" or "dbscan")
        n_clusters: Number of clusters for K-Means
        eps: Epsilon parameter for DBSCAN
        min_samples: min_samples parameter for DBSCAN
        remove_outliers: Whether to remove outliers before clustering
        z_threshold: Z-score threshold for outlier removal

    Returns:
        Dictionary with all results
    """
    print(f"\n{'='*60}")
    print(f"Clustering Analysis")
    print(f"  Dataset: {dataset_name}")
    print(f"  Algorithm: {algorithm.upper()}")
    print(f"  Remove outliers: {remove_outliers}")
    print(f"{'='*60}")

    # Load and preprocess data
    print("\nLoading data...")
    df, label_col, feature_name_map = load_dataset(dataset_name)
    print(f"  Loaded {len(df)} samples")

    z_threshold = z_threshold or PREPROCESSING["outlier_z_threshold"]
    X_scaled, labels, scaler, feature_names = preprocess_features(
        df, label_col, feature_name_map=feature_name_map,
        remove_outliers=remove_outliers, z_threshold=z_threshold
    )
    print(f"  Features: {len(feature_names)}")
    print(f"  Defective: {labels.sum()} ({labels.mean()*100:.1f}%)")

    # Run clustering
    print(f"\nRunning {algorithm.upper()} clustering...")
    if algorithm == "kmeans":
        n_clusters = n_clusters or CLUSTERING_DEFAULTS["kmeans"]["n_clusters"]
        model, cluster_labels = run_kmeans(X_scaled, n_clusters=n_clusters)
        centers = model.cluster_centers_
        algo_info = {"n_clusters": n_clusters}
        print(f"  Clusters: {n_clusters}")

    elif algorithm == "dbscan":
        min_samples = min_samples or CLUSTERING_DEFAULTS["dbscan"]["min_samples"]
        model, cluster_labels, algo_info = run_dbscan(
            X_scaled, eps=eps, min_samples=min_samples
        )
        centers = get_cluster_centers(X_scaled, cluster_labels)
        print(f"  eps: {algo_info['eps']:.4f} (auto: {algo_info['auto_eps']})")
        print(f"  Clusters found: {algo_info['n_clusters']}")
        print(f"  Noise points: {algo_info['n_noise']} ({algo_info['noise_ratio']*100:.1f}%)")

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Compute metrics
    print("\nComputing metrics...")
    internal_metrics = compute_internal_metrics(X_scaled, cluster_labels)
    external_metrics = compute_external_metrics(cluster_labels, labels)
    cluster_stats = compute_cluster_stats(cluster_labels, labels)

    if internal_metrics["silhouette_score"] is not None:
        print(f"  Silhouette Score: {internal_metrics['silhouette_score']:.4f}")
    if external_metrics["v_measure"] is not None:
        print(f"  V-Measure: {external_metrics['v_measure']:.4f}")
        print(f"  Homogeneity: {external_metrics['homogeneity']:.4f}")
        print(f"  Completeness: {external_metrics['completeness']:.4f}")

    # Compute feature relevance
    print("\nComputing feature relevance...")
    relevance_df = compute_feature_relevance(
        X_scaled, cluster_labels, feature_names, model
    )
    print(f"  Top 5 features:")
    for i, row in relevance_df.head(5).iterrows():
        print(f"    {i+1}. {row['feature']}: {row['relevance']:.4f}")

    # Create output directory
    outlier_suffix = "_no_outliers" if remove_outliers else ""
    output_dir = os.path.join(
        OUTPUT["base_dir"], dataset_name, f"{algorithm}{outlier_suffix}"
    )
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_clusters_pca(
        X_scaled, cluster_labels, labels,
        f"{dataset_name} - {algorithm.upper()} Clusters",
        os.path.join(viz_dir, "clusters_pca.png"),
        centers=centers,
    )

    plot_feature_relevance(
        relevance_df,
        f"{dataset_name} - Top Features ({algorithm.upper()})",
        os.path.join(viz_dir, "feature_relevance.png"),
    )

    # Combined metrics for plotting
    all_metrics = {**internal_metrics, **external_metrics}
    plot_metrics_comparison(
        all_metrics,
        f"{dataset_name} - Clustering Metrics ({algorithm.upper()})",
        os.path.join(viz_dir, "metrics_comparison.png"),
    )

    # For DBSCAN, also plot k-distance graph
    if algorithm == "dbscan":
        plot_k_distance(
            X_scaled, algo_info["min_samples"],
            os.path.join(viz_dir, "k_distance.png"),
            optimal_eps=algo_info["eps"],
        )

    # Save results
    print("\nSaving results...")
    results = {
        "dataset": dataset_name,
        "algorithm": algorithm,
        "algorithm_params": algo_info,
        "preprocessing": {
            "remove_outliers": remove_outliers,
            "z_threshold": z_threshold if remove_outliers else None,
        },
        "n_samples": int(len(X_scaled)),
        "n_features": len(feature_names),
        "n_defective": int(labels.sum()),
        "defect_rate": round(labels.mean() * 100, 2),
        "internal_metrics": internal_metrics,
        "external_metrics": external_metrics,
        "cluster_stats": cluster_stats,
        "feature_relevance": relevance_df.to_dict("records"),
    }

    # Save JSON
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results: {json_path}")

    # Save feature rankings
    rankings_path = os.path.join(output_dir, "top_features.txt")
    save_feature_rankings(
        relevance_df, rankings_path, dataset_name, algorithm,
        len(X_scaled), len(feature_names)
    )
    print(f"  Feature rankings: {rankings_path}")

    # Save metrics summary
    summary_path = os.path.join(output_dir, "metrics_summary.txt")
    save_metrics_summary(results, summary_path)
    print(f"  Metrics summary: {summary_path}")

    print(f"\nResults saved to: {output_dir}")
    return results


def save_metrics_summary(results: dict, file_path: str) -> None:
    """Save a human-readable metrics summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("CLUSTERING METRICS SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Dataset: {results['dataset']}")
    lines.append(f"Algorithm: {results['algorithm'].upper()}")
    lines.append(f"Samples: {results['n_samples']}")
    lines.append(f"Features: {results['n_features']}")
    lines.append(f"Defective: {results['n_defective']} ({results['defect_rate']}%)")
    lines.append("")

    lines.append("Algorithm Parameters:")
    for k, v in results["algorithm_params"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("Internal Metrics:")
    for k, v in results["internal_metrics"].items():
        if v is not None:
            lines.append(f"  {k}: {v:.4f}")
    lines.append("")

    lines.append("External Metrics:")
    for k, v in results["external_metrics"].items():
        if v is not None:
            lines.append(f"  {k}: {v:.4f}")
    lines.append("")

    lines.append("Cluster Statistics:")
    for cluster, stats in results["cluster_stats"].items():
        risk_marker = "!!" if stats["risk"] == "HIGH" else "  "
        lines.append(
            f"  {risk_marker} {cluster}: {stats['total']} samples, "
            f"{stats['defective']} defective ({stats['defect_rate']}%) - {stats['risk']} RISK"
        )

    lines.append("")
    lines.append("=" * 60)

    with open(file_path, "w") as f:
        f.write("\n".join(lines))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run clustering analysis on software defect metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_clustering.py --dataset ant-ivy --algorithm dbscan
  python run_clustering.py --dataset calcite --algorithm kmeans --k 3
  python run_clustering.py --dataset ant-ivy --algorithm dbscan --no-outliers
        """,
    )

    parser.add_argument(
        "--dataset", "-d",
        choices=list(DATASETS.keys()),
        required=True,
        help="Dataset to analyze",
    )

    parser.add_argument(
        "--algorithm", "-a",
        choices=["kmeans", "dbscan"],
        default="dbscan",
        help="Clustering algorithm (default: dbscan)",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of clusters for K-Means",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Epsilon for DBSCAN (auto-detected if not specified)",
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="min_samples for DBSCAN",
    )

    parser.add_argument(
        "--no-outliers",
        action="store_true",
        help="Remove outliers before clustering",
    )

    parser.add_argument(
        "--z-threshold",
        type=float,
        default=None,
        help="Z-score threshold for outlier removal",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    results = run_analysis(
        dataset_name=args.dataset,
        algorithm=args.algorithm,
        n_clusters=args.k,
        eps=args.eps,
        min_samples=args.min_samples,
        remove_outliers=args.no_outliers,
        z_threshold=args.z_threshold,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
