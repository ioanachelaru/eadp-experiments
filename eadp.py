#!/usr/bin/env python3
"""
EADP: Effort-Aware Defect Prediction CLI

Unified command-line interface for defect prediction experiments.

Usage:
    python eadp.py prepare --action extract-sm --dataset calcite
    python eadp.py cluster --dataset calcite --algorithm dbscan
    python eadp.py classify --dataset calcite-effort-cov-only --classifier rf
    python eadp.py compare --dataset calcite --algorithm dbscan
"""

import argparse
import json
import os
import sys
from datetime import datetime

from src.config import DATASETS, CLUSTERING_DEFAULTS, PREPROCESSING, OUTPUT


def cmd_prepare(args):
    """Handle prepare subcommand."""
    from src.prepare import run_prepare_action

    kwargs = {}
    if args.output:
        kwargs["output_path"] = args.output

    if args.action == "create-top30-sm" and args.v11_plus:
        kwargs["v1_1_plus"] = True

    output_file = run_prepare_action(args.action, args.dataset, **kwargs)
    print(f"\nCompleted: {output_file}")
    return 0


def cmd_cluster(args):
    """Handle cluster subcommand."""
    from src.data_utils import load_dataset, preprocess_features
    from src.clustering import run_kmeans, run_dbscan, get_cluster_centers
    from src.metrics import (
        compute_internal_metrics,
        compute_external_metrics,
        compute_cluster_stats,
        compute_cluster_prediction_metrics,
    )
    from src.feature_analysis import compute_feature_relevance, save_feature_rankings
    from src.plotting import (
        plot_clusters_pca,
        plot_feature_relevance,
        plot_k_distance,
        plot_metrics_comparison,
    )

    dataset_name = args.dataset
    algorithm = args.algorithm

    print(f"\n{'='*60}")
    print(f"Clustering Analysis")
    print(f"  Dataset: {dataset_name}")
    print(f"  Algorithm: {algorithm.upper()}")
    print(f"  Remove outliers: {args.no_outliers}")
    print(f"{'='*60}")

    # Load and preprocess data
    print("\nLoading data...")
    df, label_col, feature_name_map = load_dataset(dataset_name)
    print(f"  Loaded {len(df)} samples")

    z_threshold = args.z_threshold or PREPROCESSING["outlier_z_threshold"]
    X_scaled, labels, scaler, feature_names = preprocess_features(
        df, label_col, feature_name_map=feature_name_map,
        remove_outliers=args.no_outliers, z_threshold=z_threshold
    )
    print(f"  Features: {len(feature_names)}")
    print(f"  Defective: {labels.sum()} ({labels.mean()*100:.1f}%)")

    # Run clustering
    print(f"\nRunning {algorithm.upper()} clustering...")
    if algorithm == "kmeans":
        n_clusters = args.k or CLUSTERING_DEFAULTS["kmeans"]["n_clusters"]
        model, cluster_labels = run_kmeans(X_scaled, n_clusters=n_clusters)
        centers = model.cluster_centers_
        algo_info = {"n_clusters": n_clusters}
        print(f"  Clusters: {n_clusters}")

    elif algorithm == "dbscan":
        min_samples = args.min_samples or CLUSTERING_DEFAULTS["dbscan"]["min_samples"]
        model, cluster_labels, algo_info = run_dbscan(
            X_scaled, eps=args.eps, min_samples=min_samples
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
    prediction_metrics = compute_cluster_prediction_metrics(cluster_labels, labels)

    if internal_metrics["silhouette_score"] is not None:
        print(f"  Silhouette Score: {internal_metrics['silhouette_score']:.4f}")
    if external_metrics["v_measure"] is not None:
        print(f"  V-Measure: {external_metrics['v_measure']:.4f}")
        print(f"  Homogeneity: {external_metrics['homogeneity']:.4f}")
        print(f"  Completeness: {external_metrics['completeness']:.4f}")

    print("\nCluster Prediction Metrics (high-risk cluster membership):")
    print(f"  Precision: {prediction_metrics['precision']:.4f}")
    print(f"  Recall: {prediction_metrics['recall']:.4f}")
    print(f"  F1-Score: {prediction_metrics['f1_score']:.4f}")
    print(f"  Inspection Rate: {prediction_metrics['inspection_rate']:.2%}")
    print(f"  High-risk clusters: {prediction_metrics['high_risk_clusters']}/{prediction_metrics['total_clusters']}")
    print(f"  Defects captured: {prediction_metrics['defects_captured']}/{prediction_metrics['total_defects']}")

    # Compute feature relevance
    print("\nComputing feature relevance...")
    relevance_df = compute_feature_relevance(
        X_scaled, cluster_labels, feature_names, model
    )
    print(f"  Top 5 features:")
    for i, row in relevance_df.head(5).iterrows():
        print(f"    {i+1}. {row['feature']}: {row['relevance']:.4f}")

    # Create output directory
    outlier_suffix = "_no_outliers" if args.no_outliers else ""
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
            "remove_outliers": args.no_outliers,
            "z_threshold": z_threshold if args.no_outliers else None,
        },
        "n_samples": int(len(X_scaled)),
        "n_features": len(feature_names),
        "n_defective": int(labels.sum()),
        "defect_rate": round(labels.mean() * 100, 2),
        "internal_metrics": internal_metrics,
        "external_metrics": external_metrics,
        "prediction_metrics": prediction_metrics,
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
    return 0


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

    if "prediction_metrics" in results:
        pm = results["prediction_metrics"]
        lines.append("Cluster Prediction Metrics (high-risk cluster membership):")
        lines.append(f"  precision: {pm['precision']:.4f}")
        lines.append(f"  recall: {pm['recall']:.4f}")
        lines.append(f"  f1_score: {pm['f1_score']:.4f}")
        lines.append(f"  inspection_rate: {pm['inspection_rate']:.2%}")
        lines.append(f"  high_risk_clusters: {pm['high_risk_clusters']}/{pm['total_clusters']}")
        lines.append(f"  defects_captured: {pm['defects_captured']}/{pm['total_defects']}")
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


def cmd_classify(args):
    """Handle classify subcommand."""
    import numpy as np
    from src.data_utils import load_dataset, preprocess_features
    from src.classification import (
        get_classifier,
        run_cross_validation,
        get_feature_importances,
        format_cv_results,
    )

    # Datasets to compare
    COMPARISON_DATASETS = [
        "calcite-top30-sm-only-v1.1+",
        "calcite-effort-cov-only",
        "calcite-top30-sm-cov-effort",
    ]

    if args.compare_all:
        print("\n" + "=" * 70)
        print("SUPERVISED LEARNING COMPARISON: SM vs Effort Features")
        print("=" * 70)

        all_results = {}
        output_dir = os.path.join(OUTPUT["base_dir"], "classification")
        os.makedirs(output_dir, exist_ok=True)

        for dataset_name in COMPARISON_DATASETS:
            results = run_single_classification(
                dataset_name, args.classifier, args.cv_folds
            )
            all_results[dataset_name] = results
            save_classification_results(results, output_dir)

        # Generate comparison report
        report_path = os.path.join(OUTPUT["base_dir"], "supervised_learning_comparison.txt")
        generate_classification_report(all_results, args.classifier, report_path)

        print(f"\n{'='*70}")
        print(f"Comparison report saved to: {report_path}")
        print(f"Individual results saved to: {output_dir}/")

    elif args.dataset:
        results = run_single_classification(
            args.dataset, args.classifier, args.cv_folds
        )
        output_dir = os.path.join(OUTPUT["base_dir"], "classification")
        json_path = save_classification_results(results, output_dir)
        print(f"\nResults saved to: {json_path}")

    else:
        print("Error: Please specify --dataset or --compare-all")
        return 1

    return 0


def run_single_classification(
    dataset_name: str,
    classifier_type: str = 'rf',
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Run classification experiment on a dataset."""
    from src.data_utils import load_dataset, preprocess_features
    from src.classification import (
        get_classifier,
        run_cross_validation,
        get_feature_importances,
        format_cv_results,
    )

    print(f"\n{'='*60}")
    print(f"Classification Experiment")
    print(f"  Dataset: {dataset_name}")
    print(f"  Classifier: {classifier_type.upper()}")
    print(f"  CV Folds: {n_splits}")
    print(f"{'='*60}")

    # Load and preprocess data
    print("\nLoading data...")
    df, label_col, feature_name_map = load_dataset(dataset_name)
    print(f"  Loaded {len(df)} samples")

    X, y, scaler, feature_names = preprocess_features(
        df, label_col, feature_name_map=feature_name_map
    )
    print(f"  Features: {len(feature_names)}")
    print(f"  Defective: {y.sum()} ({y.mean()*100:.1f}%)")

    # Get classifier
    clf = get_classifier(classifier_type, random_state=random_state)
    print(f"\nRunning {n_splits}-fold stratified cross-validation...")

    # Run cross-validation
    cv_results = run_cross_validation(clf, X, y, n_splits=n_splits, random_state=random_state)
    formatted_results = format_cv_results(cv_results)

    # Print results
    print("\nResults (mean +/- std):")
    for metric, values in formatted_results.items():
        print(f"  {metric:15s}: {values['mean']:.4f} +/- {values['std']:.4f}")

    # Fit on full data for feature importances
    print("\nComputing feature importances...")
    clf.fit(X, y)
    feature_importances = get_feature_importances(clf, feature_names)

    print(f"  Top 10 features:")
    for i, (fname, importance) in enumerate(feature_importances[:10]):
        print(f"    {i+1:2d}. {fname}: {importance:.4f}")

    # Prepare results dict
    results = {
        "dataset": dataset_name,
        "description": DATASETS[dataset_name].get("description", ""),
        "classifier": classifier_type,
        "cv_folds": n_splits,
        "random_state": random_state,
        "n_samples": int(len(X)),
        "n_features": len(feature_names),
        "n_defective": int(y.sum()),
        "defect_rate": round(y.mean() * 100, 2),
        "metrics": formatted_results,
        "feature_importances": [
            {"feature": f, "importance": round(imp, 6)}
            for f, imp in feature_importances
        ],
    }

    return results


def save_classification_results(results: dict, output_dir: str) -> str:
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = results["dataset"]
    classifier = results["classifier"]
    json_path = os.path.join(output_dir, f"{dataset_name}_{classifier}_results.json")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return json_path


def generate_classification_report(all_results: dict, classifier_type: str, report_path: str):
    """Generate a comparison report for all datasets."""
    lines = []
    lines.append("=" * 80)
    lines.append("SUPERVISED LEARNING COMPARISON: SM vs Effort Features")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Classifier: {classifier_type.upper()}")
    lines.append("")

    # Dataset summary
    lines.append("-" * 80)
    lines.append("DATASETS")
    lines.append("-" * 80)
    for name, results in all_results.items():
        lines.append(f"\n{name}:")
        lines.append(f"  Description: {results['description']}")
        lines.append(f"  Samples: {results['n_samples']}")
        lines.append(f"  Features: {results['n_features']}")
        lines.append(f"  Defective: {results['n_defective']} ({results['defect_rate']}%)")

    # Metrics comparison table
    lines.append("")
    lines.append("-" * 80)
    lines.append("METRICS COMPARISON (5-fold stratified CV)")
    lines.append("-" * 80)
    lines.append("")

    metrics = ['precision', 'recall', 'f1', 'roc_auc', 'avg_precision']
    metric_display = {
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score',
        'roc_auc': 'ROC-AUC',
        'avg_precision': 'PR-AUC',
    }

    # Table header
    header = f"{'Metric':<15}"
    for name in all_results.keys():
        short_name = name.replace('calcite-', '').replace('-', ' ')[:20]
        header += f" | {short_name:^22}"
    lines.append(header)
    lines.append("-" * len(header))

    # Table rows
    for metric in metrics:
        row = f"{metric_display[metric]:<15}"
        for name, results in all_results.items():
            m = results['metrics'].get(metric, {'mean': 0, 'std': 0})
            row += f" | {m['mean']:.4f} +/- {m['std']:.4f}  "
        lines.append(row)

    # Winner analysis
    lines.append("")
    lines.append("-" * 80)
    lines.append("ANALYSIS")
    lines.append("-" * 80)
    lines.append("")

    # Find best dataset for each metric
    for metric in metrics:
        best_name = None
        best_value = -1
        for name, results in all_results.items():
            m = results['metrics'].get(metric, {'mean': 0})
            if m['mean'] > best_value:
                best_value = m['mean']
                best_name = name

        short_name = best_name.replace('calcite-', '') if best_name else "N/A"
        lines.append(f"Best {metric_display[metric]}: {short_name} ({best_value:.4f})")

    # Feature importances
    lines.append("")
    lines.append("-" * 80)
    lines.append("TOP 10 FEATURES BY DATASET")
    lines.append("-" * 80)

    for name, results in all_results.items():
        lines.append(f"\n{name}:")
        for i, fi in enumerate(results['feature_importances'][:10]):
            lines.append(f"  {i+1:2d}. {fi['feature']}: {fi['importance']:.4f}")

    # Comparison with clustering
    lines.append("")
    lines.append("-" * 80)
    lines.append("COMPARISON: SUPERVISED LEARNING vs CLUSTERING")
    lines.append("-" * 80)
    lines.append("")
    lines.append("Clustering (DBSCAN) results for reference:")
    lines.append("  - Recall: ~5-8%")
    lines.append("  - Precision: ~25-47%")
    lines.append("  - F1-Score: ~8-14%")
    lines.append("")
    lines.append("Supervised learning improvement:")

    sm_results = all_results.get('calcite-top30-sm-only-v1.1+', {})
    combined_results = all_results.get('calcite-top30-sm-cov-effort', {})

    if sm_results and 'metrics' in sm_results:
        sm_recall = sm_results['metrics'].get('recall', {}).get('mean', 0)
        sm_f1 = sm_results['metrics'].get('f1', {}).get('mean', 0)
        lines.append(f"  - SM-only Recall: {sm_recall:.1%} (vs ~6% for clustering)")
        lines.append(f"  - SM-only F1: {sm_f1:.1%} (vs ~10% for clustering)")

    if combined_results and 'metrics' in combined_results:
        comb_recall = combined_results['metrics'].get('recall', {}).get('mean', 0)
        comb_f1 = combined_results['metrics'].get('f1', {}).get('mean', 0)
        lines.append(f"  - Combined Recall: {comb_recall:.1%}")
        lines.append(f"  - Combined F1: {comb_f1:.1%}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("CONCLUSION")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Supervised learning significantly outperforms clustering for defect")
    lines.append("prediction. The class_weight='balanced' parameter addresses the")
    lines.append("class imbalance issue (7.5% defective) that clustering cannot handle.")
    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def cmd_compare(args):
    """Handle compare subcommand."""
    from src.compare import run_comparison

    return run_comparison(
        dataset=args.dataset,
        algorithm=args.algorithm,
        effort_data_path=args.effort_data,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EADP: Effort-Aware Defect Prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Data preparation
  python eadp.py prepare --action extract-sm --dataset calcite
  python eadp.py prepare --action merge-coverage
  python eadp.py prepare --action create-combined

  # Run experiments
  python eadp.py cluster --dataset calcite --algorithm dbscan
  python eadp.py classify --dataset calcite-effort-cov-only --classifier rf
  python eadp.py classify --compare-all

  # Analysis
  python eadp.py compare --dataset calcite --algorithm dbscan
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # =========================================================================
    # prepare subcommand
    # =========================================================================
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Data preparation commands",
        description="Prepare datasets for experiments",
    )
    prepare_parser.add_argument(
        "--action", "-a",
        required=True,
        choices=["extract-sm", "merge-coverage", "create-combined", "create-top30-sm", "create-effort-only"],
        help="Preparation action to perform",
    )
    prepare_parser.add_argument(
        "--dataset", "-d",
        choices=["calcite", "ant-ivy"],
        default="calcite",
        help="Dataset to process (default: calcite)",
    )
    prepare_parser.add_argument(
        "--output", "-o",
        help="Custom output path",
    )
    prepare_parser.add_argument(
        "--v11-plus",
        action="store_true",
        dest="v11_plus",
        help="For create-top30-sm: filter to versions 1.1.0+ only",
    )

    # =========================================================================
    # cluster subcommand
    # =========================================================================
    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Run clustering experiments",
        description="Run clustering analysis on a dataset",
    )
    cluster_parser.add_argument(
        "--dataset", "-d",
        choices=list(DATASETS.keys()),
        required=True,
        help="Dataset to analyze",
    )
    cluster_parser.add_argument(
        "--algorithm", "-a",
        choices=["kmeans", "dbscan"],
        default="dbscan",
        help="Clustering algorithm (default: dbscan)",
    )
    cluster_parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of clusters for K-Means",
    )
    cluster_parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Epsilon for DBSCAN (auto-detected if not specified)",
    )
    cluster_parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="min_samples for DBSCAN",
    )
    cluster_parser.add_argument(
        "--no-outliers",
        action="store_true",
        help="Remove outliers before clustering",
    )
    cluster_parser.add_argument(
        "--z-threshold",
        type=float,
        default=None,
        help="Z-score threshold for outlier removal",
    )

    # =========================================================================
    # classify subcommand
    # =========================================================================
    classify_parser = subparsers.add_parser(
        "classify",
        help="Run classification experiments",
        description="Run supervised learning experiments for defect prediction",
    )
    classify_parser.add_argument(
        "--dataset", "-d",
        choices=list(DATASETS.keys()),
        help="Dataset to analyze",
    )
    classify_parser.add_argument(
        "--classifier", "-c",
        choices=["rf", "lr"],
        default="rf",
        help="Classifier: rf=RandomForest, lr=LogisticRegression (default: rf)",
    )
    classify_parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare all three feature sets (SM, Effort, Combined)",
    )
    classify_parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )

    # =========================================================================
    # compare subcommand
    # =========================================================================
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare clustering features with effort data",
        description="Compare clustering feature rankings with effort-data feature groups",
    )
    compare_parser.add_argument(
        "--dataset", "-d",
        default="ant-ivy",
        help="Dataset name (default: ant-ivy)",
    )
    compare_parser.add_argument(
        "--algorithm", "-a",
        default="dbscan",
        help="Clustering algorithm (default: dbscan)",
    )
    compare_parser.add_argument(
        "--effort-data",
        default=None,
        help="Path to effort_data Excel file",
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Dispatch to appropriate handler
    if args.command == "prepare":
        return cmd_prepare(args)
    elif args.command == "cluster":
        return cmd_cluster(args)
    elif args.command == "classify":
        return cmd_classify(args)
    elif args.command == "compare":
        return cmd_compare(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
