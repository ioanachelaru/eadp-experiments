#!/usr/bin/env python3
"""
Supervised learning comparison: SM vs Effort features for defect prediction.

This script trains classifiers on different feature sets to compare their
defect prediction capabilities.

Usage:
    python run_classification.py --dataset calcite-top30-sm-only-v1.1+
    python run_classification.py --dataset calcite-effort-cov-only --classifier lr
    python run_classification.py --compare-all
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

from src.config import DATASETS, OUTPUT
from src.data_utils import load_dataset, preprocess_features
from src.classification import (
    get_classifier,
    run_cross_validation,
    get_feature_importances,
    format_cv_results,
)


# Datasets to compare (must have same samples for fair comparison)
COMPARISON_DATASETS = [
    "calcite-top30-sm-only-v1.1+",
    "calcite-effort-cov-only",
    "calcite-top30-sm-cov-effort",
]


def run_classification(
    dataset_name: str,
    classifier_type: str = 'rf',
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Run classification experiment on a dataset.

    Args:
        dataset_name: Name of the dataset
        classifier_type: 'rf' for Random Forest, 'lr' for Logistic Regression
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        Dictionary with all results
    """
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


def save_results(results: dict, output_dir: str) -> str:
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = results["dataset"]
    classifier = results["classifier"]
    json_path = os.path.join(output_dir, f"{dataset_name}_{classifier}_results.json")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return json_path


def compare_all_datasets(classifier_type: str = 'rf') -> dict:
    """
    Run classification on all comparison datasets and generate report.

    Args:
        classifier_type: Classifier to use

    Returns:
        Dictionary with all comparison results
    """
    print("\n" + "=" * 70)
    print("SUPERVISED LEARNING COMPARISON: SM vs Effort Features")
    print("=" * 70)

    all_results = {}
    output_dir = os.path.join(OUTPUT["base_dir"], "classification")
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name in COMPARISON_DATASETS:
        results = run_classification(dataset_name, classifier_type=classifier_type)
        all_results[dataset_name] = results
        save_results(results, output_dir)

    # Generate comparison report
    report_path = os.path.join(OUTPUT["base_dir"], "supervised_learning_comparison.txt")
    generate_comparison_report(all_results, classifier_type, report_path)

    print(f"\n{'='*70}")
    print(f"Comparison report saved to: {report_path}")
    print(f"Individual results saved to: {output_dir}/")

    return all_results


def generate_comparison_report(all_results: dict, classifier_type: str, report_path: str):
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

    # Get SM-only and combined results for comparison
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run supervised learning experiments for defect prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_classification.py --dataset calcite-top30-sm-only-v1.1+
  python run_classification.py --dataset calcite-effort-cov-only --classifier lr
  python run_classification.py --compare-all
  python run_classification.py --compare-all --classifier lr
        """,
    )

    parser.add_argument(
        "--dataset", "-d",
        choices=list(DATASETS.keys()),
        help="Dataset to analyze",
    )

    parser.add_argument(
        "--classifier", "-c",
        choices=["rf", "lr"],
        default="rf",
        help="Classifier: rf=RandomForest, lr=LogisticRegression (default: rf)",
    )

    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare all three feature sets (SM, Effort, Combined)",
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.compare_all:
        compare_all_datasets(classifier_type=args.classifier)
    elif args.dataset:
        results = run_classification(
            dataset_name=args.dataset,
            classifier_type=args.classifier,
            n_splits=args.cv_folds,
        )
        output_dir = os.path.join(OUTPUT["base_dir"], "classification")
        json_path = save_results(results, output_dir)
        print(f"\nResults saved to: {json_path}")
    else:
        print("Error: Please specify --dataset or --compare-all")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
