"""
Statistical comparison of classification results.

Wilcoxon signed-rank tests for paired CV fold results.
"""

import json
import os

import numpy as np
from scipy.stats import wilcoxon


METRICS = ["precision", "recall", "f1", "roc_auc", "avg_precision"]


def wilcoxon_test(values_a: list[float], values_b: list[float]) -> dict:
    """
    Run a Wilcoxon signed-rank test on paired values.

    Args:
        values_a: Per-fold metric values from experiment A
        values_b: Per-fold metric values from experiment B

    Returns:
        Dict with statistic, p_value, mean_a, mean_b, mean_diff
    """
    a = np.array(values_a)
    b = np.array(values_b)
    diff = a - b

    # If all differences are zero, test is undefined
    if np.all(diff == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff": 0.0,
            "n_pairs": len(a),
            "note": "all differences are zero",
        }

    try:
        stat, p_value = wilcoxon(a, b)
    except ValueError as e:
        # Too few samples or other issues
        return {
            "statistic": None,
            "p_value": None,
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff": float(np.mean(diff)),
            "n_pairs": len(a),
            "note": str(e),
        }

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(diff)),
        "n_pairs": len(a),
    }


def compare_results(result_a_path: str, result_b_path: str, output_path: str = None) -> dict:
    """
    Compare two chronological CV result files using Wilcoxon signed-rank tests.

    Aligns folds by test version and runs the test for each metric.

    Args:
        result_a_path: Path to first result JSON
        result_b_path: Path to second result JSON
        output_path: Optional path to save comparison results

    Returns:
        Dictionary with comparison results
    """
    with open(result_a_path) as f:
        result_a = json.load(f)
    with open(result_b_path) as f:
        result_b = json.load(f)

    # Align folds by test version
    folds_a = {fd["test_version"]: fd for fd in result_a["fold_details"]}
    folds_b = {fd["test_version"]: fd for fd in result_b["fold_details"]}
    common_versions = sorted(set(folds_a.keys()) & set(folds_b.keys()))

    if not common_versions:
        raise ValueError("No common fold versions found between the two results")

    print(f"Comparing {len(common_versions)} common folds: {common_versions}")

    comparison = {
        "result_a": result_a_path,
        "result_b": result_b_path,
        "dataset_a": result_a.get("dataset", "unknown"),
        "dataset_b": result_b.get("dataset", "unknown"),
        "n_common_folds": len(common_versions),
        "common_versions": common_versions,
        "tests": {},
    }

    print(f"\n{'Metric':<18} {'Mean A':>10} {'Mean B':>10} {'Diff':>10} {'p-value':>10} {'Sig?':>6}")
    print("-" * 70)

    for metric in METRICS:
        values_a = [folds_a[v][metric] for v in common_versions]
        values_b = [folds_b[v][metric] for v in common_versions]

        test_result = wilcoxon_test(values_a, values_b)
        comparison["tests"][metric] = test_result

        sig = ""
        if test_result["p_value"] is not None:
            if test_result["p_value"] < 0.01:
                sig = "**"
            elif test_result["p_value"] < 0.05:
                sig = "*"

        p_str = f"{test_result['p_value']:.4f}" if test_result["p_value"] is not None else "N/A"
        print(f"{metric:<18} {test_result['mean_a']:>10.4f} {test_result['mean_b']:>10.4f} "
              f"{test_result['mean_diff']:>+10.4f} {p_str:>10} {sig:>6}")

    print("-" * 70)
    print("* p < 0.05, ** p < 0.01")

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return comparison
