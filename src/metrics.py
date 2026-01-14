"""
Clustering evaluation metrics.
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)


def compute_internal_metrics(
    X: np.ndarray, cluster_labels: np.ndarray
) -> dict:
    """
    Compute internal clustering metrics.

    Args:
        X: Scaled feature array
        cluster_labels: Cluster assignments

    Returns:
        Dictionary with silhouette_score and davies_bouldin_index
    """
    # Filter out noise points (label -1) for metric computation
    mask = cluster_labels >= 0
    X_valid = X[mask]
    labels_valid = cluster_labels[mask]

    # Need at least 2 clusters for these metrics
    n_clusters = len(set(labels_valid))
    if n_clusters < 2 or len(X_valid) < 2:
        return {
            "silhouette_score": None,
            "davies_bouldin_index": None,
        }

    return {
        "silhouette_score": float(silhouette_score(X_valid, labels_valid)),
        "davies_bouldin_index": float(davies_bouldin_score(X_valid, labels_valid)),
    }


def compute_external_metrics(
    cluster_labels: np.ndarray, true_labels: np.ndarray
) -> dict:
    """
    Compute external clustering metrics using ground truth labels.

    Args:
        cluster_labels: Cluster assignments from algorithm
        true_labels: Ground truth labels (defective/non-defective)

    Returns:
        Dictionary with homogeneity, completeness, and v_measure
    """
    # Filter out noise points (label -1)
    mask = cluster_labels >= 0
    labels_pred = cluster_labels[mask]
    labels_true = true_labels[mask]

    if len(labels_pred) < 2:
        return {
            "homogeneity": None,
            "completeness": None,
            "v_measure": None,
        }

    return {
        "homogeneity": float(homogeneity_score(labels_true, labels_pred)),
        "completeness": float(completeness_score(labels_true, labels_pred)),
        "v_measure": float(v_measure_score(labels_true, labels_pred)),
    }


def compute_cluster_stats(
    cluster_labels: np.ndarray, true_labels: np.ndarray
) -> dict:
    """
    Compute statistics for each cluster.

    Args:
        cluster_labels: Cluster assignments
        true_labels: Ground truth labels

    Returns:
        Dictionary with cluster statistics
    """
    stats = {}
    unique_labels = sorted(set(cluster_labels))

    # Overall defect rate for comparison
    overall_defect_rate = true_labels.mean() * 100

    for label in unique_labels:
        mask = cluster_labels == label
        total = int(mask.sum())
        defective = int(true_labels[mask].sum())
        defect_rate = (defective / total * 100) if total > 0 else 0.0

        cluster_name = "noise" if label == -1 else f"cluster_{label}"
        stats[cluster_name] = {
            "total": total,
            "defective": defective,
            "defect_rate": round(defect_rate, 2),
            "risk": "HIGH" if defect_rate > overall_defect_rate else "LOW",
        }

    return stats
