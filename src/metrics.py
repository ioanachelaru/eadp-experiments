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


def compute_cluster_prediction_metrics(
    cluster_labels: np.ndarray, true_labels: np.ndarray
) -> dict:
    """
    Compute prediction metrics based on high-risk cluster membership.

    High-risk clusters = clusters with defect rate > overall defect rate
    Prediction: file in high-risk cluster -> predicted defective

    Args:
        cluster_labels: Cluster assignments from algorithm
        true_labels: Ground truth labels (defective/non-defective)

    Returns:
        Dictionary with precision, recall, f1_score, and inspection_rate
    """
    # Calculate overall defect rate
    overall_defect_rate = true_labels.mean()

    # Identify high-risk clusters (defect rate > overall)
    high_risk_clusters = []
    unique_labels = set(cluster_labels)

    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        mask = cluster_labels == label
        cluster_defect_rate = true_labels[mask].mean()
        if cluster_defect_rate > overall_defect_rate:
            high_risk_clusters.append(label)

    # Create predictions: 1 if in high-risk cluster, 0 otherwise
    # Noise points (-1) are excluded from predictions
    non_noise_mask = cluster_labels >= 0
    predictions = np.isin(cluster_labels, high_risk_clusters).astype(int)

    # Only consider non-noise points for metrics
    pred_valid = predictions[non_noise_mask]
    true_valid = true_labels[non_noise_mask]

    # Calculate metrics
    true_positives = ((pred_valid == 1) & (true_valid == 1)).sum()
    false_positives = ((pred_valid == 1) & (true_valid == 0)).sum()
    false_negatives = ((pred_valid == 0) & (true_valid == 1)).sum()
    total_predicted_positive = (pred_valid == 1).sum()
    total_actual_positive = (true_valid == 1).sum()
    total_valid = len(pred_valid)

    # Precision: TP / (TP + FP)
    precision = true_positives / total_predicted_positive if total_predicted_positive > 0 else 0.0

    # Recall: TP / (TP + FN)
    recall = true_positives / total_actual_positive if total_actual_positive > 0 else 0.0

    # F1-score: harmonic mean of precision and recall
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Inspection rate: % of files in high-risk clusters
    inspection_rate = total_predicted_positive / total_valid if total_valid > 0 else 0.0

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1_score), 4),
        "inspection_rate": round(float(inspection_rate), 4),
        "high_risk_clusters": len(high_risk_clusters),
        "total_clusters": len([l for l in unique_labels if l >= 0]),
        "files_in_high_risk": int(total_predicted_positive),
        "defects_captured": int(true_positives),
        "total_defects": int(total_actual_positive),
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
