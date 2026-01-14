"""
Feature relevance analysis for clustering.
"""

import numpy as np
import pandas as pd

from .clustering import get_cluster_centers


def compute_feature_relevance(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    feature_names: list[str],
    model=None,
) -> pd.DataFrame:
    """
    Compute feature relevance based on cluster separation.

    For clustering, features that vary most between cluster centers
    are considered most relevant for distinguishing clusters.

    Args:
        X: Scaled feature array
        cluster_labels: Cluster assignments
        feature_names: List of feature names
        model: Fitted clustering model (optional, for K-Means centroids)

    Returns:
        DataFrame with features ranked by relevance
    """
    # Get cluster centers
    if model is not None and hasattr(model, "cluster_centers_"):
        centers = model.cluster_centers_
    else:
        centers = get_cluster_centers(X, cluster_labels)

    if len(centers) < 2:
        # Not enough clusters for comparison
        return pd.DataFrame({
            "feature": feature_names,
            "relevance": [0.0] * len(feature_names),
        })

    # Compute relevance as standard deviation across cluster centers
    # Higher std means the feature varies more across clusters
    relevance = np.std(centers, axis=0)

    # Convert to native Python types for JSON serialization
    relevance_df = pd.DataFrame({
        "feature": [str(f) for f in feature_names],
        "relevance": [float(r) for r in relevance],
    })
    relevance_df = relevance_df.sort_values("relevance", ascending=False)
    relevance_df = relevance_df.reset_index(drop=True)

    return relevance_df


def get_top_features(
    relevance_df: pd.DataFrame, n: int = 20
) -> list[dict]:
    """
    Get top N most relevant features.

    Args:
        relevance_df: DataFrame with feature relevance
        n: Number of top features to return

    Returns:
        List of dicts with feature name and relevance
    """
    top = relevance_df.head(n)
    return top.to_dict("records")


def save_feature_rankings(
    relevance_df: pd.DataFrame,
    file_path: str,
    dataset_name: str,
    algorithm: str,
    n_samples: int,
    n_features: int,
) -> None:
    """
    Save feature rankings to a text file.

    Args:
        relevance_df: DataFrame with feature relevance
        file_path: Path to save the rankings
        dataset_name: Name of the dataset
        algorithm: Clustering algorithm used
        n_samples: Number of samples
        n_features: Number of features
    """
    lines = []
    lines.append("=" * 70)
    lines.append("TOP FEATURES BY CLUSTER RELEVANCE")
    lines.append(f"Dataset: {dataset_name}")
    lines.append(f"Algorithm: {algorithm.upper()}")
    lines.append(f"Samples: {n_samples}, Features: {n_features}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Rank':<6} {'Feature':<50} {'Relevance':>12}")
    lines.append("-" * 70)

    for i, row in relevance_df.iterrows():
        rank = i + 1
        lines.append(f"{rank:<6} {row['feature']:<50} {row['relevance']:>12.6f}")

    lines.append("")
    lines.append("=" * 70)

    with open(file_path, "w") as f:
        f.write("\n".join(lines))
