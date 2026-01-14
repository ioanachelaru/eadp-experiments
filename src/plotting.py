"""
Visualization functions for clustering analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .config import OUTPUT


def plot_clusters_pca(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    title: str,
    save_path: str,
    centers: np.ndarray = None,
) -> None:
    """
    Create PCA visualization of clusters with defect overlay.

    Args:
        X: Scaled feature array
        cluster_labels: Cluster assignments
        true_labels: Ground truth labels
        title: Plot title
        save_path: Path to save the plot
        centers: Cluster centers (optional)
    """
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Get unique cluster labels
    unique_labels = sorted(set(cluster_labels))
    n_clusters = len([l for l in unique_labels if l >= 0])

    # Color map - use special color for noise (-1)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))

    # Left plot: Clusters
    ax1 = axes[0]
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        if label == -1:
            # Noise points
            ax1.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c="gray", marker="x", s=30, alpha=0.5,
                label=f"Noise ({mask.sum()})"
            )
        else:
            color_idx = label % len(colors)
            ax1.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=[colors[color_idx]], marker="o", s=30, alpha=0.6,
                label=f"Cluster {label} ({mask.sum()})"
            )

    # Plot cluster centers if provided
    if centers is not None and len(centers) > 0:
        centers_2d = pca.transform(centers)
        ax1.scatter(
            centers_2d[:, 0], centers_2d[:, 1],
            c="black", marker="X", s=200, edgecolors="white",
            linewidths=2, label="Centroids"
        )

    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax1.set_title("Cluster Assignments")
    ax1.legend(loc="best", fontsize=8)

    # Right plot: Defect overlay
    ax2 = axes[1]
    defective_mask = true_labels == 1
    ax2.scatter(
        X_2d[~defective_mask, 0], X_2d[~defective_mask, 1],
        c="green", marker="o", s=30, alpha=0.4,
        label=f"Non-defective ({(~defective_mask).sum()})"
    )
    ax2.scatter(
        X_2d[defective_mask, 0], X_2d[defective_mask, 1],
        c="red", marker="o", s=30, alpha=0.6,
        label=f"Defective ({defective_mask.sum()})"
    )

    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax2.set_title("Defect Distribution")
    ax2.legend(loc="best", fontsize=8)

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_relevance(
    relevance_df: pd.DataFrame,
    title: str,
    save_path: str,
    n_features: int = None,
) -> None:
    """
    Create bar chart of top features by relevance.

    Args:
        relevance_df: DataFrame with feature and relevance columns
        title: Plot title
        save_path: Path to save the plot
        n_features: Number of top features to display
    """
    n_features = n_features or OUTPUT["top_features_display"]
    top_features = relevance_df.head(n_features)

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_features["relevance"], color="steelblue")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Relevance Score (Centroid Separation)")
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, top_features["relevance"]):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_k_distance(
    X: np.ndarray,
    k: int,
    save_path: str,
    optimal_eps: float = None,
) -> None:
    """
    Plot k-distance graph for DBSCAN parameter selection.

    Args:
        X: Scaled feature array
        k: Number of neighbors (min_samples)
        save_path: Path to save the plot
        optimal_eps: Optimal eps value to mark on the plot
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, k - 1])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(range(len(k_distances)), k_distances, "b-", linewidth=1)
    ax.set_xlabel("Points (sorted by distance)")
    ax.set_ylabel(f"{k}-NN Distance")
    ax.set_title(f"K-Distance Graph (k={k})")

    if optimal_eps is not None:
        ax.axhline(y=optimal_eps, color="r", linestyle="--", linewidth=2,
                   label=f"Optimal eps = {optimal_eps:.3f}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(
    metrics: dict,
    title: str,
    save_path: str,
) -> None:
    """
    Create bar chart comparing clustering metrics.

    Args:
        metrics: Dictionary with metric names and values
        title: Plot title
        save_path: Path to save the plot
    """
    # Filter out None values
    valid_metrics = {k: v for k, v in metrics.items() if v is not None}

    if not valid_metrics:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(valid_metrics.keys())
    values = list(valid_metrics.values())

    bars = ax.bar(names, values, color="steelblue")

    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
