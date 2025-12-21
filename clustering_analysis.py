"""
Clustering Analysis for Software Defect Metrics

This script performs K-Means clustering on software metrics from Ant and Calcite
projects to identify which metrics are relevant to software defects.

Usage:
    python clustering_analysis.py              # With outliers (default)
    python clustering_analysis.py --no-outliers  # Remove outliers first
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "data_file": "data/All Calcite 1.0.0-1.15.0 effort-related metrics.xlsx",
    "sheets": ["Ant_All", "Calcite_All"],
    "label_columns": ["Bug"],
    "n_clusters": 2,
    "random_state": 42,
    "output_dir": "results",
    "top_features": 170,  # Number of top features to display
    # Outlier removal settings (modified by --no-outliers flag)
    "remove_outliers": False,
    "outlier_z_threshold": 3.0,
}


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================


def load_sheet_data(file_path: str, sheet_name: str) -> tuple[pd.DataFrame, str]:
    """
    Load data from an Excel sheet.

    Args:
        file_path: Path to the Excel file
        sheet_name: Name of the sheet to load

    Returns:
        Tuple of (DataFrame, label_column_name)
    """
    # Read with header at row 9 (0-indexed) as per the Excel structure
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=9)

    # Drop rows where all values are NaN (empty rows at the beginning)
    df = df.dropna(how="all")

    # Find the label column
    label_col = None
    for col in CONFIG["label_columns"]:
        if col in df.columns:
            label_col = col
            break
        # Case-insensitive search
        for df_col in df.columns:
            if isinstance(df_col, str) and df_col.lower() == col.lower():
                label_col = df_col
                break
        if label_col:
            break

    if label_col is None:
        raise ValueError(
            f"Could not find label column. Looked for: {CONFIG['label_columns']}"
        )

    return df, label_col


def preprocess_features(
    df: pd.DataFrame, label_col: str
) -> tuple[np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """
    Preprocess features for clustering.

    Args:
        df: Input DataFrame
        label_col: Name of the label column

    Returns:
        Tuple of (scaled_features, labels, scaler, feature_names)
    """
    # Columns to exclude (non-feature columns)
    exclude_patterns = ["ID", "file", "version", "Unnamed"]

    # Extract labels (convert to binary: 0 = no bugs, 1 = has bugs)
    labels = (df[label_col] > 0).astype(int).values

    # Select numeric columns only, excluding the label column and non-feature columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = []
    for col in numeric_cols:
        col_str = str(col)
        if col == label_col:
            continue
        if any(pattern.lower() in col_str.lower() for pattern in exclude_patterns):
            continue
        feature_cols.append(col)

    # Extract features
    X = df[feature_cols].copy()

    # Handle missing values by dropping rows with NaN
    mask = ~X.isna().any(axis=1)
    X = X[mask]
    labels = labels[mask]

    # Store feature names (convert to string for output)
    feature_names = [str(col) for col in feature_cols]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Remove outliers if configured
    if CONFIG["remove_outliers"]:
        outlier_mask = remove_outliers(X_scaled, CONFIG["outlier_z_threshold"])
        n_removed = len(X_scaled) - outlier_mask.sum()
        X_scaled = X_scaled[outlier_mask]
        labels = labels[outlier_mask]
        print(f"    Removed {n_removed} outliers (Z > {CONFIG['outlier_z_threshold']})")

    return X_scaled, labels, scaler, feature_names


def remove_outliers(X_scaled: np.ndarray, z_threshold: float = 3.0) -> np.ndarray:
    """
    Identify samples to keep (those without extreme outlier values).

    Args:
        X_scaled: Standardized feature matrix
        z_threshold: Z-score threshold for outlier detection

    Returns:
        Boolean mask of samples to keep (True = keep, False = outlier)
    """
    # A sample is an outlier if ANY feature has |z-score| > threshold
    z_scores = np.abs(X_scaled)
    mask = (z_scores <= z_threshold).all(axis=1)
    return mask


# =============================================================================
# Clustering
# =============================================================================


def perform_clustering(
    X_scaled: np.ndarray, n_clusters: int, random_state: int
) -> tuple[KMeans, np.ndarray]:
    """
    Perform K-Means clustering.

    Args:
        X_scaled: Scaled feature matrix
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (fitted KMeans model, cluster labels)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    return kmeans, cluster_labels


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_internal_metrics(X_scaled: np.ndarray, cluster_labels: np.ndarray) -> dict:
    """
    Compute internal clustering metrics.

    Args:
        X_scaled: Scaled feature matrix
        cluster_labels: Cluster assignments

    Returns:
        Dictionary with silhouette_score and davies_bouldin_index
    """
    return {
        "silhouette_score": float(silhouette_score(X_scaled, cluster_labels)),
        "davies_bouldin_index": float(davies_bouldin_score(X_scaled, cluster_labels)),
    }


def compute_external_metrics(
    cluster_labels: np.ndarray, true_labels: np.ndarray
) -> dict:
    """
    Compute external clustering metrics using ground truth labels.

    Args:
        cluster_labels: Cluster assignments
        true_labels: Ground truth labels (defective/non-defective)

    Returns:
        Dictionary with homogeneity, completeness, and v_measure
    """
    return {
        "homogeneity": float(homogeneity_score(true_labels, cluster_labels)),
        "completeness": float(completeness_score(true_labels, cluster_labels)),
        "v_measure": float(v_measure_score(true_labels, cluster_labels)),
    }


def compute_feature_relevance(
    kmeans: KMeans, feature_names: list[str]
) -> pd.DataFrame:
    """
    Compute feature relevance based on cluster centroid separation.

    Features with larger differences between cluster centroids are considered
    more relevant for distinguishing clusters.

    For k=2: Uses absolute difference between centroids.
    For k>2: Uses standard deviation of centroids across clusters.

    Args:
        kmeans: Fitted KMeans model
        feature_names: List of feature names

    Returns:
        DataFrame with features ranked by relevance
    """
    centroids = kmeans.cluster_centers_
    n_clusters = centroids.shape[0]

    if n_clusters == 2:
        # For k=2: absolute difference between centroids
        relevance = np.abs(centroids[0] - centroids[1])
    else:
        # For k>2: standard deviation of each feature across cluster centroids
        # Higher std means the feature varies more across clusters
        relevance = np.std(centroids, axis=0)

    relevance_df = pd.DataFrame(
        {"feature": feature_names, "relevance": relevance}
    )
    relevance_df = relevance_df.sort_values("relevance", ascending=False)
    relevance_df = relevance_df.reset_index(drop=True)

    return relevance_df


# =============================================================================
# Visualization
# =============================================================================


def plot_clusters_pca(
    X_scaled: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    kmeans: KMeans,
    cluster_sizes: dict,
    title: str,
    save_path: str,
) -> None:
    """
    Create a 2D PCA visualization of clusters.

    Args:
        X_scaled: Scaled feature matrix
        cluster_labels: Cluster assignments
        true_labels: Ground truth labels
        kmeans: Fitted KMeans model (for centroids)
        cluster_sizes: Dictionary with cluster size counts
        title: Plot title
        save_path: Path to save the figure
    """
    # Reduce to 2D using PCA
    pca = PCA(n_components=2, random_state=CONFIG["random_state"])
    X_pca = pca.fit_transform(X_scaled)

    # Transform centroids to PCA space
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    # Determine minority cluster for highlighting
    sizes = list(cluster_sizes.values())
    minority_cluster = 0 if sizes[0] < sizes[1] else 1
    minority_ratio = min(sizes) / sum(sizes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Create point sizes - larger for minority cluster
    point_sizes = np.where(cluster_labels == minority_cluster, 50, 10)

    # Plot 1: Cluster assignments
    scatter1 = axes[0].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=cluster_labels,
        s=point_sizes,
        cmap="viridis",
        alpha=0.6,
        edgecolors="none",
    )
    # Plot centroids as stars
    axes[0].scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        c="red",
        marker="*",
        s=300,
        edgecolors="black",
        linewidths=1,
        label="Centroids",
        zorder=10,
    )
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")

    # Add cluster size annotation
    cluster_info = " | ".join([f"C{i}: {v}" for i, v in enumerate(sizes)])
    axes[0].set_title(f"K-Means Cluster Assignments\n({cluster_info})")
    axes[0].legend(loc="upper right")
    plt.colorbar(scatter1, ax=axes[0], label="Cluster")

    # Plot 2: True labels (defective/non-defective)
    n_defective = int(true_labels.sum())
    n_non_defective = len(true_labels) - n_defective
    scatter2 = axes[1].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=true_labels,
        cmap="coolwarm",
        alpha=0.6,
        edgecolors="none",
    )
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    axes[1].set_title(f"Actual Defect Labels\n(Non-def: {n_non_defective} | Def: {n_defective})")
    plt.colorbar(scatter2, ax=axes[1], label="Defective")

    # Add warning if highly imbalanced
    if minority_ratio < 0.1:
        fig.text(
            0.5, 0.02,
            f"WARNING: Highly imbalanced clusters (minority: {minority_ratio:.1%})",
            ha="center",
            fontsize=11,
            color="red",
            fontweight="bold",
        )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(
    internal: dict, external: dict, title: str, save_path: str
) -> None:
    """
    Create a bar chart comparing clustering metrics.

    Args:
        internal: Internal metrics dictionary
        external: External metrics dictionary
        title: Plot title
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Internal metrics
    internal_names = list(internal.keys())
    internal_values = list(internal.values())
    colors_internal = ["steelblue", "coral"]
    bars1 = axes[0].bar(internal_names, internal_values, color=colors_internal)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Internal Metrics")
    axes[0].set_ylim(0, max(internal_values) * 1.2)
    for bar, val in zip(bars1, internal_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # External metrics
    external_names = list(external.keys())
    external_values = list(external.values())
    colors_external = ["forestgreen", "mediumpurple", "goldenrod"]
    bars2 = axes[1].bar(external_names, external_values, color=colors_external)
    axes[1].set_ylabel("Score")
    axes[1].set_title("External Metrics (vs Defect Labels)")
    axes[1].set_ylim(0, 1.0)
    for bar, val in zip(bars2, external_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_relevance(
    relevance_df: pd.DataFrame, title: str, save_path: str, top_n: int = 15
) -> None:
    """
    Create a horizontal bar chart of top features by relevance.

    Args:
        relevance_df: DataFrame with feature relevance scores
        title: Plot title
        save_path: Path to save the figure
        top_n: Number of top features to display
    """
    top_features = relevance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(top_features))
    bars = ax.barh(
        y_pos, top_features["relevance"].values, color="teal", edgecolor="none"
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Centroid Separation (Standardized)")
    ax.set_title(title)

    # Add value labels
    for bar, val in zip(bars, top_features["relevance"].values):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Results Output
# =============================================================================


def save_results_json(results: dict, json_path: str) -> None:
    """Save results to a JSON file."""
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)


def generate_text_report(all_results: list[dict]) -> str:
    """
    Generate a formatted text report.

    Args:
        all_results: List of result dictionaries for each dataset

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("CLUSTERING ANALYSIS REPORT")
    lines.append("Software Defect Metrics - K-Means Clustering (k=2)")
    lines.append("=" * 70)
    lines.append("")

    for result in all_results:
        lines.append(f"{'=' * 70}")
        lines.append(f"Dataset: {result['dataset']}")
        lines.append(f"{'=' * 70}")
        lines.append(f"Samples: {result['n_samples']}")
        lines.append(f"Features: {result['n_features']}")
        lines.append(f"Defective instances: {result['n_defective']} ({result['defect_ratio']:.1%})")
        lines.append("")

        # Cluster distribution
        lines.append("CLUSTER DISTRIBUTION")
        lines.append("-" * 40)
        cluster_sizes = result.get("cluster_sizes", {})
        total = sum(cluster_sizes.values())
        for cluster_name, count in cluster_sizes.items():
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"  {cluster_name}: {count} samples ({pct:.1f}%)")

        # Check for imbalance
        sizes = list(cluster_sizes.values())
        if len(sizes) >= 2:
            minority_ratio = min(sizes) / sum(sizes)
            if minority_ratio < 0.1:
                lines.append("")
                lines.append("  !! WARNING: Highly imbalanced clusters detected.")
                lines.append("  The high Silhouette score reflects outlier separation,")
                lines.append("  not necessarily meaningful cluster structure.")
        lines.append("")

        lines.append("INTERNAL METRICS (Clustering Quality)")
        lines.append("-" * 40)
        im = result["internal_metrics"]
        lines.append(f"  Silhouette Score:      {im['silhouette_score']:.4f}")
        lines.append(f"    (Range: -1 to 1, higher is better)")
        lines.append(f"  Davies-Bouldin Index:  {im['davies_bouldin_index']:.4f}")
        lines.append(f"    (Lower is better)")
        lines.append("")

        lines.append("EXTERNAL METRICS (Alignment with Defect Labels)")
        lines.append("-" * 40)
        em = result["external_metrics"]
        lines.append(f"  Homogeneity:   {em['homogeneity']:.4f}")
        lines.append(f"    (Each cluster contains only one class)")
        lines.append(f"  Completeness:  {em['completeness']:.4f}")
        lines.append(f"    (All members of a class are in the same cluster)")
        lines.append(f"  V-Measure:     {em['v_measure']:.4f}")
        lines.append(f"    (Harmonic mean of homogeneity and completeness)")
        lines.append("")

        lines.append(f"TOP {CONFIG['top_features']} RELEVANT FEATURES (by centroid separation)")
        lines.append("-" * 40)
        for i, feat in enumerate(result["feature_relevance"][: CONFIG["top_features"]]):
            lines.append(f"  {i + 1:2d}. {feat['feature']:30s} {feat['relevance']:.4f}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# Main Analysis
# =============================================================================


def analyze_dataset(file_path: str, sheet_name: str, output_dir: str) -> dict:
    """
    Perform complete clustering analysis on a dataset.

    Args:
        file_path: Path to the Excel file
        sheet_name: Name of the sheet to analyze
        output_dir: Directory for output files

    Returns:
        Dictionary with all results
    """
    print(f"\nAnalyzing {sheet_name}...")

    # Load data
    df, label_col = load_sheet_data(file_path, sheet_name)
    print(f"  Loaded {len(df)} rows, label column: '{label_col}'")

    # Preprocess
    X_scaled, labels, scaler, feature_names = preprocess_features(df, label_col)
    print(f"  After preprocessing: {len(X_scaled)} samples, {len(feature_names)} features")

    # Perform clustering
    kmeans, cluster_labels = perform_clustering(
        X_scaled, CONFIG["n_clusters"], CONFIG["random_state"]
    )
    print(f"  Clustering complete")

    # Compute metrics
    internal_metrics = compute_internal_metrics(X_scaled, cluster_labels)
    external_metrics = compute_external_metrics(cluster_labels, labels)
    relevance_df = compute_feature_relevance(kmeans, feature_names)

    # Compute cluster sizes
    cluster_sizes = {}
    for i in range(CONFIG["n_clusters"]):
        count = int(np.sum(cluster_labels == i))
        cluster_sizes[f"cluster_{i}"] = count
    print(f"  Cluster sizes: {cluster_sizes}")

    # Create visualizations directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Generate visualizations
    dataset_name = sheet_name.replace("_All", "").lower()

    plot_clusters_pca(
        X_scaled,
        cluster_labels,
        labels,
        kmeans,
        cluster_sizes,
        f"{sheet_name} - Cluster vs Defect Labels (PCA)",
        os.path.join(viz_dir, f"{dataset_name}_clusters.png"),
    )

    plot_metrics_comparison(
        internal_metrics,
        external_metrics,
        f"{sheet_name} - Clustering Metrics",
        os.path.join(viz_dir, f"{dataset_name}_metrics_comparison.png"),
    )

    plot_feature_relevance(
        relevance_df,
        f"{sheet_name} - Top Features by Relevance",
        os.path.join(viz_dir, f"{dataset_name}_feature_relevance.png"),
        CONFIG["top_features"],
    )

    print(f"  Visualizations saved")

    # Prepare results dictionary
    results = {
        "dataset": sheet_name,
        "n_samples": int(len(X_scaled)),
        "n_features": int(len(feature_names)),
        "n_defective": int(labels.sum()),
        "defect_ratio": float(labels.mean()),
        "cluster_sizes": cluster_sizes,
        "internal_metrics": internal_metrics,
        "external_metrics": external_metrics,
        "feature_relevance": relevance_df.to_dict("records"),
    }

    # Save individual JSON results
    json_path = os.path.join(output_dir, f"{dataset_name}_results.json")
    save_results_json(results, json_path)
    print(f"  Results saved to {json_path}")

    return results


# =============================================================================
# Multi-K Analysis
# =============================================================================


def analyze_dataset_for_k(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    k: int,
    dataset_name: str,
    output_dir: str,
) -> dict:
    """
    Analyze a dataset with a specific k value.

    Returns dict with metrics and cluster info.
    """
    # Perform clustering
    kmeans, cluster_labels = perform_clustering(X_scaled, k, CONFIG["random_state"])

    # Compute metrics
    internal_metrics = compute_internal_metrics(X_scaled, cluster_labels)
    external_metrics = compute_external_metrics(cluster_labels, labels)

    # Get inertia (for elbow method)
    inertia = float(kmeans.inertia_)

    # Compute feature relevance
    relevance_df = compute_feature_relevance(kmeans, feature_names)

    # Compute cluster sizes and defect rates
    cluster_sizes = {}
    cluster_defects = {}
    dataset_defect_rate = float(labels.mean()) * 100  # Overall defect rate

    for i in range(k):
        mask = cluster_labels == i
        total = int(mask.sum())
        defective = int(labels[mask].sum())
        defect_rate = (defective / total * 100) if total > 0 else 0.0

        cluster_sizes[f"cluster_{i}"] = total
        cluster_defects[f"cluster_{i}"] = {
            "total": total,
            "defective": defective,
            "defect_rate": round(defect_rate, 2),
            "risk": "HIGH" if defect_rate > dataset_defect_rate else "LOW",
        }

    # Create visualizations directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Generate cluster visualization
    plot_clusters_pca(
        X_scaled,
        cluster_labels,
        labels,
        kmeans,
        cluster_sizes,
        f"{dataset_name} - K={k} Clusters",
        os.path.join(viz_dir, f"{dataset_name.lower()}_clusters.png"),
    )

    return {
        "k": k,
        "dataset": dataset_name,
        "n_samples": int(len(X_scaled)),
        "n_features": len(feature_names),
        "n_defective": int(labels.sum()),
        "defect_rate": round(dataset_defect_rate, 2),
        "cluster_sizes": cluster_sizes,
        "cluster_defects": cluster_defects,
        "inertia": inertia,
        "internal_metrics": internal_metrics,
        "external_metrics": external_metrics,
        "feature_relevance": relevance_df.to_dict("records"),
    }


def run_multi_k_analysis(
    file_path: str, sheet_name: str, k_values: list[int], base_output_dir: str
) -> list[dict]:
    """
    Run clustering for multiple k values on a dataset.

    Returns list of results for each k.
    """
    print(f"\nMulti-K Analysis for {sheet_name}...")

    # Load and preprocess data once
    df, label_col = load_sheet_data(file_path, sheet_name)
    X_scaled, labels, scaler, feature_names = preprocess_features(df, label_col)
    print(f"  Samples: {len(X_scaled)}, Features: {len(feature_names)}")

    results = []
    for k in k_values:
        print(f"  Running k={k}...")
        k_output_dir = os.path.join(base_output_dir, f"k{k}")
        os.makedirs(k_output_dir, exist_ok=True)

        result = analyze_dataset_for_k(
            X_scaled, labels, feature_names, k, sheet_name, k_output_dir
        )
        results.append(result)

        # Save individual result
        dataset_name = sheet_name.replace("_All", "").lower()
        json_path = os.path.join(k_output_dir, f"{dataset_name}_results.json")
        save_results_json(result, json_path)

        # Save feature rankings to text file
        feature_path = os.path.join(k_output_dir, f"{dataset_name}_top_features.txt")
        save_feature_rankings(result, feature_path)

    return results


def save_feature_rankings(result: dict, file_path: str) -> None:
    """
    Save feature rankings to a text file.

    Args:
        result: Result dictionary containing feature_relevance
        file_path: Path to save the rankings
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"TOP FEATURES BY CLUSTER RELEVANCE")
    lines.append(f"Dataset: {result['dataset']}, K={result['k']}")
    lines.append(f"Samples: {result['n_samples']}, Features: {result['n_features']}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Rank':<6} {'Feature':<50} {'Relevance':>12}")
    lines.append("-" * 70)

    for i, feat in enumerate(result["feature_relevance"]):
        lines.append(f"{i+1:<6} {feat['feature']:<50} {feat['relevance']:>12.6f}")

    lines.append("")
    lines.append("=" * 70)

    with open(file_path, "w") as f:
        f.write("\n".join(lines))


def plot_k_comparison(
    results_by_dataset: dict[str, list[dict]], save_path: str
) -> None:
    """
    Create comparison charts for different k values.

    Args:
        results_by_dataset: Dict mapping dataset name to list of results per k
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {"Ant_All": "steelblue", "Calcite_All": "coral"}

    for dataset_name, results in results_by_dataset.items():
        k_values = [r["k"] for r in results]
        silhouette = [r["internal_metrics"]["silhouette_score"] for r in results]
        db_index = [r["internal_metrics"]["davies_bouldin_index"] for r in results]
        v_measure = [r["external_metrics"]["v_measure"] for r in results]
        inertia = [r["inertia"] for r in results]

        color = colors.get(dataset_name, "gray")
        label = dataset_name.replace("_All", "")

        # Silhouette Score
        axes[0, 0].plot(k_values, silhouette, "o-", color=color, label=label, linewidth=2)
        axes[0, 0].set_xlabel("Number of Clusters (k)")
        axes[0, 0].set_ylabel("Silhouette Score")
        axes[0, 0].set_title("Silhouette Score vs K\n(Higher is better)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Davies-Bouldin Index
        axes[0, 1].plot(k_values, db_index, "o-", color=color, label=label, linewidth=2)
        axes[0, 1].set_xlabel("Number of Clusters (k)")
        axes[0, 1].set_ylabel("Davies-Bouldin Index")
        axes[0, 1].set_title("Davies-Bouldin Index vs K\n(Lower is better)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # V-Measure
        axes[1, 0].plot(k_values, v_measure, "o-", color=color, label=label, linewidth=2)
        axes[1, 0].set_xlabel("Number of Clusters (k)")
        axes[1, 0].set_ylabel("V-Measure")
        axes[1, 0].set_title("V-Measure vs K\n(Higher = better alignment with defects)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Inertia (Elbow Method)
        axes[1, 1].plot(k_values, inertia, "o-", color=color, label=label, linewidth=2)
        axes[1, 1].set_xlabel("Number of Clusters (k)")
        axes[1, 1].set_ylabel("Inertia (SSE)")
        axes[1, 1].set_title("Elbow Method\n(Look for elbow point)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison chart saved to {save_path}")


def generate_multi_k_report(results_by_dataset: dict[str, list[dict]]) -> str:
    """Generate a comparison report for multi-k analysis."""
    lines = []
    lines.append("=" * 70)
    lines.append("MULTI-K CLUSTERING COMPARISON REPORT")
    lines.append("=" * 70)
    lines.append("")

    for dataset_name, results in results_by_dataset.items():
        lines.append(f"{'=' * 70}")
        lines.append(f"Dataset: {dataset_name}")
        lines.append(f"{'=' * 70}")
        lines.append(f"Samples: {results[0]['n_samples']}")
        lines.append(f"Defective: {results[0]['n_defective']}")
        lines.append("")

        # Table header
        lines.append(f"{'K':>3} | {'Silhouette':>10} | {'DB Index':>10} | {'V-Measure':>10} | {'Inertia':>12} | Cluster Sizes")
        lines.append("-" * 80)

        best_silhouette = max(results, key=lambda r: r["internal_metrics"]["silhouette_score"])
        best_db = min(results, key=lambda r: r["internal_metrics"]["davies_bouldin_index"])
        best_v = max(results, key=lambda r: r["external_metrics"]["v_measure"])

        for r in results:
            k = r["k"]
            sil = r["internal_metrics"]["silhouette_score"]
            db = r["internal_metrics"]["davies_bouldin_index"]
            vm = r["external_metrics"]["v_measure"]
            inertia = r["inertia"]
            sizes = ", ".join([f"{v}" for v in r["cluster_sizes"].values()])

            # Mark best values
            sil_mark = " *" if r == best_silhouette else ""
            db_mark = " *" if r == best_db else ""
            vm_mark = " *" if r == best_v else ""

            lines.append(
                f"{k:>3} | {sil:>10.4f}{sil_mark:<2} | {db:>10.4f}{db_mark:<2} | {vm:>10.4f}{vm_mark:<2} | {inertia:>12.1f} | {sizes}"
            )

        lines.append("")
        lines.append("* = Best value for this metric")
        lines.append("")

        # Recommendation
        lines.append("RECOMMENDATION:")
        lines.append(f"  Best Silhouette: k={best_silhouette['k']} ({best_silhouette['internal_metrics']['silhouette_score']:.4f})")
        lines.append(f"  Best DB Index: k={best_db['k']} ({best_db['internal_metrics']['davies_bouldin_index']:.4f})")
        lines.append(f"  Best V-Measure: k={best_v['k']} ({best_v['external_metrics']['v_measure']:.4f})")
        lines.append("")

        # Cluster defect breakdown for each k
        lines.append("-" * 70)
        lines.append("CLUSTER DEFECT ANALYSIS (Defect Rate per Cluster)")
        lines.append("-" * 70)
        dataset_rate = results[0].get("defect_rate", 0)
        lines.append(f"Dataset average defect rate: {dataset_rate:.1f}%")
        lines.append("")

        for r in results:
            k = r["k"]
            lines.append(f"K={k}:")
            cluster_defects = r.get("cluster_defects", {})

            # Sort clusters by defect rate (highest first)
            sorted_clusters = sorted(
                cluster_defects.items(),
                key=lambda x: x[1]["defect_rate"],
                reverse=True
            )

            for cluster_name, info in sorted_clusters:
                total = info["total"]
                defective = info["defective"]
                rate = info["defect_rate"]
                risk = info["risk"]
                risk_marker = "!!" if risk == "HIGH" else "  "
                lines.append(
                    f"  {risk_marker} {cluster_name}: {total:>5} samples, "
                    f"{defective:>4} defective ({rate:>5.1f}% defect rate) - {risk} RISK"
                )
            lines.append("")

    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Clustering Analysis for Software Defect Metrics"
    )
    parser.add_argument(
        "--no-outliers",
        action="store_true",
        help="Remove outliers (Z-score > 3) before clustering",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Z-score threshold for outlier removal (default: 3.0)",
    )
    parser.add_argument(
        "--multi-k",
        action="store_true",
        help="Run clustering with multiple k values (2,3,4,5) and compare",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse command-line arguments
    args = parse_args()

    # Update CONFIG based on arguments
    if args.no_outliers:
        CONFIG["remove_outliers"] = True
        CONFIG["outlier_z_threshold"] = args.z_threshold

    # Handle multi-k analysis
    if args.multi_k:
        run_multi_k_main(args)
        return

    # Set output directory for single-k analysis
    if args.no_outliers:
        CONFIG["output_dir"] = "results_no_outliers"

    print("=" * 60)
    print("Software Defect Metrics - Clustering Analysis")
    if CONFIG["remove_outliers"]:
        print(f"  Mode: Outliers removed (Z > {CONFIG['outlier_z_threshold']})")
    else:
        print("  Mode: All samples included")
    print("=" * 60)

    # Create output directory
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Analyze each dataset
    all_results = []
    for sheet_name in CONFIG["sheets"]:
        try:
            result = analyze_dataset(CONFIG["data_file"], sheet_name, output_dir)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR analyzing {sheet_name}: {e}")

    # Generate and save text report
    if all_results:
        report = generate_text_report(all_results)
        report_path = os.path.join(output_dir, "analysis_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nText report saved to {report_path}")

        # Print report to console as well
        print("\n" + report)

    print("\nAnalysis complete!")
    print(f"Results saved to: {output_dir}/")


def run_multi_k_main(args):
    """Run multi-k analysis mode."""
    k_values = [2, 3, 4, 5]

    # Determine output directory
    if args.no_outliers:
        output_dir = "results_multi_k_no_outliers"
    else:
        output_dir = "results_multi_k"

    print("=" * 60)
    print("Software Defect Metrics - Multi-K Clustering Analysis")
    print(f"  K values: {k_values}")
    if CONFIG["remove_outliers"]:
        print(f"  Mode: Outliers removed (Z > {CONFIG['outlier_z_threshold']})")
    else:
        print("  Mode: All samples included")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Run analysis for each dataset
    results_by_dataset = {}
    for sheet_name in CONFIG["sheets"]:
        try:
            results = run_multi_k_analysis(
                CONFIG["data_file"], sheet_name, k_values, output_dir
            )
            results_by_dataset[sheet_name] = results
        except Exception as e:
            print(f"  ERROR analyzing {sheet_name}: {e}")

    # Generate comparison chart
    if results_by_dataset:
        chart_path = os.path.join(output_dir, "k_comparison_chart.png")
        plot_k_comparison(results_by_dataset, chart_path)

        # Generate comparison report
        report = generate_multi_k_report(results_by_dataset)
        report_path = os.path.join(output_dir, "multi_k_comparison.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nComparison report saved to {report_path}")

        # Print report
        print("\n" + report)

    print("\nMulti-K analysis complete!")
    print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
