"""
Clustering algorithms: K-Means and DBSCAN.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors

from .config import CLUSTERING_DEFAULTS


def run_kmeans(
    X: np.ndarray,
    n_clusters: int = None,
    random_state: int = None,
) -> tuple[KMeans, np.ndarray]:
    """
    Run K-Means clustering.

    Args:
        X: Scaled feature array
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        Tuple of (fitted KMeans model, cluster labels)
    """
    defaults = CLUSTERING_DEFAULTS["kmeans"]
    n_clusters = n_clusters or defaults["n_clusters"]
    random_state = random_state if random_state is not None else defaults["random_state"]

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=defaults["n_init"],
        max_iter=defaults["max_iter"],
    )
    cluster_labels = kmeans.fit_predict(X)

    return kmeans, cluster_labels


def find_optimal_eps(X: np.ndarray, min_samples: int = 5, max_samples: int = 1000) -> float:
    """
    Find optimal eps for DBSCAN using k-distance graph (knee method).

    Uses random sampling for memory efficiency on large datasets.

    Args:
        X: Scaled feature array
        min_samples: min_samples parameter for DBSCAN
        max_samples: Maximum samples to use for eps estimation

    Returns:
        Optimal eps value
    """
    # Sample data if too large (memory efficiency)
    n_samples = X.shape[0]
    if n_samples > max_samples:
        np.random.seed(42)
        indices = np.random.choice(n_samples, max_samples, replace=False)
        X_sample = X[indices]
        print(f"    Using {max_samples} samples for eps estimation (from {n_samples})")
    else:
        X_sample = X

    # Compute k-distances on sample
    k = min_samples
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_sample)
    distances, _ = nbrs.kneighbors(X_sample)

    # Get k-th nearest neighbor distances, sorted
    k_distances = np.sort(distances[:, k - 1])

    # Find knee point using simple heuristic:
    # Look for the point of maximum curvature
    # Using the perpendicular distance from line connecting first and last point

    n = len(k_distances)
    if n < 2:
        return 0.5  # Default fallback

    # Line from first to last point
    p1 = np.array([0, k_distances[0]])
    p2 = np.array([n - 1, k_distances[-1]])

    # Perpendicular distance for each point
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        return float(np.median(k_distances))

    distances_to_line = []
    for i, d in enumerate(k_distances):
        point = np.array([i, d])
        dist = np.abs(np.cross(line_vec, p1 - point)) / line_len
        distances_to_line.append(dist)

    # Knee is at maximum distance
    knee_idx = np.argmax(distances_to_line)
    eps = float(k_distances[knee_idx])

    # Ensure eps is reasonable (only set minimum)
    if eps < 0.1:
        eps = 0.5

    return eps


def run_dbscan(
    X: np.ndarray,
    eps: float = None,
    min_samples: int = None,
) -> tuple[DBSCAN, np.ndarray, dict]:
    """
    Run DBSCAN clustering.

    Args:
        X: Scaled feature array
        eps: Maximum distance between samples. If None, auto-detect.
        min_samples: Minimum samples in a neighborhood

    Returns:
        Tuple of (fitted DBSCAN model, cluster labels, info dict)
    """
    defaults = CLUSTERING_DEFAULTS["dbscan"]
    min_samples = min_samples or defaults["min_samples"]

    # Auto-detect eps if not provided
    if eps is None:
        eps = find_optimal_eps(X, min_samples)
        auto_eps = True
    else:
        auto_eps = False

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=defaults["metric"],
    )
    cluster_labels = dbscan.fit_predict(X)

    # Compute cluster info (convert to native Python types for JSON)
    unique_labels = set(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = int((cluster_labels == -1).sum())

    info = {
        "eps": float(eps),
        "auto_eps": auto_eps,
        "min_samples": int(min_samples),
        "n_clusters": int(n_clusters),
        "n_noise": n_noise,
        "noise_ratio": float(n_noise / len(cluster_labels)) if len(cluster_labels) > 0 else 0.0,
    }

    return dbscan, cluster_labels, info


def get_cluster_centers(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute cluster centers (works for any clustering algorithm).

    For DBSCAN, this computes the mean of each cluster (excluding noise).

    Args:
        X: Feature array
        labels: Cluster labels

    Returns:
        Array of cluster centers, shape (n_clusters, n_features)
    """
    unique_labels = sorted(set(labels))
    # Exclude noise label (-1)
    cluster_labels = [l for l in unique_labels if l >= 0]

    centers = []
    for label in cluster_labels:
        mask = labels == label
        center = X[mask].mean(axis=0)
        centers.append(center)

    return np.array(centers) if centers else np.empty((0, X.shape[1]))
