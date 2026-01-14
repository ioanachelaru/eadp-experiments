"""
Configuration for clustering analysis experiments.
"""

# Dataset configurations
DATASETS = {
    "ant-ivy": {
        "file": "data/ant-ivy-all versions.xlsx",
        "sheet": "ant-ivy-all versions",
        "header_row": 8,
        "feature_name_row": 7,  # Row containing actual feature names
        "label_column": "Label",
        "description": "Ant-Ivy software metrics",
    },
    "calcite": {
        "file": "data/All Calcite 1.0.0-1.15.0 software metrics.xlsx",
        "sheet": "All SM",
        "header_row": 9,
        "feature_name_row": None,  # No separate feature name row
        "label_column": "Bug",
        "description": "Calcite software metrics",
    },
}

# Clustering algorithm defaults
CLUSTERING_DEFAULTS = {
    "kmeans": {
        "n_clusters": 2,
        "random_state": 42,
        "n_init": 10,
        "max_iter": 300,
    },
    "dbscan": {
        "eps": None,  # Auto-detect using k-distance
        "min_samples": 5,
        "metric": "euclidean",
    },
}

# Preprocessing defaults
PREPROCESSING = {
    "remove_outliers": False,
    "outlier_z_threshold": 3.0,
}

# Output settings
OUTPUT = {
    "base_dir": "results",
    "top_features_display": 20,  # Number of top features to show in plots
}
