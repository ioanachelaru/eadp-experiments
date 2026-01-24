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
    "calcite-with-coverage": {
        "file": "data/Calcite-with-coverage.csv",
        "sheet": None,  # CSV file, no sheet
        "header_row": 0,  # pandas saves header at row 0
        "feature_name_row": None,
        "label_column": "Bug",
        "description": "Calcite software metrics + test coverage (versions 1.1.0-1.15.0)",
    },
    "calcite-top30-coverage": {
        "file": "data/Calcite-top30-with-coverage.csv",
        "sheet": None,
        "header_row": 0,
        "feature_name_row": None,
        "label_column": "Bug",
        "description": "Calcite top-30 features + 5 coverage features (35 total)",
    },
    "calcite-top30-effort-cov": {
        "file": "data/Calcite-top30-effort-cov.csv",
        "sheet": None,
        "header_row": 0,
        "feature_name_row": None,
        "label_column": "Bug",
        "description": "Calcite: top-30 SM + 26 effort + 5 coverage (61 features)",
    },
    "calcite-sm-only": {
        "file": "data/Calcite-SM-only.csv",
        "sheet": None,
        "header_row": 0,
        "feature_name_row": None,
        "label_column": "Bug",
        "description": "Calcite SM_* software metrics only (2859 features)",
    },
    "ant-ivy-sm-only": {
        "file": "data/Ant-Ivy-SM-only.csv",
        "sheet": None,
        "header_row": 0,
        "feature_name_row": None,
        "label_column": "Label",
        "description": "Ant-Ivy SM_* software metrics only (3624 features)",
    },
    "calcite-top30-sm-cov-effort": {
        "file": "data/Calcite-top30-sm-cov-effort.csv",
        "sheet": None,
        "header_row": 0,
        "feature_name_row": None,
        "label_column": "Bug",
        "description": "Calcite: top-30 SM + 5 coverage + 26 effort (61 features)",
    },
    "calcite-top30-sm-only": {
        "file": "data/Calcite-top30-sm-only.csv",
        "sheet": None,
        "header_row": 0,
        "feature_name_row": None,
        "label_column": "Bug",
        "description": "Calcite: top-30 SM features only",
    },
    "calcite-effort-cov-only": {
        "file": "data/Calcite-effort-cov-only.csv",
        "sheet": None,
        "header_row": 0,
        "feature_name_row": None,
        "label_column": "Bug",
        "description": "Calcite: 26 effort + 5 coverage features (31 total)",
    },
    "calcite-top30-sm-only-v1.1+": {
        "file": "data/Calcite-top30-sm-only-v1.1+.csv",
        "sheet": None,
        "header_row": 0,
        "feature_name_row": None,
        "label_column": "Bug",
        "description": "Calcite: top-30 SM features (v1.1.0+ only, for fair comparison)",
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
