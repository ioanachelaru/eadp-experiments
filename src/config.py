"""
Configuration for defect prediction experiments.

Centralizes all paths, feature definitions, and dataset configurations.
"""

# =============================================================================
# RAW DATA PATHS
# =============================================================================
RAW_DATA = {
    "calcite_sm": "data/All Calcite 1.0.0-1.15.0 software metrics.xlsx",
    "ant_ivy_sm": "data/ant-ivy-all versions.xlsx",
    "effort_data": "effort_data/All Calcite 1.0.0-1.15.0 effort-related metrics.xlsx",
    "coverage_pattern": "data/Coverage-Calcite-*-filename.csv",
}

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================
COVERAGE_FEATURES = [
    "COV_INSTRUCTION",
    "COV_BRANCH",
    "COV_LINE",
    "COV_COMPLEXITY",
    "COV_METHOD",
]

EFFORT_FEATURES_26 = [
    "CHANGE_TYPE_computation",
    "CHANGE_TYPE_data",
    "CHANGE_TYPE_other",
    "HASSAN_edhcm",
    "HASSAN_hcm",
    "HASSAN_ldhcm",
    "HASSAN_lgdhcm",
    "HASSAN_whcm",
    "ISSUE_major_bug",
    "ISSUE_major_improvement",
    "MOSER_authors",
    "MOSER_bugfix",
    "MOSER_revisions",
    "MOSER_sum_lines_deleted",
    "MOSER_weighted_age",
    "PMD_arp",
    "PMD_cis",
    "PMD_odpl",
    "PMD_rule_type_basic rules",
    "PMD_rule_type_controversial rules",
    "PMD_rule_type_design rules",
    "PMD_rule_type_string and stringbuffer rules",
    "PMD_rule_type_unnecessary and unused code rules",
    "PMD_severity_critical",
    "PMD_severity_major",
    "PMD_severity_minor",
]

EFFORT_PREFIXES = ("CHANGE_TYPE_", "HASSAN_", "ISSUE_", "MOSER_", "PMD_")

# =============================================================================
# METADATA COLUMNS
# =============================================================================
METADATA_COLUMNS = {
    "calcite": ["Calcite version", "ID", "file", "Version-ID", "Bug"],
    "ant_ivy": ["Version", "Id", "Class", "Label"],
}

# Effort data sheet configurations
EFFORT_SHEETS = {
    "26_common": {
        "sheet": "26 Metrics (Ant int Calcite)",
        "description": "26 metrics common to Ant & Calcite (corr>=0.1)",
    },
    "ant_all": {
        "sheet": "Ant_All",
        "header_row": 9,
        "exclude_cols": ["Ant version", "ID", "file", "Bug"],
        "description": "149 Ant effort-data features (all CHANGE, HASSAN, ISSUE, MOSER, PMD)",
    },
    "calcite_all": {
        "sheet": "Calcite_All",
        "header_row": 9,
        "exclude_cols": ["Calcite version", "ID", "file", "Bug", "Version-ID"],
        "description": "170 Calcite effort-data features (all CHANGE, HASSAN, ISSUE, MOSER, PMD)",
    },
}

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================
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
