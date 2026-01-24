"""
Data loading and preprocessing utilities.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import DATASETS


def load_dataset(dataset_name: str) -> tuple[pd.DataFrame, str, dict]:
    """
    Load a dataset by name from the configuration.

    Args:
        dataset_name: Name of the dataset (e.g., "ant-ivy", "calcite")

    Returns:
        Tuple of (DataFrame, label_column_name, feature_name_map)
        feature_name_map maps column names to actual feature names (if available)

    Raises:
        ValueError: If dataset name is not found in configuration
    """
    if dataset_name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

    config = DATASETS[dataset_name]
    file_path = config["file"]
    is_csv = file_path.endswith(".csv")

    # Load feature name mapping if available (only for Excel files)
    feature_name_map = {}
    feature_name_row = config.get("feature_name_row")
    if feature_name_row is not None and not is_csv:
        # Read just the feature name row
        df_names = pd.read_excel(
            file_path,
            sheet_name=config["sheet"],
            header=None,
            nrows=feature_name_row + 1,
        )
        # Also read header row to get column indices
        df_header = pd.read_excel(
            file_path,
            sheet_name=config["sheet"],
            header=None,
            skiprows=config["header_row"],
            nrows=1,
        )
        # Build mapping: header column name -> feature name
        for i in range(len(df_header.columns)):
            header_name = df_header.iloc[0, i]
            if i < len(df_names.columns):
                feature_name = df_names.iloc[feature_name_row, i]
                if pd.notna(feature_name):
                    feature_name_map[header_name] = str(feature_name)

    # Load main data
    if is_csv:
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(
            file_path,
            sheet_name=config["sheet"],
            header=config["header_row"],
        )

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove rows with NaN in label column
    label_col = config["label_column"]
    if label_col in df.columns:
        df = df.dropna(subset=[label_col])

    return df, label_col, feature_name_map


def get_feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    """
    Get list of numeric feature columns (excluding label and ID columns).

    Args:
        df: DataFrame with data
        label_col: Name of the label column

    Returns:
        List of feature column names
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Columns to exclude (convert to lowercase strings for comparison)
    exclude = {label_col.lower(), "id", "version", "version-id"}
    feature_cols = [
        col for col in numeric_cols
        if str(col).lower() not in exclude and col != label_col
    ]

    return feature_cols


def preprocess_features(
    df: pd.DataFrame,
    label_col: str,
    feature_name_map: dict = None,
    remove_outliers: bool = False,
    z_threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """
    Preprocess features: extract, scale, and optionally remove outliers.

    Args:
        df: DataFrame with data
        label_col: Name of the label column
        feature_name_map: Optional mapping from column names to feature names
        remove_outliers: Whether to remove outliers
        z_threshold: Z-score threshold for outlier removal

    Returns:
        Tuple of (scaled_features, labels, scaler, feature_names)
    """
    feature_cols = get_feature_columns(df, label_col)

    # Map column names to actual feature names if mapping provided
    if feature_name_map:
        feature_names = [
            feature_name_map.get(col, str(col)) for col in feature_cols
        ]
    else:
        feature_names = [str(col) for col in feature_cols]

    # Extract features and labels
    X = df[feature_cols].values
    labels = (df[label_col] > 0).astype(int).values

    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Remove outliers if requested
    if remove_outliers:
        X_scaled, labels, mask = remove_outliers_zscore(X_scaled, labels, z_threshold)
        print(f"    Removed {(~mask).sum()} outliers (Z > {z_threshold})")

    return X_scaled, labels, scaler, feature_names


def remove_outliers_zscore(
    X: np.ndarray, labels: np.ndarray, z_threshold: float = 3.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove samples with any feature having Z-score above threshold.

    Args:
        X: Scaled feature array
        labels: Label array
        z_threshold: Z-score threshold

    Returns:
        Tuple of (filtered_X, filtered_labels, keep_mask)
    """
    # Identify samples with any feature exceeding threshold
    z_scores = np.abs(X)
    max_z = np.max(z_scores, axis=1)
    keep_mask = max_z <= z_threshold

    return X[keep_mask], labels[keep_mask], keep_mask
