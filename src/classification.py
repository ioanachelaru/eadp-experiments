"""
Classification utilities for supervised defect prediction.
"""

import fnmatch
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def get_classifier(classifier_type: str, random_state: int = 42, n_estimators: int = 100):
    """
    Get a classifier instance.

    Args:
        classifier_type: 'rf' for Random Forest, 'lr' for Logistic Regression
        random_state: Random seed for reproducibility
        n_estimators: Number of trees for Random Forest

    Returns:
        Configured classifier instance
    """
    if classifier_type == 'rf':
        return RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
        )
    elif classifier_type == 'lr':
        return LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs',
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


def run_cross_validation(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Run stratified k-fold cross-validation.

    Args:
        clf: Classifier instance
        X: Feature matrix
        y: Labels
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        Dictionary with CV results
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': 'roc_auc',
        'avg_precision': make_scorer(average_precision_score),
    }

    results = cross_validate(
        clf, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    return results


def get_feature_importances(clf, feature_names: list) -> list[tuple[str, float]]:
    """
    Extract feature importances from a fitted classifier.

    Args:
        clf: Fitted classifier
        feature_names: List of feature names

    Returns:
        List of (feature_name, importance) tuples, sorted by importance
    """
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        # For logistic regression, use absolute coefficient values
        importances = np.abs(clf.coef_).flatten()
    else:
        return []

    # Pair with feature names and sort
    feature_importance_pairs = list(zip(feature_names, importances))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

    return feature_importance_pairs


def format_cv_results(results: dict) -> dict:
    """
    Format cross-validation results with mean and std.

    Args:
        results: Raw CV results from cross_validate

    Returns:
        Dictionary with mean and std for each metric
    """
    formatted = {}
    for key in results:
        if key.startswith('test_'):
            metric_name = key.replace('test_', '')
            values = results[key]
            formatted[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': [float(v) for v in values],
            }
    return formatted


def save_cv_folds(
    df: pd.DataFrame,
    y: np.ndarray,
    output_dir: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> None:
    """
    Save CV fold splits to disk (indices + full CSVs).

    Args:
        df: Original DataFrame with all columns (including metadata)
        y: Target labels array
        output_dir: Directory to save folds (e.g., results/classification/folds/dataset_name)
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df, y)):
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        # Save indices
        indices = {"train": train_idx.tolist(), "test": test_idx.tolist()}
        with open(os.path.join(fold_dir, "indices.json"), "w") as f:
            json.dump(indices, f)

        # Save CSVs
        df.iloc[train_idx].to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        df.iloc[test_idx].to_csv(os.path.join(fold_dir, "test.csv"), index=False)


def apply_feature_exclusions(
    feature_cols: list[str],
    exclude_patterns: list[str],
) -> tuple[list[str], list[str]]:
    """
    Filter feature columns based on exclusion patterns.

    Args:
        feature_cols: List of feature column names
        exclude_patterns: List of patterns — exact names ('MOSER_bugfix')
                         or wildcard prefixes ('ISSUE_*')

    Returns:
        Tuple of (filtered_features, excluded_features)
    """
    excluded = []
    kept = []
    for col in feature_cols:
        should_exclude = False
        for pattern in exclude_patterns:
            if "*" in pattern or "?" in pattern:
                if fnmatch.fnmatch(col, pattern):
                    should_exclude = True
                    break
            else:
                if col == pattern:
                    should_exclude = True
                    break
        if should_exclude:
            excluded.append(col)
        else:
            kept.append(col)
    return kept, excluded


def run_chronological_cv(
    classifier_type: str,
    df: pd.DataFrame,
    label_col: str,
    feature_cols: list[str],
    version_col: str,
    release_order: list[str],
    output_dir: str,
    random_state: int = 42,
    n_estimators: int = 100,
    exclude_patterns: list[str] | None = None,
) -> dict:
    """
    Run chronological expanding-window cross-validation.

    For each split point k (1..len(release_order)-1):
      - Train on releases 0..k-1
      - Test on release k
      - Fit scaler on train only, transform both
      - Fresh classifier instance per fold

    Args:
        classifier_type: 'rf' or 'lr'
        df: Full DataFrame with metadata, features, and label
        label_col: Name of the label column
        feature_cols: List of feature column names
        version_col: Name of the version column
        release_order: Ordered list of release version strings
        output_dir: Directory to save per-fold results
        random_state: Random seed
        n_estimators: Number of trees for RF
        exclude_patterns: Optional list of feature exclusion patterns

    Returns:
        Dictionary with summary results and per-fold details
    """
    # Apply feature exclusions if specified
    excluded_features = []
    if exclude_patterns:
        feature_cols, excluded_features = apply_feature_exclusions(
            feature_cols, exclude_patterns
        )
        print(f"  Feature exclusion: {len(excluded_features)} features removed, {len(feature_cols)} remaining")
        if excluded_features:
            print(f"    Excluded: {excluded_features}")

    # Filter to only versions that exist in the data
    available_versions = set(str(v) for v in df[version_col].dropna().unique())
    release_order = [v for v in release_order if v in available_versions]
    print(f"  Releases in data: {release_order}")

    os.makedirs(output_dir, exist_ok=True)

    fold_details = []
    all_metrics = {
        "precision": [], "recall": [], "f1": [],
        "roc_auc": [], "avg_precision": [],
    }

    # Aggregate feature importances across folds
    importance_accumulator = np.zeros(len(feature_cols))
    n_importance_folds = 0

    for k in range(1, len(release_order)):
        train_versions = release_order[:k]
        test_version = release_order[k]

        # Split data
        train_mask = df[version_col].astype(str).isin(train_versions)
        test_mask = df[version_col].astype(str) == test_version

        df_train = df[train_mask]
        df_test = df[test_mask]

        if len(df_train) == 0 or len(df_test) == 0:
            print(f"  Skipping fold {test_version}: empty train or test set")
            continue

        X_train = df_train[feature_cols].values.astype(float)
        X_test = df_test[feature_cols].values.astype(float)
        y_train = df_train[label_col].values.astype(int)
        y_test = df_test[label_col].values.astype(int)

        # Handle NaN values
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        # Scale per fold (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fresh classifier
        clf = get_classifier(classifier_type, random_state=random_state, n_estimators=n_estimators)
        clf.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else None

        # Metrics
        fold_metrics = {
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        if y_proba is not None and len(np.unique(y_test)) > 1:
            fold_metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            fold_metrics["avg_precision"] = float(average_precision_score(y_test, y_proba))
        else:
            fold_metrics["roc_auc"] = 0.0
            fold_metrics["avg_precision"] = 0.0

        for metric_name in all_metrics:
            all_metrics[metric_name].append(fold_metrics[metric_name])

        # Feature importances
        if hasattr(clf, "feature_importances_"):
            importance_accumulator += clf.feature_importances_
            n_importance_folds += 1

        # Fold detail record
        fold_info = {
            "test_version": test_version,
            "train_versions": train_versions,
            "n_train": len(df_train),
            "n_test": len(df_test),
            "n_defective_train": int(y_train.sum()),
            "n_defective_test": int(y_test.sum()),
            "defect_rate_train": float(y_train.mean() * 100),
            "defect_rate_test": float(y_test.mean() * 100),
            **fold_metrics,
        }
        fold_details.append(fold_info)

        # Save per-fold data
        fold_dir = os.path.join(output_dir, f"fold_{test_version}")
        os.makedirs(fold_dir, exist_ok=True)

        df_train.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        df_test.to_csv(os.path.join(fold_dir, "test.csv"), index=False)

        # Predictions CSV
        pred_df = pd.DataFrame({
            "true_label": y_test,
            "predicted": y_pred,
        })
        if y_proba is not None:
            pred_df["probability"] = y_proba
        # Add file/version metadata if available
        if "file" in df_test.columns:
            pred_df.insert(0, "file", df_test["file"].values)
        if version_col in df_test.columns:
            pred_df.insert(0, "version", df_test[version_col].values)
        pred_df.to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)

        # Fold metrics JSON
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(fold_info, f, indent=2)

        print(f"  Fold {test_version}: train={len(df_train)}, test={len(df_test)}, "
              f"F1={fold_metrics['f1']:.4f}, AUC={fold_metrics['roc_auc']:.4f}")

    # Summary metrics
    metrics_summary = {}
    fold_versions = [fd["test_version"] for fd in fold_details]
    for metric_name, values in all_metrics.items():
        metrics_summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
            "fold_versions": fold_versions,
        }

    # Average feature importances
    feature_importances = []
    if n_importance_folds > 0:
        avg_importances = importance_accumulator / n_importance_folds
        pairs = sorted(
            zip(feature_cols, avg_importances),
            key=lambda x: x[1],
            reverse=True,
        )
        feature_importances = [
            {"feature": name, "importance": round(float(imp), 6)}
            for name, imp in pairs
        ]

    results = {
        "cv_mode": "chronological",
        "n_folds": len(fold_details),
        "release_order": release_order,
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "metrics": metrics_summary,
        "fold_details": fold_details,
        "feature_importances": feature_importances,
    }

    if excluded_features:
        results["excluded_features"] = excluded_features

    return results
