"""
Classification utilities for supervised defect prediction.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)


def get_classifier(classifier_type: str, random_state: int = 42):
    """
    Get a classifier instance.

    Args:
        classifier_type: 'rf' for Random Forest, 'lr' for Logistic Regression
        random_state: Random seed for reproducibility

    Returns:
        Configured classifier instance
    """
    if classifier_type == 'rf':
        return RandomForestClassifier(
            n_estimators=100,
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
