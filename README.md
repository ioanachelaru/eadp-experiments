# eadp-experiments

Effort-aware defect prediction experiments comparing clustering and supervised learning approaches on software metrics.

## Overview

This project analyzes software metrics from Apache Calcite to evaluate different approaches for predicting software defects. We compare:

1. **Clustering (DBSCAN)** - Unsupervised approach using feature relevance for cluster separation
2. **Supervised Learning (Random Forest, Logistic Regression)** - Classification with cross-validation

We also compare three feature sets:
- **Software Metrics (SM)** - Code complexity, coupling, cohesion metrics
- **Effort + Coverage** - Code churn, file age, test coverage, PMD warnings
- **Combined** - All features together

## Key Findings

### Supervised Learning Dramatically Outperforms Clustering

| Approach | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Clustering (DBSCAN) | 25-47% | 5-8% | 8-14% |
| Random Forest (SM-only) | 39.2% | 32.0% | 35.2% |
| Random Forest (Effort+Cov) | 80.8% | 51.5% | 62.9% |

### Effort + Coverage Features Outperform Software Metrics

| Feature Set | F1-Score | ROC-AUC |
|-------------|----------|---------|
| SM-only (30 features) | 35.2% | 0.662 |
| Effort+Coverage (31 features) | 62.9% | 0.947 |
| Combined (61 features) | 63.4% | 0.949 |

**Top predictive features:**
1. Test coverage (`COV_BRANCH`, `COV_INSTRUCTION`)
2. Code churn (`HASSAN_whcm`)
3. File age (`MOSER_weighted_age`)
4. Code quality warnings (`PMD_severity_minor`, `PMD_severity_major`)

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
eadp-experiments/
├── eadp.py                    # Unified CLI entry point
├── README.md
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py              # ALL configuration (paths, features, defaults)
│   ├── data_utils.py          # Data loading and preprocessing
│   ├── prepare.py             # Data preparation functions
│   ├── compare.py             # Feature comparison logic
│   ├── clustering.py          # K-Means and DBSCAN implementations
│   ├── classification.py      # Random Forest, Logistic Regression
│   ├── metrics.py             # Internal and external metrics
│   ├── feature_analysis.py    # Feature relevance computation
│   └── plotting.py            # Visualization functions
│
├── data/                      # Input data
│   ├── Calcite-top30-sm-only-v1.1+.csv
│   ├── Calcite-effort-cov-only.csv
│   ├── Calcite-top30-sm-cov-effort.csv
│   └── ...
│
├── effort_data/               # Effort-related metrics
│
└── results/                   # Output results
    ├── supervised_learning_comparison.txt
    ├── classification/        # Per-dataset JSON results
    ├── calcite/dbscan/        # Clustering results
    └── ant-ivy/dbscan/
```

## Usage

All functionality is accessed through the unified `eadp.py` CLI:

```bash
python eadp.py --help
```

### Data Preparation

```bash
# Extract SM features from raw Excel data
python eadp.py prepare --action extract-sm --dataset calcite
python eadp.py prepare --action extract-sm --dataset ant-ivy

# Merge coverage data with SM data
python eadp.py prepare --action merge-coverage

# Create combined dataset (top-30 SM + effort + coverage)
python eadp.py prepare --action create-combined

# Create top-30 SM only dataset
python eadp.py prepare --action create-top30-sm
python eadp.py prepare --action create-top30-sm --v11-plus  # Exclude v1.0.0

# Create effort + coverage only dataset
python eadp.py prepare --action create-effort-only
```

### Supervised Learning (Recommended)

```bash
# Compare all three feature sets with Random Forest
python eadp.py classify --compare-all

# Compare with Logistic Regression
python eadp.py classify --compare-all --classifier lr

# Run on a single dataset
python eadp.py classify --dataset calcite-effort-cov-only --classifier rf
```

### Clustering

```bash
# Run DBSCAN on Calcite dataset
python eadp.py cluster --dataset calcite --algorithm dbscan

# Run on Ant-Ivy dataset
python eadp.py cluster --dataset ant-ivy --algorithm dbscan

# With outlier removal
python eadp.py cluster --dataset ant-ivy --algorithm dbscan --no-outliers

# K-Means with custom cluster count
python eadp.py cluster --dataset ant-ivy --algorithm kmeans --k 3
```

### Compare Clustering with Effort Features

```bash
python eadp.py compare --dataset calcite --algorithm dbscan
```

## Available Commands

| Command | Description |
|---------|-------------|
| `prepare` | Data preparation (extract, merge, create datasets) |
| `cluster` | Run clustering experiments (DBSCAN, K-Means) |
| `classify` | Run supervised learning experiments (RF, LR) |
| `compare` | Compare clustering features with effort data |

### Prepare Actions

| Action | Description |
|--------|-------------|
| `extract-sm` | Extract SM_* features from raw Excel data |
| `merge-coverage` | Merge coverage CSVs with SM data |
| `create-combined` | Create top-30 SM + effort + coverage dataset |
| `create-top30-sm` | Create top-30 SM features only dataset |
| `create-effort-only` | Create effort + coverage only dataset |

## Experiments

### Experiment 1: Clustering Analysis

**Goal:** Identify features that best separate clusters in the data.

**Method:** DBSCAN with automatic epsilon detection, feature relevance based on standard deviation across cluster centers.

**Results:**
- Ant-Ivy: 4 clusters, Silhouette 0.507, top features are SM_interface_* metrics
- Calcite: 65 clusters, Silhouette 0.423, top features are SM_enum_* metrics
- Clustering achieves poor defect prediction: ~6% recall, ~35% precision

**Conclusion:** Clustering identifies structurally distinct code but fails at defect prediction due to class imbalance (7.5% defective).

### Experiment 2: Supervised Learning Comparison

**Goal:** Compare SM vs Effort+Coverage features for defect prediction.

**Method:**
- 5-fold stratified cross-validation (preserves class ratio in each fold)
- `class_weight='balanced'` to handle class imbalance
- Random Forest and Logistic Regression classifiers

**Datasets (all 18,676 samples, 7.5% defective):**

| Dataset | Features | Description |
|---------|----------|-------------|
| `calcite-top30-sm-only-v1.1+` | 30 | Top software metrics |
| `calcite-effort-cov-only` | 31 | 26 effort + 5 coverage |
| `calcite-top30-sm-cov-effort` | 61 | Combined |

**Results (Random Forest):**

| Feature Set | Precision | Recall | F1 | ROC-AUC |
|-------------|-----------|--------|-----|---------|
| SM-only | 39.2% | 32.0% | 35.2% | 0.662 |
| Effort+Cov | 80.8% | 51.5% | 62.9% | 0.947 |
| Combined | 81.7% | 51.9% | 63.4% | 0.949 |

**Conclusions:**
1. Effort+Coverage features outperform SM by 79% (F1: 35% -> 63%)
2. Adding SM to Effort+Cov provides marginal improvement (+0.5% F1)
3. Random Forest outperforms Logistic Regression (63% vs 31% F1)
4. Supervised learning vastly outperforms clustering (~9x better recall)

## Output Files

### Supervised Learning
```
results/supervised_learning_comparison.txt    # Main comparison report
results/classification/
├── calcite-top30-sm-only-v1.1+_rf_results.json
├── calcite-effort-cov-only_rf_results.json
├── calcite-top30-sm-cov-effort_rf_results.json
└── ..._lr_results.json
```

### Clustering
```
results/{dataset}/{algorithm}/
├── results.json               # Full results with feature rankings
├── top_features.txt           # Human-readable feature rankings
├── metrics_summary.txt        # Clustering metrics
├── comparison_report.txt      # Comparison with effort-data
└── visualizations/
    ├── clusters_pca.png
    ├── feature_relevance.png
    └── k_distance.png         # DBSCAN only
```

## Metrics

### Classification Metrics
- **Precision**: Of predicted defects, how many are actual defects
- **Recall**: Of actual defects, how many were predicted
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (0.5 = random, 1.0 = perfect)
- **PR-AUC**: Area under Precision-Recall curve (better for imbalanced data)

### Clustering Metrics
- **Silhouette Score**: Cluster cohesion vs separation (-1 to 1, higher is better)
- **V-Measure**: Harmonic mean of homogeneity and completeness (0 to 1)

## Configuration

All configuration is centralized in `src/config.py`:

```python
# Raw data paths
RAW_DATA = {
    "calcite_sm": "data/All Calcite 1.0.0-1.15.0 software metrics.xlsx",
    "ant_ivy_sm": "data/ant-ivy-all versions.xlsx",
    "effort_data": "effort_data/All Calcite 1.0.0-1.15.0 effort-related metrics.xlsx",
    "coverage_pattern": "data/Coverage-Calcite-*-filename.csv",
}

# Feature definitions
COVERAGE_FEATURES = ["COV_INSTRUCTION", "COV_BRANCH", "COV_LINE", ...]
EFFORT_FEATURES_26 = ["CHANGE_TYPE_computation", "HASSAN_edhcm", ...]

# Dataset configurations
DATASETS = {
    "calcite-top30-sm-only-v1.1+": {...},
    "calcite-effort-cov-only": {...},
    "calcite-top30-sm-cov-effort": {...},
    # ... more datasets
}
```

## License

See [LICENSE](LICENSE) for details.
