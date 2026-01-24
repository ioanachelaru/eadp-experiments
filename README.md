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
├── run_clustering.py          # Clustering experiments
├── run_classification.py      # Supervised learning experiments
├── compare_features.py        # Compare clustering vs effort-data features
├── src/
│   ├── config.py              # Dataset and algorithm configurations
│   ├── data_utils.py          # Data loading and preprocessing
│   ├── clustering.py          # K-Means and DBSCAN implementations
│   ├── classification.py      # Random Forest, Logistic Regression
│   ├── metrics.py             # Internal and external metrics
│   ├── feature_analysis.py    # Feature relevance computation
│   └── plotting.py            # Visualization functions
├── data/
│   ├── Calcite-top30-sm-only-v1.1+.csv
│   ├── Calcite-effort-cov-only.csv
│   ├── Calcite-top30-sm-cov-effort.csv
│   └── ...
└── results/
    ├── supervised_learning_comparison.txt
    ├── classification/         # Per-dataset JSON results
    ├── calcite/dbscan/         # Clustering results
    └── ant-ivy/dbscan/
```

## Usage

### Supervised Learning (Recommended)

```bash
# Compare all three feature sets with Random Forest
python3 run_classification.py --compare-all

# Compare with Logistic Regression
python3 run_classification.py --compare-all --classifier lr

# Run on a single dataset
python3 run_classification.py --dataset calcite-effort-cov-only --classifier rf
```

### Clustering

```bash
# Run DBSCAN on Calcite dataset
python3 run_clustering.py --dataset calcite --algorithm dbscan

# Run on Ant-Ivy dataset
python3 run_clustering.py --dataset ant-ivy --algorithm dbscan

# With outlier removal
python3 run_clustering.py --dataset ant-ivy --algorithm dbscan --no-outliers

# K-Means with custom cluster count
python3 run_clustering.py --dataset ant-ivy --algorithm kmeans --k 3
```

### Compare Clustering with Effort Features

```bash
python3 compare_features.py --dataset calcite --algorithm dbscan
```

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

Dataset configurations are in `src/config.py`. Available datasets:

```python
DATASETS = {
    "calcite-top30-sm-only-v1.1+": {...},  # 30 SM features
    "calcite-effort-cov-only": {...},       # 31 effort+coverage features
    "calcite-top30-sm-cov-effort": {...},   # 61 combined features
    "ant-ivy": {...},
    "calcite": {...},
    # ... more datasets
}
```

## Practical Implications

1. **For defect prediction, use supervised learning** - clustering is not suitable due to class imbalance

2. **Prioritize effort + coverage data collection** over complex software metric extraction:
   - Test coverage (branch, instruction, complexity)
   - Code churn metrics (HASSAN_whcm, HASSAN_hcm)
   - File history (MOSER_weighted_age, MOSER_revisions)
   - Static analysis warnings (PMD severity)

3. **Software metrics alone are insufficient** - they capture code structure but not defect-proneness

## License

See [LICENSE](LICENSE) for details.
