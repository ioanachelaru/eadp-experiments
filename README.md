# eadp-experiments

Effort-aware defect prediction experiments using clustering analysis on software metrics.

## Overview

This project analyzes software metrics from Apache Ant-Ivy and Apache Calcite projects to identify which metrics are most relevant for predicting software defects. Using DBSCAN clustering with automatic epsilon detection, we evaluate feature importance based on cluster separation and compare results with effort-related metrics.

## Key Findings

| Dataset | Samples | Features | Clusters | Top Feature Category |
|---------|---------|----------|----------|---------------------|
| Ant-Ivy | 2,237 | 4,189 | 4 | SM_interface (88%) |
| Calcite | 19,751 | 5,234 | 65 | SM_enum (68%) |

- **Only 1 feature** appears in both top 100 lists (`SM_interface_nop_sum`)
- Software Metrics (SM_*) dominate top rankings in both datasets (96%)
- Effort-data features (PMD, HASSAN, MOSER) show minimal overlap with clustering results (~2% in top 100)

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
├── run_clustering.py          # Main clustering entry point
├── compare_features.py        # Compare clustering vs effort-data features
├── src/
│   ├── config.py              # Dataset and algorithm configurations
│   ├── data_utils.py          # Data loading and preprocessing
│   ├── clustering.py          # K-Means and DBSCAN implementations
│   ├── metrics.py             # Internal and external metrics
│   ├── feature_analysis.py    # Feature relevance computation
│   └── plotting.py            # Visualization functions
├── data/
│   ├── ant-ivy-all versions.xlsx
│   └── All Calcite 1.0.0-1.15.0 software metrics.xlsx
├── effort_data/
│   └── All Calcite 1.0.0-1.15.0 effort-related metrics.xlsx
└── results/
    ├── ant-ivy/dbscan/        # Ant-Ivy clustering results
    ├── calcite/dbscan/        # Calcite clustering results
    └── clustering_comparison_report.txt
```

## Usage

### Run DBSCAN Clustering

```bash
# Run on Ant-Ivy dataset
python3 run_clustering.py --dataset ant-ivy --algorithm dbscan

# Run on Calcite dataset
python3 run_clustering.py --dataset calcite --algorithm dbscan

# With outlier removal
python3 run_clustering.py --dataset ant-ivy --algorithm dbscan --no-outliers

# With custom parameters
python3 run_clustering.py --dataset ant-ivy --algorithm dbscan --eps 50 --min-samples 10
```

### Run K-Means Clustering

```bash
python3 run_clustering.py --dataset ant-ivy --algorithm kmeans --k 3
```

### Compare with Effort-Data Features

```bash
# Compare Ant-Ivy clustering results with effort-data features
python3 compare_features.py --dataset ant-ivy --algorithm dbscan

# Compare Calcite clustering results
python3 compare_features.py --dataset calcite --algorithm dbscan
```

## Output

### Clustering Results

```
results/{dataset}/{algorithm}/
├── results.json               # Full results with all feature rankings
├── top_features.txt           # Human-readable feature rankings
├── metrics_summary.txt        # Clustering metrics and cluster statistics
├── comparison_report.txt      # Comparison with effort-data features
└── visualizations/
    ├── clusters_pca.png       # PCA cluster visualization
    ├── feature_relevance.png  # Top features chart
    ├── metrics_comparison.png # Metrics bar chart
    └── k_distance.png         # K-distance graph (DBSCAN only)
```

### Comparison Report

```
results/clustering_comparison_report.txt   # Ant vs Calcite feature comparison
```

## Metrics Computed

### Internal Clustering Metrics

- **Silhouette Score**: Measures cluster cohesion vs separation. Range: -1 to 1, higher is better.
- **Davies-Bouldin Index**: Ratio of within-cluster to between-cluster distances. Lower is better.

### External Clustering Metrics

Using defect labels as ground truth:

- **Homogeneity**: Each cluster contains only members of a single class (0 to 1).
- **Completeness**: All members of a class are assigned to the same cluster (0 to 1).
- **V-Measure**: Harmonic mean of homogeneity and completeness (0 to 1).

### Feature Relevance

Features are ranked by the standard deviation across cluster centers. Higher values indicate features that better distinguish between clusters.

## Configuration

Edit `src/config.py` to modify datasets and algorithm defaults:

```python
DATASETS = {
    "ant-ivy": {
        "file": "data/ant-ivy-all versions.xlsx",
        "sheet": "ant-ivy-all versions",
        "header_row": 8,
        "feature_name_row": 7,
        "label_column": "Label",
    },
    "calcite": {
        "file": "data/All Calcite 1.0.0-1.15.0 software metrics.xlsx",
        "sheet": "All SM",
        "header_row": 9,
        "label_column": "Bug",
    },
}

CLUSTERING_DEFAULTS = {
    "kmeans": {"n_clusters": 2, "random_state": 42},
    "dbscan": {"eps": None, "min_samples": 5},  # eps auto-detected
}
```

## Results Summary

### Ant-Ivy DBSCAN Results
- **Clusters**: 4 (+ 1.5% noise)
- **Silhouette**: 0.507
- **Top features**: SM_class_nii_stdev, SM_interface_nii_*, SM_class_pda_stdev
- **Effort-data overlap**: 3/149 features in top 100 (2.0%)

### Calcite DBSCAN Results
- **Clusters**: 65 (+ 0.9% noise)
- **Silhouette**: 0.423
- **Top features**: SM_interface_nod_stdev, SM_enum_dloc_stdev, SM_enum_cloc_stdev
- **Effort-data overlap**: 4/170 features in top 100 (2.4%)

### Key Insight

The clustering analysis identifies **Software Metrics (SM_*)** as most important for cluster separation, while the effort-data correlation analysis prioritizes **PMD, HASSAN, and MOSER** metrics. The two approaches capture fundamentally different aspects of defect prediction.

## License

See [LICENSE](LICENSE) for details.
