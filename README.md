# eadp-experiments

Effort-aware defect prediction experiments using clustering analysis on software metrics.

## Overview

This project analyzes software metrics from Apache Ant and Apache Calcite projects to identify which metrics are most relevant for predicting software defects. Using K-Means clustering (k=2), we evaluate how well unsupervised clustering aligns with actual defect labels.

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Analysis (k=2)

```bash
python clustering_analysis.py                    # With all samples
python clustering_analysis.py --no-outliers      # Remove outliers (Z > 3.0)
```

### Multi-K Analysis (k=2,3,4,5)

```bash
python clustering_analysis.py --multi-k                    # Compare different k values
python clustering_analysis.py --multi-k --no-outliers      # Recommended for best results
```

Results will be saved to:
- `results/` - basic analysis with all samples
- `results_no_outliers/` - basic analysis with outlier removal
- `results_multi_k/` - multi-k analysis with all samples
- `results_multi_k_no_outliers/` - multi-k analysis with outlier removal (recommended)

## Data

The analysis uses data from:
- **File**: `data/All Calcite 1.0.0-1.15.0 effort-related metrics.xlsx`
- **Sheets**: `Ant_All` and `Calcite_All`

Each sheet contains software metrics for classes, with a `bug`/`bugs` column indicating defective instances.

## Output

### Basic Analysis Output

```
results/
├── ant_results.json           # Ant metrics (machine-readable)
├── calcite_results.json       # Calcite metrics (machine-readable)
├── analysis_report.txt        # Combined text report
└── visualizations/
    ├── ant_clusters.png           # PCA cluster visualization
    ├── calcite_clusters.png
    ├── ant_metrics_comparison.png # Metrics bar chart
    ├── calcite_metrics_comparison.png
    ├── ant_feature_relevance.png  # Top features by relevance
    └── calcite_feature_relevance.png
```

### Multi-K Analysis Output

```
results_multi_k_no_outliers/
├── findings_summary.txt       # Interpretation and recommendations
├── multi_k_comparison.txt     # Full comparison report
├── k_comparison_chart.png     # Visual comparison across k values
├── k2/
│   ├── ant_results.json       # Results for k=2
│   ├── calcite_results.json
│   └── visualizations/
├── k3/
│   └── ...                    # Results for k=3
├── k4/
│   └── ...
└── k5/
    └── ...
```

The `findings_summary.txt` includes:
- Recommended k value per dataset
- Cluster risk assessment (defect rate per cluster)
- Key differentiating metrics
- Practical recommendations for testing and code review

### Metrics Computed

#### Internal Clustering Metrics

- **Silhouette Score**: Measures how similar points are to their own cluster vs. other clusters. Range: -1 to 1, higher is better.
- **Davies-Bouldin Index**: Ratio of within-cluster to between-cluster distances. Lower is better.

#### External Clustering Metrics

Using defect labels as ground truth:

- **Homogeneity**: Each cluster contains only members of a single class (0 to 1).
- **Completeness**: All members of a class are assigned to the same cluster (0 to 1).
- **V-Measure**: Harmonic mean of homogeneity and completeness (0 to 1).

### Feature Relevance

Features are ranked by the absolute difference between cluster centroids (after standardization). Larger differences indicate features that better distinguish between clusters.

## Configuration

Edit the `CONFIG` dictionary in `clustering_analysis.py` to modify:

```python
CONFIG = {
    "data_file": "data/...",      # Path to Excel file
    "sheets": ["Ant_All", ...],   # Sheets to analyze
    "n_clusters": 2,              # Number of clusters
    "random_state": 42,           # Random seed for reproducibility
    "output_dir": "results",      # Output directory
    "top_features": 15,           # Number of top features to display
}
```

## License

See [LICENSE](LICENSE) for details.
