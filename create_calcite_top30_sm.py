#!/usr/bin/env python3
"""
Create Calcite dataset with only the top-30 SM features from clustering relevance.

This script:
1. Loads top 30 feature names from results/calcite/dbscan/results.json
2. Loads Calcite SM data from data/Calcite-SM-only.csv
3. Filters to keep only top 30 SM features that exist (excludes PMD_* features)
4. Outputs: data/Calcite-top30-sm-only.csv
"""

import json
import pandas as pd

# Load top 30 features from clustering results
with open("results/calcite/dbscan/results.json", "r") as f:
    results = json.load(f)

# Filter for SM_* features only, then take top 30
sm_features_all = [item for item in results["feature_relevance"] if item["feature"].startswith("SM_")]
top_30_sm = sm_features_all[:30]
sm_features = [item["feature"] for item in top_30_sm]
print(f"Top 30 SM features from clustering: {len(sm_features)}")
print(f"  (Filtered from {len(results['feature_relevance'])} total ranked features)")

# Load Calcite SM-only data
print("\nLoading Calcite-SM-only.csv...")
df = pd.read_csv("data/Calcite-SM-only.csv")
print(f"Original shape: {df.shape}")

# Identify metadata columns (non-feature columns)
metadata_cols = ["Calcite version", "ID", "file", "Version-ID", "Bug"]
print(f"Metadata columns: {metadata_cols}")

# Check which SM features exist in the dataset
available_features = [col for col in df.columns if col.startswith("SM_")]
print(f"Available SM_* features in dataset: {len(available_features)}")

# Find matching features
matching_features = [f for f in sm_features if f in df.columns]
missing_features = [f for f in sm_features if f not in df.columns]
print(f"\nMatching SM features: {len(matching_features)}")
if missing_features:
    print(f"Missing features: {missing_features}")

# Select metadata + matching features
selected_cols = metadata_cols + matching_features
df_filtered = df[selected_cols]
print(f"\nFiltered shape: {df_filtered.shape}")
print(f"  - Metadata columns: {len(metadata_cols)}")
print(f"  - Feature columns: {len(matching_features)}")

# Save to CSV
output_path = "data/Calcite-top30-sm-only.csv"
df_filtered.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")

# Summary
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Input: data/Calcite-SM-only.csv ({df.shape[0]} rows, {df.shape[1]} cols)")
print(f"Output: {output_path} ({df_filtered.shape[0]} rows, {df_filtered.shape[1]} cols)")
print(f"Features selected: {len(matching_features)} (top SM features from clustering)")
