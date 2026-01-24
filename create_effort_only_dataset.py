#!/usr/bin/env python3
"""
Create Calcite dataset with effort + coverage features only.

This script:
1. Loads Calcite combined data (top-30 SM + effort + coverage)
2. Filters to keep only effort features (26) + coverage features (5)
3. Outputs: data/Calcite-effort-cov-only.csv

Effort features (26 total):
- CHANGE_TYPE_* (3 features)
- HASSAN_* (5 features)
- ISSUE_* (2 features)
- MOSER_* (5 features)
- PMD_* (11 features)

Coverage features (5 total):
- COV_INSTRUCTION, COV_BRANCH, COV_LINE, COV_COMPLEXITY, COV_METHOD
"""

import pandas as pd

# Define feature prefixes
EFFORT_PREFIXES = ("CHANGE_TYPE_", "HASSAN_", "ISSUE_", "MOSER_", "PMD_")
COVERAGE_PREFIX = "COV_"
METADATA_COLS = ["Calcite version", "ID", "file", "Version-ID", "Bug"]

# Load source data
print("Loading Calcite-top30-sm-cov-effort.csv...")
df = pd.read_csv("data/Calcite-top30-sm-cov-effort.csv")
print(f"Original shape: {df.shape}")

# Identify effort and coverage columns
effort_cols = [col for col in df.columns if col.startswith(EFFORT_PREFIXES)]
coverage_cols = [col for col in df.columns if col.startswith(COVERAGE_PREFIX)]

print(f"\nEffort features found: {len(effort_cols)}")
for prefix in EFFORT_PREFIXES:
    count = sum(1 for col in effort_cols if col.startswith(prefix))
    print(f"  {prefix}*: {count}")

print(f"\nCoverage features found: {len(coverage_cols)}")
print(f"  {coverage_cols}")

# Select columns
selected_cols = METADATA_COLS + effort_cols + coverage_cols
df_filtered = df[selected_cols]

print(f"\nFiltered shape: {df_filtered.shape}")
print(f"  - Metadata columns: {len(METADATA_COLS)}")
print(f"  - Effort features: {len(effort_cols)}")
print(f"  - Coverage features: {len(coverage_cols)}")
print(f"  - Total features: {len(effort_cols) + len(coverage_cols)}")

# Save to CSV
output_path = "data/Calcite-effort-cov-only.csv"
df_filtered.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Input: data/Calcite-top30-sm-cov-effort.csv ({df.shape[0]} rows, {df.shape[1]} cols)")
print(f"Output: {output_path} ({df_filtered.shape[0]} rows, {df_filtered.shape[1]} cols)")
print(f"Features: {len(effort_cols)} effort + {len(coverage_cols)} coverage = {len(effort_cols) + len(coverage_cols)} total")
