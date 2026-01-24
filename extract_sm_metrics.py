"""
Extract SM_* (software metrics) columns from the Calcite Excel file.

This script reads the full Calcite dataset and extracts only:
- Metadata columns: Calcite version, ID, file, Version-ID, Bug
- All SM_* feature columns

Output: data/Calcite-SM-only.csv
"""

import pandas as pd


def main():
    # Configuration
    input_file = "data/All Calcite 1.0.0-1.15.0 software metrics.xlsx"
    sheet_name = "All SM"
    header_row = 9  # 0-indexed
    output_file = "data/Calcite-SM-only.csv"

    # Metadata columns to keep
    metadata_columns = ["Calcite version", "ID", "file", "Version-ID", "Bug"]

    print(f"Reading Excel file: {input_file}")
    df = pd.read_excel(input_file, sheet_name=sheet_name, header=header_row)

    print(f"Total columns in original file: {len(df.columns)}")
    print(f"Total rows: {len(df)}")

    # Find all SM_* columns
    sm_columns = [col for col in df.columns if str(col).startswith("SM_")]
    print(f"Found {len(sm_columns)} SM_* columns")

    # Check which metadata columns exist
    available_metadata = [col for col in metadata_columns if col in df.columns]
    missing_metadata = [col for col in metadata_columns if col not in df.columns]

    if missing_metadata:
        print(f"Warning: Missing metadata columns: {missing_metadata}")

    print(f"Available metadata columns: {available_metadata}")

    # Select columns to keep
    columns_to_keep = available_metadata + sm_columns
    df_subset = df[columns_to_keep]

    print(f"\nOutput dataset:")
    print(f"  Columns: {len(df_subset.columns)} ({len(available_metadata)} metadata + {len(sm_columns)} SM_*)")
    print(f"  Rows: {len(df_subset)}")

    # Save to CSV
    df_subset.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")

    # Show sample of SM_* column names
    print(f"\nFirst 10 SM_* columns: {sm_columns[:10]}")
    print(f"Last 10 SM_* columns: {sm_columns[-10:]}")


if __name__ == "__main__":
    main()
