"""
Extract SM_* (software metrics) columns from the Ant-Ivy Excel file.

This script reads the full Ant-Ivy dataset and extracts only:
- Metadata columns: Version, Id, Class, Label
- All SM_* feature columns (3624 features)

Output: data/Ant-Ivy-SM-only.csv
"""

import pandas as pd


def main():
    # Configuration
    input_file = "data/ant-ivy-all versions.xlsx"
    sheet_name = "ant-ivy-all versions"
    header_row = 8  # 0-indexed, row 9 in Excel (data header)
    feature_name_row = 7  # 0-indexed, row 8 in Excel (feature names)
    output_file = "data/Ant-Ivy-SM-only.csv"

    # Metadata columns to keep (these have names in header_row, not feature_name_row)
    metadata_columns = ["Version", "Id", "Class", "Label"]
    num_metadata_cols = 5  # First 5 columns are metadata

    print(f"Reading Excel file: {input_file}")

    # First read to get feature names from row 7
    df_names = pd.read_excel(input_file, sheet_name=sheet_name, header=feature_name_row, nrows=0)
    feature_names = list(df_names.columns)

    # Read the actual data with header at row 8
    df = pd.read_excel(input_file, sheet_name=sheet_name, header=header_row)
    data_header_names = list(df.columns)

    # Combine: keep metadata names from row 8, use feature names from row 7 for data columns
    combined_names = data_header_names[:num_metadata_cols] + feature_names[num_metadata_cols:]

    if len(combined_names) == len(df.columns):
        df.columns = combined_names
        print(f"Applied metadata names from row 9 + feature names from row 8 (Excel)")
    else:
        print(f"Warning: Combined name count ({len(combined_names)}) doesn't match column count ({len(df.columns)})")

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
