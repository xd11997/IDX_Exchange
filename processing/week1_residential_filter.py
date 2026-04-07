"""
Week 1 Deliverable - IDX Exchange Internship
CRMLS residential filter.

What it does:
1) Reads one CSV file or all CSV files in a folder.
2) Filter residential listings.
3) Write comments showing row counts before and after filtering.

Typical usage:
    python crmls_clean_merge.py --input-dir /path/to/raw_csvs --output-dir /path/to/cleaned

Or single file:
    python crmls_clean_merge.py --input-file path/to/file --output-dir path/to/cleaned

"""
from __future__ import annotations
import argparse

import pandas as pd
import numpy as np
import os

def filter_residential_df(df):
    before = len(df)
    df_filtered = df[df["PropertyType"] == "Residential"]
    after = len(df_filtered)

    print(f"Before filter: {before} rows")
    print(f"After filter: {after} rows")

    return df_filtered

def process_file(input_path, output_path):
    print(f"\nProcessing: {input_path}")
    if os.path.isdir(output_path):
        filename = os.path.basename(input_path).replace(".csv", "_residential.csv")
        output_path = os.path.join(output_path, filename)

    df = pd.read_csv(input_path)
    df_filtered = filter_residential_df(df)
    df_filtered.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter Residential Properties")

    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")

    args = parser.parse_args()

    process_file(args.input, args.output)